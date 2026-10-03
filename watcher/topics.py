"""Folder -> topics: what each system in a folder does, at one tree sha.

A folder (a "repo" in the monorepo) has 1..N topics: one per distinct model,
pipeline or tool. Per topic:

  description     LLM-written capability card, embedded          -> overlapping purpose
  keywords        technical keywords (separate LLM pass), TF-IDF -> shared rare techniques/entities
  inputs/outputs  tables it reads/writes                         -> upstream/downstream links

plus, per folder, every function's source embedded (evidence only).

Big folders are split into chunks (by file, then by top-level definitions) that fit
the model's context. Topics are extracted per chunk and then merged: the LLM decides
which chunk topics are the same system and writes the merged description; tables and
keywords are merged in code so nothing is lost.

Two folders are compared by their best-matching pair of topics. The same code builds
topics for folders on main and on PR branches, so they're directly comparable."""
import ast
import json
import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import requests
from tqdm import tqdm

from . import config, db, gitrepo, keywords, llm_cache

log = logging.getLogger("watcher")

# The capability-card wording was chosen with eval/prompt_lab.py on Qwen3-Coder-Next
# ("task_noreuse"); here it's applied per topic.
TOPICS_PROMPT = """You are cataloguing internal ML systems at an e-commerce company so that
engineers can find overlapping or reusable work.

Below is code from the folder "{folder}"{part}. A folder may hold one system or several
independent ones (separate models, pipelines or tools, each with its own business purpose).
Split it into topics: one topic per distinct model/pipeline/tool. Everything that serves a
system belongs to that system's topic: its feature engineering, training, evaluation or
backtesting, serving, lexicons/vocabularies, config and helpers. Two pieces of code are the
same topic if one feeds the other or they work toward the same final output; only split
when they produce independent outputs for different purposes. Most folders have exactly
one topic.

For each topic write a capability card of at most 120 words, plain prose, no code. Cover:
- what problem it solves, in business terms, and the standard name of the ML/data task
  (e.g. "near-duplicate detection", "item-to-item similarity", "text classification")
- the technique used
- its inputs (data sources/tables) and outputs (what it produces)
Describe what the code actually does, not what its names suggest.

Also list the data tables each topic reads and writes, copied exactly as written in the
code (including the part before the dot, no column lists).

Reply with JSON only:
{{"topics": [{{"name": "<3-6 word name>", "description": "<card>", "inputs": ["..."], "outputs": ["..."]}}]}}

{code}"""

KEYWORDS_PROMPT = """For each topic below, list the technical keywords that identify what its code
does, so it can be matched with other code doing the same job even if written differently.

Per topic give 10-15 keywords, lowercase, each 1-3 words. Include:
- the specific task (e.g. "near-duplicate detection", "tag normalization")
- algorithms, techniques, similarity measures, model types
- the domain entities it handles (e.g. reviews, skus, sessions, merchants)
- the signals or features it uses, and what it outputs

Do not include generic terms (machine learning, ml, ai, model, data, code, python,
function, system, pipeline, algorithm, e-commerce, business), library names, or
variable/function names.

Topics:
{topics}

Reply with JSON only, one key per topic name exactly as given:
{{"<topic name>": ["keyword", "..."]}}

Code:
{code}"""

MERGE_PROMPT = """These topics were extracted separately from different parts of the same folder
"{folder}". Some may describe the same model/pipeline/tool seen from different files (e.g.
one part has its training code, another its scoring code). Group topics that are the same
system; keep genuinely different systems separate. Helpers, lexicons, configs, evaluation
or backtesting code belong to the system they serve: merge them into it (a topic that
reads and writes no tables of its own is almost always such a helper). Every id must be
used exactly once.

For each final topic write a merged capability card of at most 120 words in the same style
(problem and standard task name, technique, inputs and outputs).

{topics}

Reply with JSON only:
{{"topics": [{{"name": "<3-6 word name>", "description": "<card>", "sources": ["<id>", "..."]}}]}}"""

MIN_FUNCTION_STATEMENTS = 2
README_CONTEXT_CHARS = 3000
MAX_MERGED_KEYWORDS = 25  # a short README is prepended to every chunk for context

_embedder = None
_lock = threading.RLock()


def embed(texts):
    global _embedder
    with _lock:
        if _embedder is None:
            from sentence_transformers import SentenceTransformer

            _embedder = SentenceTransformer(config.EMBED_MODEL)
        return _embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _chat(prompt, max_tokens):
    llm = config.llm()
    body = {
        "model": llm["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        # Qwen3 thinking mode is slow and not needed here.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    cache_request = {"url": llm["base_url"], **body}
    cached = llm_cache.get(cache_request)
    if cached is not None:
        return cached
    resp = requests.post(
        f"{llm['base_url']}/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {llm['api_key']}"} if llm["api_key"] else {},
        timeout=300,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    llm_cache.put(cache_request, text)
    return text


def _json(text):
    """First JSON object in an LLM reply, or None."""
    text = re.sub(r"^```\w*\s*|\s*```$", "", text.strip())
    start = text.find("{")
    if start < 0:
        return None
    try:
        return json.JSONDecoder().raw_decode(text[start:])[0]
    except json.JSONDecodeError:
        return None


def _normalize_table(name):
    name = re.sub(r"\(.*", "", name)  # drop "(col, col)" if the model added it anyway
    return name.strip(" `'\"").lower()


_TABLE = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*$")
_FILE_EXT = re.compile(r"\.(py|csv|json|parquet|pkl|npz|npy|txt|yaml|yml|sql|html|md|joblib|pt|bin)$")


def _tables(xs):
    """Only schema.table names: the model sometimes lists dataframes, files or model
    artifacts as inputs/outputs, which would break helper folding and dataflow links."""
    names = {_normalize_table(x) for x in xs or [] if isinstance(x, str) and x.strip()}
    return sorted(n for n in names if _TABLE.match(n) and not _FILE_EXT.search(n))


def _functions(files):
    """[(qualified_name, source)] for every non-trivial function in the folder's .py files."""
    out = []
    for path, src in files:
        if not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                body = [n for n in node.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
                if len(body) >= MIN_FUNCTION_STATEMENTS:
                    out.append((f"{path.split('/', 1)[-1]}:{node.name}", ast.get_source_segment(src, node)))
    return out


# --- chunking -----------------------------------------------------------------

def _split_file(path, text, limit):
    """Pieces of one file, each <= limit chars where possible: split at top-level
    definitions for Python, at blank lines otherwise, hard-split as a last resort."""
    if len(text) <= limit:
        return [(path, text)]
    lines = text.splitlines(keepends=True)
    cuts = []
    if path.endswith(".py"):
        try:
            defs = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            cuts = [n.lineno - 1 for n in ast.parse(text).body if isinstance(n, defs)]
        except SyntaxError:
            pass
    if not cuts:
        cuts = [i for i, line in enumerate(lines) if not line.strip()]
    offset = [0]
    for line in lines:
        offset.append(offset[-1] + len(line))
    bounds = sorted({0, len(lines), *cuts})
    pieces, start, last = [], 0, 0
    for b in bounds[1:]:
        if offset[b] - offset[start] > limit and last > start:
            pieces.append((start, last))
            start = last
        last = b
    pieces.append((start, len(lines)))
    out = []
    for i, (a, b) in enumerate(pieces, 1):
        body = "".join(lines[a:b])
        for j in range(0, len(body), limit):  # a single huge definition gets hard-split
            out.append((f"{path} (part {i}, lines {a + 1}-{b})", body[j:j + limit]))
    return out


def chunk_files(files, limit=None):
    """Groups (path, text) files into chunks of <= limit chars of code. Returns a list
    of chunk strings, each with "### path" headers, and a short README (if any)
    repeated in every chunk for context."""
    limit = limit or config.CHUNK_CHARS
    readme = next((t for p, t in files if p.lower().endswith("readme.md") and len(t) <= README_CONTEXT_CHARS), None)
    pieces = [piece for p, t in files for piece in _split_file(p, t, limit)]
    chunks, current, size = [], [], 0
    for path, text in pieces:
        block = f"### {path}\n{text}"
        if current and size + len(block) > limit:
            chunks.append(current)
            current, size = [], 0
        current.append(block)
        size += len(block)
    if current:
        chunks.append(current)
    if len(chunks) > 1 and readme:
        return [f"### README.md (folder overview)\n{readme}\n\n" + "\n\n".join(c) for c in chunks]
    return ["\n\n".join(c) for c in chunks]


# --- topic extraction -------------------------------------------------------------

def _extract(folder, code, part):
    """Topics (without embeddings) for one chunk: description pass + keywords pass."""
    raw = _chat(TOPICS_PROMPT.format(folder=folder, part=part, code=code), max_tokens=1600)
    parsed = _json(raw) or {}
    topics = []
    for t in parsed.get("topics") or []:
        if isinstance(t, dict) and str(t.get("description", "")).strip():
            topics.append({
                "name": str(t.get("name") or folder).strip()[:80],
                "description": str(t["description"]).strip(),
                "inputs": _tables(t.get("inputs")),
                "outputs": _tables(t.get("outputs")),
            })
    if not topics:  # unparseable reply: keep the text as a single topic rather than lose it
        log.warning("%s%s: topics reply wasn't valid JSON; using it as one topic", folder, part)
        topics = [{"name": folder, "description": re.sub(r"[{}\[\]\"]", " ", raw)[:1200], "inputs": [], "outputs": []}]

    names = "\n".join(f"- {t['name']}: {t['description'][:200]}" for t in topics)
    kw = _json(_chat(KEYWORDS_PROMPT.format(topics=names, code=code), max_tokens=900)) or {}
    lowered = {str(k).strip().lower(): v for k, v in kw.items()}
    for t in topics:
        values = kw.get(t["name"]) or lowered.get(t["name"].lower())
        if values is None and len(topics) == 1 and kw:
            values = next(iter(kw.values()))
        t["keywords"] = keywords.parse(", ".join(map(str, values)) if isinstance(values, list) else str(values or ""))
    return topics


def _merge(folder, chunk_topics):
    """Merges topics extracted from separate chunks of one folder."""
    flat = [(f"c{ci}t{ti}", t) for ci, ts in enumerate(chunk_topics) for ti, t in enumerate(ts)]
    if len(chunk_topics) == 1 or len(flat) == 1:
        return [t for _, t in flat]
    listing = "\n\n".join(
        f"[{tid}] {t['name']}\n{t['description']}\nreads: {', '.join(t['inputs']) or '-'}; writes: {', '.join(t['outputs']) or '-'}"
        for tid, t in flat
    )
    parsed = _json(_chat(MERGE_PROMPT.format(folder=folder, topics=listing), max_tokens=2000)) or {}
    by_id = dict(flat)
    used, merged = set(), []
    for t in parsed.get("topics") or []:
        ids = [str(x).strip().strip("[]").strip() for x in t.get("sources") or []]  # model may echo "[c0t1]"
        sources = [s for s in ids if s in by_id and s not in used]
        if not sources or not str(t.get("description", "")).strip():
            continue
        used.update(sources)
        src = [by_id[s] for s in sources]
        merged.append({
            "name": str(t.get("name") or src[0]["name"]).strip()[:80],
            "description": str(t["description"]).strip(),
            # Union in code: the LLM groups and rewrites, but tables and keywords
            # are never dropped by a merge.
            "inputs": sorted({x for s in src for x in s["inputs"]}),
            "outputs": sorted({x for s in src for x in s["outputs"]}),
            "keywords": list(dict.fromkeys(k for s in src for k in s["keywords"]))[:MAX_MERGED_KEYWORDS],
        })
    leftover = [t for tid, t in flat if tid not in used]  # ids the merge forgot: keep as-is
    if leftover:
        log.info("%s: merge left %d topic(s) ungrouped; keeping them", folder, len(leftover))
    return merged + leftover


def _fold_helpers(topics):
    """Deterministic safety net after the LLM: a topic that writes no tables of its own
    (feature builders, backtests, lexicons, helpers) is folded into the topic that
    produces output from the same inputs. Separate tools each write their own outputs,
    so they stay apart. Folded topics contribute their keywords and tables."""
    producers = [t for t in topics if t["outputs"]]
    if not producers or len(producers) == len(topics):
        return topics
    for t in [t for t in topics if not t["outputs"]]:
        inputs = set(t["inputs"])
        host = max(producers, key=lambda p: (len(inputs & set(p["inputs"])), len(p["description"])))
        if inputs and not inputs & set(host["inputs"]):
            continue  # reads something no producer reads: genuinely separate, keep it
        host["inputs"] = sorted(set(host["inputs"]) | inputs)
        host["keywords"] = list(dict.fromkeys(host["keywords"] + t["keywords"]))[:MAX_MERGED_KEYWORDS]
        topics = [x for x in topics if x is not t]
    return topics


def build_folder(folder, files):
    """Topics + function evidence for a folder from raw files, uncached."""
    chunks = chunk_files(files)
    n = len(chunks)
    parts = [f" (part {i} of {n}; the other parts are processed separately)" if n > 1 else "" for i in range(1, n + 1)]
    with ThreadPoolExecutor(min(n, config.LLM_CONCURRENCY)) as pool:
        chunk_topics = list(pool.map(lambda a: _extract(folder, *a), zip(chunks, parts)))
    topics = _fold_helpers(_merge(folder, chunk_topics))
    functions = _functions(files)
    emb = embed([t["description"] for t in topics])
    for t, e in zip(topics, emb):
        t["embedding"] = e
    func_emb = embed([src for _, src in functions]) if functions else np.zeros((0, 0), dtype=np.float32)
    return {
        "folder": folder, "topics": topics, "n_chunks": n,
        "func_names": [name for name, _ in functions], "func_embeddings": func_emb,
    }


def ensure_folders(conn, sha, folders):
    """{folder: entry} for {folder: tree_sha}; missing ones are built in parallel."""
    out = {f: db.get_folder(conn, t) for f, t in folders.items()}
    missing = [f for f, entry in out.items() if entry is None]
    if missing:
        with ThreadPoolExecutor(config.LLM_CONCURRENCY) as pool, tqdm(
            total=len(missing), desc="extracting topics", unit="folder", disable=len(missing) < 2 or not config.PROGRESS
        ) as bar:
            futures = {pool.submit(lambda f=f: build_folder(f, gitrepo.folder_files(sha, f))): f for f in missing}
            for fut in as_completed(futures):  # DB writes stay on this thread
                f = futures[fut]
                db.put_folder(conn, folders[f], fut.result())
                out[f] = db.get_folder(conn, folders[f])
                bar.update()
    return out


def folder_for(conn, sha, folder, tree_sha):
    """Cached by tree sha, so unchanged folders are never re-processed."""
    return ensure_folders(conn, sha, {folder: tree_sha})[folder]


# --- comparison -----------------------------------------------------------------

def rank_against_base(conn, base_sha, pr_folder, pr_entry):
    """Compares one folder's topics with every other folder on base. Returns rows
    sorted best-first, with rank and candidate flag."""
    base = ensure_folders(conn, base_sha, gitrepo.folders(base_sha))
    idf = keywords.TfidfIndex([t["keywords"] for e in base.values() for t in e["topics"]])
    rows = [
        {"repo_id": repo_id, **compare(pr_entry, other, idf)}
        for repo_id, other in base.items()
        if repo_id != pr_folder  # don't compare against its own pre-PR version
    ]
    rows.sort(key=lambda r: -r["score"])
    similar_seen = 0
    for rank, r in enumerate(rows, 1):
        r["rank"] = rank
        if r["dataflow"]:
            # Producer/consumer links are always surfaced and don't use up a top-K slot,
            # so a PR with several upstream tables can still surface its duplicates.
            r["candidate"] = 1
            continue
        similar_seen += 1
        r["candidate"] = int(similar_seen <= config.TOP_K and r["score"] >= config.MIN_CANDIDATE_SCORE)
    return rows


def _shared_tables(ours, theirs):
    """Matches on full name, or on the bare table name when the prefix differs
    (code often writes "product_embeddings" for "features.product_embeddings")."""
    bare = lambda t: t.rsplit(".", 1)[-1]
    theirs_bare = {bare(t): t for t in theirs}
    return {theirs_bare[bare(t)] for t in ours if bare(t) in theirs_bare}


def compare(pr_entry, other, idf):
    """Signals for one (PR folder, other folder) pair: the best-matching topic pair.
    idf: keywords.TfidfIndex fitted on the topics on base."""
    best = None
    for tp in pr_entry["topics"]:
        for to in other["topics"]:
            desc = float(tp["embedding"] @ to["embedding"])
            kw = idf.similarity(tp["keywords"], to["keywords"])
            # Mean of description and keyword similarity: best ranking and separation
            # on eval/prompt_lab.py. Function similarity is evidence only (chance-level
            # on realistic code: it matches shared boilerplate).
            s = (desc + kw) / 2
            if best is None or s > best[0]:
                best = (s, desc, kw, tp, to)
    score, desc_score, kw_score, tp, to = best
    kw_match = ", ".join(idf.shared(tp["keywords"], to["keywords"])[:5]) or None

    code_score, code_match = 0.0, None
    if len(pr_entry["func_names"]) and len(other["func_names"]):
        sims = pr_entry["func_embeddings"] @ other["func_embeddings"].T
        i, j = np.unravel_index(np.argmax(sims), sims.shape)
        code_score = float(sims[i, j])
        code_match = f"{pr_entry['func_names'][i]} ~ {other['func_names'][j]}"

    # Producer/consumer links across any topics; sharing an input (everyone reads
    # catalog.products) says nothing.
    ours_in = {x for t in pr_entry["topics"] for x in t["inputs"]}
    ours_out = {x for t in pr_entry["topics"] for x in t["outputs"]}
    theirs_in = {x for t in other["topics"] for x in t["inputs"]}
    theirs_out = {x for t in other["topics"] for x in t["outputs"]}
    reads_theirs = _shared_tables(ours_in, theirs_out)
    they_read_ours = _shared_tables(ours_out, theirs_in)
    if reads_theirs:
        dataflow = "upstream:" + ",".join(sorted(reads_theirs))
    elif they_read_ours:
        dataflow = "downstream:" + ",".join(sorted(they_read_ours))
    else:
        dataflow = None

    return {
        "score": score, "desc_score": desc_score, "kw_score": kw_score, "kw_match": kw_match,
        "pr_topic": tp["name"], "repo_topic": to["name"],
        "code_score": code_score, "code_match": code_match, "dataflow": dataflow,
    }
