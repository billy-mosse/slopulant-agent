"""Folder -> card: everything we know about one system at one tree sha.

  description  LLM-written capability card, embedded           -> semantic overlap
  functions    each function's source, embedded                -> copy-paste / reimplementation
  keywords     technical keywords (separate LLM pass), TF-IDF  -> shared rare techniques/entities
  inputs/outputs  tables the code reads/writes, LLM-extracted  -> upstream/downstream links

The same code builds cards for folders on main and on PR branches, so they're
directly comparable."""
import ast
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from . import config, db, gitrepo, keywords, llm_cache

# Chosen with eval/prompt_lab.py on Qwen3-Coder-Next ("task_noreuse"): ties the
# original prompt on recall@3 but leaves a much wider gap between true matches and
# unrelated systems. Dropping "what could reuse it" stops cards from echoing each other.
CARD_PROMPT = """You are cataloguing internal ML systems at an e-commerce company so that
engineers can find overlapping or reusable work.

Below is the full code of one system. Write a capability card of at most 120 words,
plain prose, no code, no markdown. Cover:
- what problem it solves, in business terms, and the standard name of the ML/data task
  (e.g. "near-duplicate detection", "item-to-item similarity", "text classification")
- the technique used
- its inputs (data sources/tables) and outputs (what it produces)

Describe what the code actually does, not what its names suggest.

System folder: {folder}

{code}"""

DATAFLOW_PROMPT = """List the data tables this code reads and writes.
Copy each table name exactly as written in the code or docstrings, including the part
before the dot. No column lists. Reply with JSON only, no prose:
{{"inputs": ["..."], "outputs": ["..."]}}

{code}"""

MIN_FUNCTION_STATEMENTS = 2

_embedder = None
_lock = threading.RLock()


def embed(texts):
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer

        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)


def _chat(prompt, max_tokens):
    body = {
        "model": config.LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        # Qwen3 thinking mode is slow and not needed here.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    cache_request = {"url": config.LLM_BASE_URL, **body}
    cached = llm_cache.get(cache_request)
    if cached is not None:
        return cached
    resp = requests.post(
        f"{config.LLM_BASE_URL}/chat/completions",
        json=body,
        headers={"Authorization": f"Bearer {config.LLM_API_KEY}"} if config.LLM_API_KEY else {},
        timeout=300,
    )
    resp.raise_for_status()
    text = resp.json()["choices"][0]["message"]["content"]
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    llm_cache.put(cache_request, text)
    return text


def _normalize_table(name):
    name = re.sub(r"\(.*", "", name)  # drop "(col, col)" if the model added it anyway
    return name.strip(" `'\"").lower()


def _dataflow(code):
    raw = _chat(DATAFLOW_PROMPT.format(code=code), max_tokens=200)
    match = re.search(r"\{.*\}", raw, flags=re.S)
    try:
        parsed = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        parsed = {}
    clean = lambda xs: sorted({_normalize_table(x) for x in xs or [] if isinstance(x, str) and x.strip()})
    return clean(parsed.get("inputs")), clean(parsed.get("outputs"))


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


def build_card(folder, files):
    """Card from raw files, uncached. The three LLM calls run in parallel; only the
    embedder is serialized (shared between the watcher loop and the web demo)."""
    code = "\n\n".join(f"### {path}\n{text}" for path, text in files)
    with ThreadPoolExecutor(3) as pool:
        description = pool.submit(_chat, CARD_PROMPT.format(folder=folder, code=code), 400)
        dataflow = pool.submit(_dataflow, code)
        kw_text = pool.submit(_chat, keywords.KEYWORDS_PROMPT.format(code=code), 200)
    description, (inputs, outputs) = description.result(), dataflow.result()
    functions = _functions(files)
    with _lock:
        emb = embed([description])[0]
        func_emb = embed([src for _, src in functions]) if functions else np.zeros((0, 0), dtype=np.float32)
    return {
        "folder": folder, "description": description, "embedding": emb,
        "inputs": inputs, "outputs": outputs, "keywords": keywords.parse(kw_text.result()),
        "func_names": [name for name, _ in functions], "func_embeddings": func_emb,
    }


def ensure_cards(conn, sha, folders):
    """{folder: card} for {folder: tree_sha}; missing cards are built in parallel."""
    out = {f: db.get_card(conn, t) for f, t in folders.items()}
    missing = [f for f, card in out.items() if card is None]
    if missing:
        with ThreadPoolExecutor(config.LLM_CONCURRENCY) as pool:
            built = pool.map(lambda f: build_card(f, gitrepo.folder_files(sha, f)), missing)
            for f, c in zip(missing, built):
                db.put_card(conn, folders[f], c)
                out[f] = db.get_card(conn, folders[f])
    return out


def card_for(conn, sha, folder, tree_sha):
    """Cached by tree sha, so unchanged folders are never re-processed."""
    return ensure_cards(conn, sha, {folder: tree_sha})[folder]


def rank_against_base(conn, base_sha, pr_folder, pr_card):
    """Compares one folder's card with every other folder on base. Returns rows
    sorted best-first, with rank and candidate flag."""
    base = ensure_cards(conn, base_sha, gitrepo.folders(base_sha))
    idf = keywords.TfidfIndex([c["keywords"] for c in base.values()])
    rows = [
        {"repo_id": repo_id, **compare(pr_card, other, idf)}
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


def compare(pr_card, other, idf):
    """Signals for one (PR folder, other folder) pair. idf: keywords.TfidfIndex
    fitted on the folders on base."""
    card_score = float(pr_card["embedding"] @ other["embedding"])
    kw_score = idf.similarity(pr_card["keywords"], other["keywords"])
    kw_match = ", ".join(idf.shared(pr_card["keywords"], other["keywords"])[:5]) or None

    code_score, code_match = 0.0, None
    if len(pr_card["func_names"]) and len(other["func_names"]):
        sims = pr_card["func_embeddings"] @ other["func_embeddings"].T
        i, j = np.unravel_index(np.argmax(sims), sims.shape)
        code_score = float(sims[i, j])
        code_match = f"{pr_card['func_names'][i]} ~ {other['func_names'][j]}"

    # Only producer/consumer links count; sharing an input (everyone reads
    # catalog.products) says nothing.
    reads_theirs = _shared_tables(pr_card["inputs"], other["outputs"])
    they_read_ours = _shared_tables(pr_card["outputs"], other["inputs"])
    if reads_theirs:
        dataflow = "upstream:" + ",".join(sorted(reads_theirs))
    elif they_read_ours:
        dataflow = "downstream:" + ",".join(sorted(they_read_ours))
    else:
        dataflow = None

    # Mean of description and keyword similarity: best ranking and separation on the
    # 185-query eval (eval/prompt_lab.py). Function-level code similarity is kept as
    # evidence only: on realistic code it matches shared boilerplate (SQL loading,
    # argparse mains) and scored no better than chance (AUC 0.50). Dataflow links are
    # surfaced separately (see rank_against_base).
    score = (card_score + kw_score) / 2
    return {
        "score": score, "card_score": card_score, "kw_score": kw_score, "kw_match": kw_match,
        "code_score": code_score, "code_match": code_match, "dataflow": dataflow,
    }
