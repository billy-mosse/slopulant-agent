"""Experiments on how systems are described and matched, on Qwen3-Coder-Next.

Every query has known duplicates among the folders on main (or none: negative).
For each query, folders on main are ranked by a similarity signal; reported:
  R@3     share of expected folders ranked in the top 3 (what the watcher hands on)
  MRR     mean reciprocal rank of expected folders
  sep     queries where every expected folder beats every wrong one
  neg     highest score any negative query reaches
  AUC     P(true pair scores above the best wrong folder of a random query)
  window  [best wrong score of negatives, worst true score]: thresholds in between
          separate perfectly; negative width = overlap

Queries: test PR folders, duplicates planted on main, the demo's "Try it" snippets,
eval/snippets (hand-written) and eval/snippets/gen (LLM rewrites + unrelated systems).

    python -m eval.prompt_lab prompts [variant ...]   # description prompts compared
    python -m eval.prompt_lab signals                 # description vs keyword signals
    python -m eval.prompt_lab signals --by-group      # + recall per query group
"""
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from watcher import config, keywords, topics

# Local checkout of the company repo (for eval only), next to this repo by default.
COMPANY_REPO = Path(os.environ.get("COMPANY_REPO_DIR", Path(__file__).resolve().parents[2] / "slopulant-monorepo"))
CACHE = config.DATA_DIR / "prompt_lab_cache.json"

# query -> (source, positives, ignored). Positives: folders on main the query
# duplicates (fully or partially). Ignored: "related" or producer/consumer folders,
# removed from the ranking pool (neither a hit nor a wrong answer).
QUERIES = {}
SNIPPETS = Path(__file__).parent / "snippets"


def _load_queries():
    from eval.build_cases import MANIFEST, SIMILARITY, similarity_truth

    cases = json.loads((Path(__file__).parent / "cases.json").read_text())
    for branch, folders in cases.items():
        if branch.startswith("_"):
            continue
        for folder, truth in folders.items():
            pos = [r for r, rel in truth.items() if set(rel) & set(SIMILARITY)]
            ignored = {r for r in truth if r not in pos}
            QUERIES[f"{folder}@{branch}"] = (branch, pos, ignored)

    # Duplicates planted among the systems already on main.
    planted = {s for c in MANIFEST["duplicate_clusters"] for s in c} | {s for p in MANIFEST["partial"] for s in p}
    for system in sorted(planted):
        t = similarity_truth(system)
        QUERIES[system] = ("main", sorted(set(t["duplicate"] + t["partial"]) - {system}), set())

    QUERIES["spam_filter_v2"] = ("preset", ["review_moderation"], {"fake_review_detection"})
    QUERIES["similar_products"] = ("preset", ["i2i_recs", "email_product_recs"], {"product_embeddings"})
    QUERIES["warehouse_slotting"] = ("preset", [], set())

    for directory, source in ((SNIPPETS, "snippet"), (SNIPPETS / "gen", "gen")):
        for name, truth in json.loads((directory / "expected.json").read_text()).items():
            if not name.startswith("_"):
                truth = truth if isinstance(truth, dict) else {"duplicate": truth}
                QUERIES[name] = (source, truth["duplicate"], set(truth.get("related", [])))


_load_queries()


def group(name):
    source = QUERIES[name][0]
    if source == "gen":
        return "gen:" + (name.split("__")[1] if "__" in name else "negative")
    if "/" in source:
        return "pr"
    return "main" if source == "main" else "hand"


BASE = """You are cataloguing internal ML systems at an e-commerce company so that
engineers can find overlapping or reusable work.

Below is the full code of one system. Write a capability card of at most 120 words,
plain prose, no code, no markdown. Cover:
- what problem it solves, in business terms
- the technique used
- its inputs (data sources/tables) and outputs (what it produces)
- what other systems could reuse its outputs

Describe what the code actually does, not what its names suggest.

System folder: {folder}

{code}"""

TERSE = """Describe the system below for a search index whose goal is to find other systems
that do the same job, even if they are written differently.

Write 50-80 words of plain prose. Start with "Computes". State exactly:
- the core computation (what is produced from what), using standard technical terms
- the method (algorithm, similarity measure, features, model type)
- the entity it operates on (products, reviews, customers, sessions, shipments...)

Do NOT mention business value, benefits, who could reuse it, or the company.
Do NOT use filler words like "efficiently", "robust", "leverages", "enables".
Describe what the code actually does, not what its names suggest.

System folder: {folder}

{code}"""

TAXONOMY = """Describe the system below for a search index whose goal is to find other systems
that do the same job, even if they are written differently and named differently.

Write 50-80 words of plain prose:
1. First sentence: the standard name of the ML/data task it performs (e.g. "near-duplicate
   detection", "item-to-item similarity", "text classification", "regression"), and the
   entity it operates on (products, reviews, customers, orders, shipments...).
2. Then the method: algorithm, representation, similarity measure, features.
3. Then 3-5 alternative names an engineer might give a system doing the same job.

Do NOT mention business value, benefits, or who could reuse it.
Describe what the code actually does, not what its names suggest.

System folder: {folder}

{code}"""

STRUCTURED = """Summarize the system below as a fingerprint used to detect other systems that do
the same job. Use exactly these lines, terse noun phrases, no full sentences:

Task: <standard name of the ML/data task>
Entity: <what it operates on: products, reviews, customers, ...>
Method: <algorithm, representation, similarity measure, model>
Features: <signals it uses>
Produces: <what it outputs, in plain words>
Also known as: <3-5 alternative names for a system doing the same job>

No business value, no benefits, no reuse suggestions. Describe what the code
does, not what its names suggest.

System folder: {folder}

{code}"""

FOCUSED = """You are cataloguing internal ML systems so that engineers can find other systems
that do the same job, even when they are written and named differently.

Below is the full code of one system. Write a capability card of 60-100 words, plain
prose, no code, no markdown. Cover:
- the task it performs, using the standard name for it (e.g. "near-duplicate detection",
  "item-to-item similarity", "text classification"), and the entity it operates on
- the technique: algorithm, representation, similarity measure, model type
- its inputs (data sources/tables) and outputs (what it produces)

Do not mention business value, benefits, or which other systems could use it.
Describe what the code actually does, not what its names suggest.

System folder: {folder}

{code}"""

COMPONENTS = FOCUSED.replace("""System folder: {folder}""", """Then write a line "Components:" followed by 2-6 lines, each starting with "- ", naming one
building block this code implements itself, in standard technical terms (e.g. "- hashed
bag-of-words text vectorizer", "- tag normalization with an alias map and a controlled
vocabulary", "- logistic regression scorer"). Only list things the code actually implements.

System folder: {folder}""")

NOREUSE = BASE.replace("""- what other systems could reuse its outputs
""", "")

TASK = BASE.replace("""- what problem it solves, in business terms""", """- what problem it solves, in business terms, and the standard name of the ML/data task
  (e.g. "near-duplicate detection", "item-to-item similarity", "text classification")""")

TASK_NOREUSE = TASK.replace("""- what other systems could reuse its outputs
""", "")

SPECIFIC = BASE.replace("""Describe what the code actually does, not what its names suggest.""", """Describe what the code actually does, not what its names suggest.
Be specific to this system: name the exact entity, signals and output. Do not start with
"This system", and avoid phrases that would fit any ML system (e.g. "helping the business",
"a simple model", "inputs come from", "can be reused by").""")

SPECIFIC_DECISION = SPECIFIC.replace("""- what problem it solves, in business terms""", """- the specific product feature or business decision it serves""")

_SPECIFIC_RULE = """
Be specific to this system: name the exact entity, signals and output. Do not start with
"This system", and avoid phrases that would fit any ML system (e.g. "helping the business",
"a simple model", "inputs come from")."""
_DOES = """Describe what the code actually does, not what its names suggest."""

TASK_SPECIFIC = TASK_NOREUSE.replace(_DOES, _DOES + _SPECIFIC_RULE)
TASK_REUSE_SPECIFIC = TASK.replace(_DOES, _DOES + _SPECIFIC_RULE)

VARIANTS = {
    "base": BASE, "terse": TERSE, "taxonomy": TAXONOMY, "structured": STRUCTURED,
    "focused": FOCUSED, "components": COMPONENTS,
    "task_specific": TASK_SPECIFIC, "task_reuse_specific": TASK_REUSE_SPECIFIC,
    "specific": SPECIFIC, "specific_decision": SPECIFIC_DECISION,
    "noreuse": NOREUSE, "task": TASK, "task_noreuse": TASK_NOREUSE,
}


def git(*args):
    return subprocess.run(["git", "-C", str(COMPANY_REPO), *args], capture_output=True, text=True, check=True).stdout


def folder_code(ref, folder):
    paths = git("ls-tree", "-r", "--name-only", ref, f"{folder}/").split()
    return "\n\n".join(f"### {p}\n{git('show', f'{ref}:{p}')}" for p in paths)


def preset_code(name):
    html = (Path(__file__).resolve().parents[1] / "watcher/static/index.html").read_text()
    return re.search(r'\["%s", `(.*?)`\]' % name, html, re.S).group(1)


def corpus():
    """(main folders, {name: (folder name shown to the LLM, code)})."""
    main_folders = [l.split("\t")[1] for l in git("ls-tree", "main").splitlines() if l.split()[1] == "tree"]
    docs = {f: (f, folder_code("main", f)) for f in main_folders}
    for name, (source, _, _) in QUERIES.items():
        folder = name.split("@")[0]
        if source == "gen":  # file names contain the answer; show a neutral folder name
            docs[name] = ("new_system", f"### new_system/main.py\n{(SNIPPETS / 'gen' / f'{name}.py').read_text()}")
        elif source == "snippet":
            docs[name] = (folder, f"### {folder}/{folder}.py\n{(SNIPPETS / f'{folder}.py').read_text()}")
        elif source == "preset":
            docs[name] = (folder, f"### {folder}/main.py\n{preset_code(folder)}")
        elif source != "main":
            docs[name] = (folder, folder_code(source, folder))
    return main_folders, docs


class Cache:
    def __init__(self):
        self.data = json.loads(CACHE.read_text()) if CACHE.exists() else {}
        self.lock = threading.Lock()

    def generate(self, prompt_name, prompt, docs, max_tokens=400):
        """{name: LLM output} for every doc, missing ones fetched in parallel."""
        key = lambda n: f"{config.llm()['model']}|{prompt_name}|{docs[n][0]}|{hashlib.sha1(docs[n][1].encode()).hexdigest()[:12]}"
        missing = [n for n in docs if key(n) not in self.data]

        def fetch(n):
            text = topics._chat(prompt.format(folder=docs[n][0], code=docs[n][1]), max_tokens=max_tokens)
            with self.lock:
                self.data[key(n)] = text

        if missing:
            with ThreadPoolExecutor(config.LLM_CONCURRENCY) as pool:
                list(tqdm(pool.map(fetch, missing), total=len(missing), desc=f"LLM: {prompt_name}", unit="call",
                          disable=not config.PROGRESS))
            CACHE.write_text(json.dumps(self.data))
        return {n: self.data[key(n)] for n in docs}


def evaluate(sim, main_folders):
    """sim(query, folder) -> score. Returns a metrics dict."""
    rr, hits, total, sep, n_pos, neg_best, pos_scores, wrong_scores = [], 0, 0, 0, 0, 0.0, [], []
    groups = {}
    for q, (source, expected, ignored) in QUERIES.items():
        own = q.split("@")[0]
        pool = [f for f in main_folders if f != own and f not in ignored]
        expected = [e for e in expected if e in pool]
        scores = {f: sim(q, f) for f in pool}
        order = sorted(pool, key=lambda f: -scores[f])
        wrong = max(scores[f] for f in pool if f not in expected)
        wrong_scores.append(wrong)
        if not expected:
            neg_best = max(neg_best, wrong)
            continue
        n_pos += 1
        sep += min(scores[e] for e in expected) > wrong
        for e in expected:
            rank = order.index(e) + 1
            rr.append(1 / rank)
            hits += rank <= 3
            total += 1
            pos_scores.append(scores[e])
            g = groups.setdefault(group(q), [0, 0])
            g[0] += rank <= 3
            g[1] += 1
    auc = np.mean([p > w for p in pos_scores for w in wrong_scores])
    return {
        "R@3": hits / total, "MRR": float(np.mean(rr)), "sep": f"{sep}/{n_pos}", "neg": neg_best,
        "AUC": float(auc), "window": (neg_best, min(pos_scores)), "groups": groups,
    }


def print_table(rows, by_group=False):
    print(f"{'':22s} {'R@3':>6s} {'MRR':>6s} {'sep':>7s} {'neg':>6s} {'AUC':>6s}  window")
    for name, m in rows:
        lo, hi = m["window"]
        print(f"{name:22s} {m['R@3']:6.3f} {m['MRR']:6.3f} {m['sep']:>7s} {m['neg']:6.2f} {m['AUC']:6.3f}"
              f"  [{lo:.2f}, {hi:.2f}]{' overlap' if hi <= lo else ''}")
        if by_group:
            print("    " + "  ".join(f"{g} {h}/{t}" for g, (h, t) in sorted(m["groups"].items())))


def split_files(code):
    """'### path' blocks (as built by corpus()) back into [(path, text)] files."""
    parts = re.split(r"(?m)^### (.+)\n", code)
    return [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)] or [("main.py", code)]


def topic_entries(docs):
    """{name: topics.build_folder(...)} for every doc; LLM calls are cached."""
    with ThreadPoolExecutor(config.LLM_CONCURRENCY) as pool:
        futures = {n: pool.submit(topics.build_folder, docs[n][0], split_files(docs[n][1])) for n in docs}
        return {n: f.result() for n, f in tqdm(futures.items(), desc="topics", unit="doc", disable=not config.PROGRESS)}


def desc_sim(texts):
    emb = dict(zip(texts, topics.embed(list(texts.values()))))
    return lambda q, f: float(emb[q] @ emb[f])


def run_prompts(variants):
    cache, (main_folders, docs) = Cache(), corpus()
    rows = []
    for v in variants:
        texts = cache.generate(v, VARIANTS[v], docs)
        rows.append((v, evaluate(desc_sim(texts), main_folders)))
    print_table(rows)


def run_signals(by_group):
    cache, (main_folders, docs) = Cache(), corpus()
    desc = cache.generate("task_noreuse", VARIANTS["task_noreuse"], docs)
    kws = {n: keywords.parse(t) for n, t in cache.generate("keywords", keywords.KEYWORDS_PROMPT, docs, 200).items()}
    d = desc_sim(desc)
    idf = keywords.TfidfIndex([kws[f] for f in main_folders])
    kw_idf = lambda q, f: idf.similarity(kws[q], kws[f])
    kw_emb_vec = dict(zip(kws, topics.embed([", ".join(k) for k in kws.values()])))
    kw_emb = lambda q, f: float(kw_emb_vec[q] @ kw_emb_vec[f])
    kw_jac = lambda q, f: len(set(kws[q]) & set(kws[f])) / (len(set(kws[q]) | set(kws[f])) or 1)
    funcs = {n: topics._functions([(f"x/{n}.py", code.split("\n", 1)[1]) for code in [docs[n][1]]]) for n in docs}
    func_emb = {n: topics.embed([src for _, src in fs]) if fs else None for n, fs in funcs.items()}

    def code(q, f):
        a, b = func_emb[q], func_emb[f]
        return float((a @ b.T).max()) if a is not None and b is not None else 0.0

    sem = lambda q, f: (d(q, f) + kw_idf(q, f)) / 2
    entries = topic_entries(docs)
    tidf = keywords.TfidfIndex([t["keywords"] for f in main_folders for t in entries[f]["topics"]])

    def topic_sim(q, f):  # what the watcher now does: best-matching topic pair
        return max(
            (float(a["embedding"] @ b["embedding"]) + tidf.similarity(a["keywords"], b["keywords"])) / 2
            for a in entries[q]["topics"] for b in entries[f]["topics"]
        )

    multi = sum(len(entries[f]["topics"]) > 1 for f in main_folders)
    chunked = [f for f in main_folders if entries[f]["n_chunks"] > 1]
    print(f"topics on main: {sum(len(entries[f]['topics']) for f in main_folders)} across {len(main_folders)} folders; "
          f"{multi} folders with >1 topic; chunked: {chunked or 'none'}")

    signals = {
        "topics: best pair [in use]": topic_sim,
        "description": d,
        "code (functions)": code,
        "max(desc, code) [current]": lambda q, f: max(d(q, f), code(q, f)),
        "kw_tfidf": kw_idf,
        "kw_embed": kw_emb,
        "kw_jaccard": kw_jac,
        "desc+kw_tfidf (mean)": lambda q, f: (d(q, f) + kw_idf(q, f)) / 2,
        "desc+kw_embed (mean)": lambda q, f: (d(q, f) + kw_emb(q, f)) / 2,
        "desc+0.5*kw_tfidf": lambda q, f: d(q, f) + 0.5 * kw_idf(q, f),
        "max(desc+kw mean, code)": lambda q, f: max(sem(q, f), code(q, f)),
        "mean(desc, kw, code)": lambda q, f: (d(q, f) + kw_idf(q, f) + code(q, f)) / 3,
        "desc+kw mean + 0.25*code": lambda q, f: sem(q, f) + 0.25 * code(q, f),
    }
    n_pos = sum(1 for _, e, _ in QUERIES.values() if e)
    print(f"model {config.llm()['model']} | {len(QUERIES)} queries ({n_pos} positive) | embed {config.EMBED_MODEL.split('/')[-1]}")
    print_table([(name, evaluate(sim, main_folders)) for name, sim in signals.items()], by_group)


def main():
    args = sys.argv[1:]
    mode = args.pop(0) if args and args[0] in ("prompts", "signals") else "signals"
    if mode == "prompts":
        run_prompts(args or ["base", "task_noreuse"])
    else:
        run_signals("--by-group" in args)


if __name__ == "__main__":
    main()
