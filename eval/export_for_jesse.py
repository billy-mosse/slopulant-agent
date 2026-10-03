"""Exports a self-contained SQLite file for tuning the classifier (CLM) without
re-generating any topics: every topic on main and on every test PR branch, every
(PR folder, folder on main) pair with the candidate generator's signals, and the
ground-truth relation for each pair. See for_jesse_readme.md.

    python -m eval.export_for_jesse                      # -> eval/for_jesse.sqlite
    python -m eval.export_for_jesse --out /tmp/x.sqlite --no-dummy-classifier

Uses whatever LLM profile is active (LLM_PROFILE / dashboard switch); topics already
in the watcher's cache are reused, so on a warm DATA_DIR this only computes what's
missing.
"""
import argparse
import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

from eval.build_cases import MANIFEST, POSITIVE
from eval.run_eval import CASES, score_branch
from watcher import classifier, config, db, gitrepo, topics

OUT = Path(__file__).parent / "for_jesse.sqlite"

SCHEMA = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);

-- One row per folder: the 41 systems on main, and every folder a test PR touches.
CREATE TABLE folders (
    source    TEXT NOT NULL,   -- 'main' | 'pr'
    branch    TEXT NOT NULL,   -- 'main' or the test PR branch
    folder    TEXT NOT NULL,
    tree_sha  TEXT NOT NULL,
    n_chunks  INTEGER NOT NULL,
    owner     TEXT,            -- main: author with most commits in the folder; pr: the PR author
    PRIMARY KEY (branch, folder)
);

-- 1..N topics per folder (one per model / pipeline / tool), exactly as the watcher sees them.
CREATE TABLE topics (
    branch      TEXT NOT NULL,
    folder      TEXT NOT NULL,
    idx         INTEGER NOT NULL,
    name        TEXT NOT NULL,
    description TEXT NOT NULL,
    keywords    TEXT NOT NULL,   -- JSON list
    inputs      TEXT NOT NULL,   -- JSON list of tables read
    outputs     TEXT NOT NULL,   -- JSON list of tables written
    embedding   BLOB NOT NULL,   -- float32[384], all-MiniLM-L6-v2 of the description, L2-normalised
    PRIMARY KEY (branch, folder, idx)
);

-- Every (PR folder, folder on main) pair. Signals come from the best-matching topic pair.
CREATE TABLE pairs (
    branch      TEXT NOT NULL,
    pr_folder   TEXT NOT NULL,
    repo_id     TEXT NOT NULL,   -- folder on main
    truth       TEXT NOT NULL,   -- comma-joined: duplicate, partial, upstream, downstream, related; or 'none'
    is_positive INTEGER NOT NULL,-- 1 if truth has duplicate/partial/upstream/downstream ('related' = 0, see readme)
    is_related  INTEGER NOT NULL,-- 1 if truth is only 'related' (ambiguous: exclude when scoring)
    score       REAL NOT NULL,   -- (desc_score + kw_score) / 2 of the best topic pair
    desc_score  REAL NOT NULL,   -- cosine of the two topics' description embeddings
    kw_score    REAL NOT NULL,   -- TF-IDF cosine of the two topics' keywords
    kw_match    TEXT,            -- shared keywords, rarest first
    pr_topic    TEXT NOT NULL,   -- topics.name on the PR side (branch, pr_folder)
    repo_topic  TEXT NOT NULL,   -- topics.name on the main side ('main', repo_id)
    dataflow    TEXT,            -- 'upstream:<tables>' | 'downstream:<tables>' | NULL
    sim_rank    INTEGER,         -- rank by score among non-dataflow pairs of this PR folder (1 = best)
    candidate   INTEGER NOT NULL,-- 1 if the watcher hands it to the classifier (dataflow, or top-K above the floor)
    PRIMARY KEY (branch, pr_folder, repo_id)
);

-- Baseline to beat: the current dummy LLM classifier, on candidate pairs only.
CREATE TABLE dummy_classifier (
    branch     TEXT NOT NULL,
    pr_folder  TEXT NOT NULL,
    repo_id    TEXT NOT NULL,
    relation   TEXT NOT NULL,
    confidence REAL NOT NULL,
    reason     TEXT NOT NULL,
    PRIMARY KEY (branch, pr_folder, repo_id)
);

CREATE TABLE prs (branch TEXT PRIMARY KEY, title TEXT, author TEXT, head_sha TEXT);
"""


def write_folder(out, branch, source, folder, tree_sha, entry, owner):
    out.execute("INSERT OR REPLACE INTO folders VALUES (?, ?, ?, ?, ?, ?)",
                (source, branch, folder, tree_sha, entry["n_chunks"], owner))
    for i, t in enumerate(entry["topics"]):
        out.execute("INSERT OR REPLACE INTO topics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (
            branch, folder, i, t["name"], t["description"], json.dumps(t["keywords"]),
            json.dumps(t["inputs"]), json.dumps(t["outputs"]), np.asarray(t["embedding"], dtype=np.float32).tobytes()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--no-dummy-classifier", action="store_true")
    args = ap.parse_args()
    started = time.time()

    conn = db.connect()
    gitrepo.ensure_clone()
    base_sha = gitrepo.fetch_base()
    branches = gitrepo.fetch_branches()
    base_folders = gitrepo.folders(base_sha)
    main_entries = topics.ensure_folders(conn, base_sha, base_folders)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        args.out.unlink()
    out = sqlite3.connect(args.out)
    out.executescript(SCHEMA)
    llm = config.llm()
    meta = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"), "llm": llm["label"], "llm_model": llm["model"],
        "embed_model": config.EMBED_MODEL, "repo": config.GITHUB_REPO, "base_sha": base_sha,
        "top_k": config.TOP_K, "min_candidate_score": config.MIN_CANDIDATE_SCORE,
        "scoring": "score = (desc_score + kw_score) / 2 of the best topic pair; candidates = dataflow links + top-K by score >= floor",
        "classifier_baseline": classifier.NAME,
    }
    out.executemany("INSERT INTO meta VALUES (?, ?)", [(k, str(v)) for k, v in meta.items()])

    for folder, tree_sha in base_folders.items():
        write_folder(out, "main", "main", folder, tree_sha, main_entries[folder], gitrepo.owner(base_sha, folder))

    pr_entries, jobs = {}, []
    for branch, truths in tqdm(CASES.items(), desc="scoring test branches", unit="branch", disable=not config.PROGRESS):
        if branch not in branches:
            print(f"skipping {branch}: branch not found")
            continue
        head = branches[branch]
        spec = MANIFEST["prs"].get(branch, {})
        out.execute("INSERT OR REPLACE INTO prs VALUES (?, ?, ?, ?)",
                    (branch, spec.get("title"), spec.get("author") or gitrepo.commit_author(head), head))
        by_folder = score_branch(conn, base_sha, head)
        head_folders = gitrepo.folders(head)
        for pr_folder, rows in by_folder.items():
            entry = topics.folder_for(conn, head, pr_folder, head_folders[pr_folder])
            pr_entries[(branch, pr_folder)] = entry
            write_folder(out, branch, "pr", pr_folder, head_folders[pr_folder], entry, gitrepo.commit_author(head))
            truth = truths.get(pr_folder, {})
            sim_rank = 0
            for r in rows:
                rel = truth.get(r["repo_id"], [])
                if not r["dataflow"]:
                    sim_rank += 1
                out.execute("INSERT OR REPLACE INTO pairs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (
                    branch, pr_folder, r["repo_id"], ",".join(rel) or "none",
                    int(bool(set(rel) & set(POSITIVE))), int(rel == ["related"]),
                    r["score"], r["desc_score"], r["kw_score"], r["kw_match"], r["pr_topic"], r["repo_topic"],
                    r["dataflow"], None if r["dataflow"] else sim_rank, r["candidate"]))
                if r["candidate"]:
                    jobs.append((branch, pr_folder, r))
    out.commit()

    if not args.no_dummy_classifier and jobs:
        def judge(job):
            branch, pr_folder, r = job
            a = pr_entries[(branch, pr_folder)]
            b = main_entries[r["repo_id"]]
            return classifier.classify(classifier.as_input(a, classifier.topic_by_name(a, r["pr_topic"])),
                                       classifier.as_input(b, classifier.topic_by_name(b, r["repo_topic"])), r)

        with ThreadPoolExecutor(config.LLM_CONCURRENCY) as pool:
            verdicts = list(tqdm(pool.map(judge, jobs), total=len(jobs), desc="dummy classifier", disable=not config.PROGRESS))
        out.executemany("INSERT OR REPLACE INTO dummy_classifier VALUES (?, ?, ?, ?, ?, ?)", [
            (b, f, r["repo_id"], v["relation"], v["confidence"], v["reason"]) for (b, f, r), v in zip(jobs, verdicts)])
        out.commit()

    counts = {t: out.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
              for t in ("folders", "topics", "pairs", "dummy_classifier", "prs")}
    positives = out.execute("SELECT count(*) FROM pairs WHERE is_positive = 1").fetchone()[0]
    out.execute("INSERT OR REPLACE INTO meta VALUES ('seconds_to_build', ?)", (f"{time.time() - started:.0f}",))
    out.commit()
    print(f"wrote {args.out}: {counts}, {positives} positive pairs, {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
