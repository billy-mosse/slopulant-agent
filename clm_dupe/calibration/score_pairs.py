"""Score every (PR folder, main folder) pair in for_jesse.sqlite with CLM.

State layout and prompts are the reuse-detector's (clm-stack/reuse-detector), unchanged:
main folder = target (existing system), PR folder = candidate (new system), one topic each
(the best-matching topic pair the candidate generator picked).

    HF_HUB_OFFLINE=1 CLM_DEVICE=cpu CLM_CKPT=... ~/clm-venv/bin/python score_pairs.py
    -> clm_scores.sqlite (table clm_scores)
"""
import sqlite3
import sys
import time
from pathlib import Path

from clm import Engine

HERE = Path(__file__).resolve().parent
EVAL_DB = HERE.parent.parent / "eval" / "for_jesse.sqlite"

sys.path.insert(0, "/home/dell/repos/RINGUSB1/clm-stack/reuse-detector/eval")
from prompts import PROMPTS  # noqa: E402

QUESTIONS = {"reuse": PROMPTS["core_own|again/diff"],   # chosen prompt (infer.py)
             "baseline_v1": PROMPTS["baseline_v1"]}

src = sqlite3.connect(EVAL_DB)
src.row_factory = sqlite3.Row
topics = {(t["branch"], t["folder"], t["name"]): t for t in src.execute("SELECT * FROM topics")}


def pair_state(target_repo, t, cand_repo, c):
    return (f"Target repository: {target_repo}\nTopic: {t['name']}\n{t['description']}"
            f"\n\n\nCandidate repository: {cand_repo}\nTopic: {c['name']}\n{c['description']}")


out = sqlite3.connect(HERE / "clm_scores.sqlite")
out.execute("""CREATE TABLE IF NOT EXISTS clm_scores (branch TEXT, pr_folder TEXT, repo_id TEXT,
               question TEXT, p_yes REAL, PRIMARY KEY (branch, pr_folder, repo_id, question))""")
done = {tuple(r) for r in out.execute("SELECT DISTINCT branch, pr_folder, repo_id FROM clm_scores")}

engine = Engine(emb_url="http://127.0.0.1:8090/v1/embeddings")
pairs = src.execute("SELECT branch, pr_folder, repo_id, pr_topic, repo_topic FROM pairs").fetchall()
t0 = time.time()
for i, p in enumerate(pairs):
    key = (p["branch"], p["pr_folder"], p["repo_id"])
    if key in done:
        continue
    pr_t = topics[(p["branch"], p["pr_folder"], p["pr_topic"])]
    main_t = topics[("main", p["repo_id"], p["repo_topic"])]
    ans = engine.answer(pair_state(p["repo_id"], main_t, p["pr_folder"], pr_t), QUESTIONS)["answers"]
    out.executemany("INSERT OR REPLACE INTO clm_scores VALUES (?,?,?,?,?)",
                    [(*key, q, a["noul"]) for q, a in ans.items()])
    if i % 100 == 0:
        out.commit()
        print(f"{i}/{len(pairs)}  {time.time() - t0:.0f}s", flush=True)
out.commit()
print(f"done: {len(pairs)} pairs in {time.time() - t0:.0f}s")
