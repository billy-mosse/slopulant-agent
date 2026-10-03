"""Search CLM prompt variants (state layout x question) on the 33 non-dataflow candidate pairs.

Selection is evaluated with nested leave-one-PR-out: for each held-out PR branch, the best
variant (by AUC) and its full-recall threshold are picked on the other branches only, then
applied to the held-out branch.  The pooled result estimates how "search, then ship the
winner" generalises; the in-sample leaderboard alone would overstate it.

    HF_HUB_OFFLINE=1 CLM_DEVICE=cpu CLM_CKPT=... ~/clm-venv/bin/python prompt_search.py
"""
import json
import sqlite3
import time
from pathlib import Path

from clm import Engine

HERE = Path(__file__).resolve().parent
EVAL_DB = HERE.parent.parent / "eval" / "for_jesse.sqlite"

db = sqlite3.connect(EVAL_DB)
db.row_factory = sqlite3.Row
topics = {(t["branch"], t["folder"], t["name"]): t for t in db.execute("SELECT * FROM topics")}
S = [dict(r) for r in db.execute(
    "SELECT * FROM pairs WHERE candidate = 1 AND is_related = 0 AND dataflow IS NULL")]


# --------------------------------------------------------------------------- state layouts

def block(label, folder, t, kw=False, io=False):
    s = f"{label}: {folder}\nTopic: {t['name']}\n{t['description']}"
    if kw:
        s += "\nKeywords: " + ", ".join(json.loads(t["keywords"]))
    if io:
        s += ("\nReads: " + (", ".join(json.loads(t["inputs"])) or "-")
              + "\nWrites: " + (", ".join(json.loads(t["outputs"])) or "-"))
    return s


def layout(name):
    kw = name in ("kw", "kw_io", "sys_kw")
    io = name == "kw_io"
    if name.startswith("sys"):
        a, b = "Existing system on main", "New system in a pull request"
    else:
        a, b = "Target repository", "Candidate repository"
    swap = name == "swap"

    def make(p):
        main_t = topics[("main", p["repo_id"], p["repo_topic"])]
        pr_t = topics[(p["branch"], p["pr_folder"], p["pr_topic"])]
        x, y = block(a, p["repo_id"], main_t, kw, io), block(b, p["pr_folder"], pr_t, kw, io)
        return f"{y}\n\n\n{x}" if swap else f"{x}\n\n\n{y}"
    return make


LAYOUTS = ["base", "kw", "kw_io", "swap", "sys", "sys_kw"]


# --------------------------------------------------------------------------- questions
# (question, scorer): noul -> p(yes); choice -> summed probability of the "duplicate" labels

def noul(q, t, f):
    return {"type": "noul", "instructions": q, "criteria": {"true": t, "false": f}}


def choice(q, crit, pos):
    return {"type": "choice", "instructions": q, "criteria": crit, "_pos": pos}


QUESTIONS = {
    "reuse": noul("Is the core building block of the candidate topic their own copy of one the target topic already provides?",
                  "The same building block, built again.", "A different building block."),
    "base_v1": noul("Is the core building block of the candidate topic one the target topic already provides?",
                    "The same building block.", "A different building block."),
    "same_job": noul("Does the new system do the same job as the existing system?",
                     "The same job, done again.", "A different job."),
    "reimpl": noul("Is the pull request re-implementing functionality the existing system already provides?",
                   "It re-implements existing functionality.", "It builds something the existing system does not provide."),
    "could_use": noul("Could the team behind the new system have used or extended the existing system instead of building their own?",
                      "Yes, the existing system already covers it.", "No, the existing system does not cover it."),
    "same_problem": noul("Do these two systems solve the same problem, even if they use different techniques?",
                         "The same problem.", "Different problems."),
    "dupes": noul("Are these two systems duplicates of each other?",
                  "Duplicates.", "Not duplicates."),
    "rel3": choice("How does the new system relate to the existing system?",
                   {"duplicate": "It does the same job as the existing system.",
                    "partial": "It re-implements part of what the existing system does.",
                    "different": "It does a different job."}, ["duplicate", "partial"]),
    "rel3_tech": choice("How does the new system relate to the existing system? Ignore differences in code, naming and technique.",
                        {"duplicate": "Same problem solved again, possibly with a different technique.",
                         "partial": "Rebuilds a component the existing system already has.",
                         "different": "Solves a different problem."}, ["duplicate", "partial"]),
}


def clean(q):
    return {k: v for k, v in q.items() if not k.startswith("_")}


def score_of(q, ans):
    if q["type"] == "noul":
        return ans["noul"]
    probs = ans["probabilities"]
    return sum(probs[k] for k in q["_pos"])


# --------------------------------------------------------------------------- run (cached)

out = sqlite3.connect(HERE / "prompt_search.sqlite")
out.execute("""CREATE TABLE IF NOT EXISTS scores (layout TEXT, question TEXT, branch TEXT, pr_folder TEXT,
               repo_id TEXT, score REAL, PRIMARY KEY (layout, question, branch, pr_folder, repo_id))""")
have = {tuple(r) for r in out.execute("SELECT layout, question, branch, pr_folder, repo_id FROM scores")}
engine = Engine(emb_url="http://127.0.0.1:8090/v1/embeddings")
t0 = time.time()
for L in LAYOUTS:
    make = layout(L)
    for p in S:
        todo = {k: q for k, q in QUESTIONS.items() if (L, k, p["branch"], p["pr_folder"], p["repo_id"]) not in have}
        if not todo:
            continue
        ans = engine.answer(make(p), {k: clean(q) for k, q in todo.items()})["answers"]
        out.executemany("INSERT INTO scores VALUES (?,?,?,?,?,?)",
                        [(L, k, p["branch"], p["pr_folder"], p["repo_id"], score_of(q, ans[k])) for k, q in todo.items()])
    out.commit()
    print(f"layout {L} done  {time.time() - t0:.0f}s", flush=True)
