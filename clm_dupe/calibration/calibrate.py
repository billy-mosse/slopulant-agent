"""Calibrate CLM's "surface this to the author" threshold on for_jesse.sqlite.

Decision rule under test (per for_jesse_readme.md "What to tune"):
  - candidate pair with a dataflow link  -> upstream/downstream from the tables (no CLM)
  - candidate pair without one           -> duplicate if CLM score >= THRESHOLD, else unrelated

Threshold = highest value that keeps every positive (max precision at full recall), picked on
candidate = 1 AND is_related = 0 AND dataflow IS NULL.  Leave-one-PR-out estimates how a
threshold picked this way does on a PR it has not seen.
"""
import sqlite3
import statistics
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
EVAL_DB = HERE.parent.parent / "eval" / "for_jesse.sqlite"

db = sqlite3.connect(EVAL_DB)
db.row_factory = sqlite3.Row
db.execute("ATTACH ? AS s", (str(HERE / "clm_scores.sqlite"),))

rows = [dict(r) for r in db.execute("""
    SELECT p.*, r.p_yes AS reuse, b.p_yes AS base_v1, d.relation AS dummy
    FROM pairs p
    JOIN s.clm_scores r ON (r.branch, r.pr_folder, r.repo_id, r.question) = (p.branch, p.pr_folder, p.repo_id, 'reuse')
    JOIN s.clm_scores b ON (b.branch, b.pr_folder, b.repo_id, b.question) = (p.branch, p.pr_folder, p.repo_id, 'baseline_v1')
    LEFT JOIN dummy_classifier d USING (branch, pr_folder, repo_id)""")]

# centering (reuse-detector style): subtract the PR folder's median over all its main-folder pairs
by_pr = defaultdict(list)
for r in rows:
    if not r["is_related"]:
        by_pr[(r["branch"], r["pr_folder"])].append(r)
for grp in by_pr.values():
    for q in ("reuse", "base_v1"):
        med = statistics.median(r[q] for r in grp)
        for r in grp:
            r[q + "_c"] = r[q] - med

S = [r for grp in by_pr.values() for r in grp if r["candidate"] and r["dataflow"] is None]
SCORERS = {"CG score (baseline)": "score", "CLM reuse, raw": "reuse", "CLM reuse, centered": "reuse_c",
           "CLM baseline_v1, raw": "base_v1", "CLM baseline_v1, centered": "base_v1_c"}


def auc(pos, neg):
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def full_recall_th(rs, k):
    return min(r[k] for r in rs if r["is_positive"])


def confusion(rs, k, th):
    tp = sum(r[k] >= th and r["is_positive"] for r in rs)
    fp = sum(r[k] >= th and not r["is_positive"] for r in rs)
    fn = sum(r[k] < th and r["is_positive"] for r in rs)
    return tp, fp, fn


def fmt(tp, fp, fn):
    prec = tp / (tp + fp) if tp + fp else float("nan")
    return f"recall {tp}/{tp + fn}  precision {tp}/{tp + fp} = {prec:.2f}"


npos = sum(r["is_positive"] for r in S)
print(f"non-dataflow candidates: {len(S)} ({npos} duplicates, {len(S) - npos} none)\n")

print("== separation and in-sample full-recall threshold")
print(f"{'scorer':28s} {'AUC':>5s} {'threshold':>9s}  at that threshold")
for name, k in SCORERS.items():
    pos = [r[k] for r in S if r["is_positive"]]
    neg = [r[k] for r in S if not r["is_positive"]]
    th = full_recall_th(S, k)
    print(f"{name:28s} {auc(pos, neg):5.3f} {th:9.3f}  {fmt(*confusion(S, k, th))}")

print("\n== leave-one-PR-out (threshold picked on the other PRs, applied to the held-out one)")
branches = sorted({r["branch"] for r in S})
for name, k in SCORERS.items():
    tp = fp = fn = 0
    ths = []
    for b in branches:
        train = [r for r in S if r["branch"] != b]
        test = [r for r in S if r["branch"] == b]
        th = full_recall_th(train, k)
        ths.append(th)
        a, c, d = confusion(test, k, th)
        tp, fp, fn = tp + a, fp + c, fn + d
    print(f"{name:28s} {fmt(tp, fp, fn)}   thresholds {min(ths):.3f}..{max(ths):.3f}")

print("\n== dummy LLM classifier on the same pairs (surfaced = relation != unrelated)")
tp = sum(r["dummy"] not in (None, "unrelated") and r["is_positive"] for r in S)
fp = sum(r["dummy"] not in (None, "unrelated") and not r["is_positive"] for r in S)
fn = sum(r["dummy"] in (None, "unrelated") and r["is_positive"] for r in S)
print(f"{'dummy classifier':28s} {fmt(tp, fp, fn)}")

for k in ("reuse", "reuse_c"):
    print(f"\n== per-pair scores, {k} (sorted)")
    for r in sorted(S, key=lambda r: -r[k]):
        print(f"  {'DUP ' if r['is_positive'] else '    '}{r[k]:+.3f}  cg={r['score']:.3f}  "
              f"{r['pr_folder']:22s} vs {r['repo_id']:26s} dummy={r['dummy']}")

print("\n== positives the candidate generator misses (candidate = 0): CLM rank within their PR")
for r in rows:
    if r["is_positive"] and not r["candidate"]:
        grp = sorted(by_pr[(r["branch"], r["pr_folder"])], key=lambda x: -x["reuse_c"])
        rank = next(i for i, x in enumerate(grp, 1) if x is r)
        print(f"  {r['pr_folder']} vs {r['repo_id']} ({r['truth']}): cg={r['score']:.3f}  "
              f"reuse={r['reuse']:.3f}  centered={r['reuse_c']:+.3f}  CLM rank {rank}/{len(grp)}")
