"""Leaderboard + nested leave-one-PR-out for prompt_search.py results."""
import sqlite3
from pathlib import Path
from collections import defaultdict

HERE = Path(__file__).resolve().parent
EVAL_DB = HERE.parent.parent / "eval" / "for_jesse.sqlite"

db = sqlite3.connect(EVAL_DB)
db.execute("ATTACH ? AS s", (str(HERE / "prompt_search.sqlite"),))
label = {(b, f, r): pos for b, f, r, pos in db.execute(
    "SELECT branch, pr_folder, repo_id, is_positive FROM pairs WHERE candidate = 1 AND is_related = 0 AND dataflow IS NULL")}
cg = {(b, f, r): s for b, f, r, s in db.execute("SELECT branch, pr_folder, repo_id, score FROM pairs")}

V = defaultdict(dict)   # variant -> {pair: score}
for L, q, b, f, r, s in db.execute("SELECT * FROM s.scores"):
    V[f"{L} | {q}"][(b, f, r)] = s
V["(CG score)"] = {k: cg[k] for k in label}
keys = sorted(label)
branches = sorted({k[0] for k in keys})


def auc(sc, ks):
    pos = [sc[k] for k in ks if label[k]]
    neg = [sc[k] for k in ks if not label[k]]
    if not pos or not neg:
        return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def full_recall(sc, ks):
    th = min(sc[k] for k in ks if label[k])
    tp = sum(sc[k] >= th and label[k] for k in ks)
    fp = sum(sc[k] >= th and not label[k] for k in ks)
    return th, tp, fp


def lopo(sc):
    tp = fp = fn = 0
    for b in branches:
        tr = [k for k in keys if k[0] != b]
        te = [k for k in keys if k[0] == b]
        th, _, _ = full_recall(sc, tr)
        tp += sum(sc[k] >= th and label[k] for k in te)
        fp += sum(sc[k] >= th and not label[k] for k in te)
        fn += sum(sc[k] < th and label[k] for k in te)
    return tp, fp, fn


npos = sum(label.values())
print(f"{len(V) - 1} CLM variants, {len(keys)} pairs ({npos} duplicates)\n")
print(f"{'variant':30s} {'AUC':>5s}  {'full-recall th':>14s}  {'precision':>9s}   LOPO (fixed variant)")
board = sorted(V, key=lambda v: -auc(V[v], keys))
for v in board[:15] + (["(CG score)"] if "(CG score)" not in board[:15] else []):
    th, tp, fp = full_recall(V[v], keys)
    a, b_, c = lopo(V[v])
    print(f"{v:30s} {auc(V[v], keys):5.3f}  {th:14.3f}  {tp}/{tp + fp} = {tp / (tp + fp):.2f}   "
          f"R {a}/{a + c}  P {a}/{a + b_} = {a / (a + b_):.2f}")

print("\nAUC by layout (rows) x question (cols):")
Ls = sorted({v.split(' | ')[0] for v in V if '|' in v})
Qs = sorted({v.split(' | ')[1] for v in V if '|' in v})
print(" " * 8 + "".join(f"{q[:11]:>12s}" for q in Qs))
for L in Ls:
    print(f"{L:8s}" + "".join(f"{auc(V[f'{L} | {q}'], keys):12.3f}" for q in Qs))

# nested: choose the variant on the training branches, apply to the held-out one
print("\nNested leave-one-PR-out (variant AND threshold chosen without the held-out PR):")
for pool_name, pool in [("CLM variants only", [v for v in V if v != "(CG score)"]), ("CLM variants + CG score", list(V))]:
    tp = fp = fn = 0
    picks = defaultdict(int)
    for b in branches:
        tr = [k for k in keys if k[0] != b]
        te = [k for k in keys if k[0] == b]
        best = max(pool, key=lambda v: auc(V[v], tr))
        picks[best] += 1
        th, _, _ = full_recall(V[best], tr)
        tp += sum(V[best][k] >= th and label[k] for k in te)
        fp += sum(V[best][k] >= th and not label[k] for k in te)
        fn += sum(V[best][k] < th and label[k] for k in te)
    print(f"  {pool_name:26s} recall {tp}/{tp + fn}  precision {tp}/{tp + fp} = {tp / (tp + fp):.2f}   "
          f"picked: {dict(sorted(picks.items(), key=lambda kv: -kv[1]))}")
