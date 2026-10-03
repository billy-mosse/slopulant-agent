"""End-to-end eval of the watcher (a candidate generator) on the [TEST] PRs.

Runs one watcher pass (every open PR scored at its current head), then rebuilds each
PR folder's candidate set for several K and compares it with eval/cases.json:

  candidates(K)   every dataflow-linked folder + the top-K other folders by similarity
                  (optionally only those >= MIN_CANDIDATE_SCORE)
  recall          share of expected folders (duplicate / partial / upstream /
                  downstream) that are candidates; "related" folders don't count
  cands/folder    average candidate-set size: the work handed to the next stage
  silent negs     unrelated PR folders that produce no candidates at all

    python -m eval.run_eval            # K table + per-relation recall + per-PR misses
    python -m eval.run_eval --quiet    # K table + per-relation recall
"""
import json
import sys
from collections import Counter
from pathlib import Path

import requests

from eval.build_cases import POSITIVE
from watcher import config, db, github, gitrepo
from watcher.main import tick

CASES = {k: v for k, v in json.loads((Path(__file__).parent / "cases.json").read_text()).items() if not k.startswith("_")}
KS = (1, 3, 5, 10)


def open_prs_by_branch():
    resp = requests.get(
        f"https://api.github.com/repos/{config.GITHUB_REPO}/pulls",
        params={"state": "open", "per_page": 100},
        headers={"Authorization": f"Bearer {github._token()}"},
        timeout=20,
    )
    resp.raise_for_status()
    return {pr["head"]["ref"]: (pr["number"], pr["head"]["sha"]) for pr in resp.json()}


def candidates(rows, k, floor):
    flow = {r["repo_id"] for r in rows if r["dataflow"]}
    similar = [r for r in rows if not r["dataflow"] and (floor is None or r["score"] >= floor)]
    return flow | {r["repo_id"] for r in similar[:k]}


def main():
    quiet = "--quiet" in sys.argv
    conn = db.connect()
    gitrepo.ensure_clone()
    tick(conn)
    prs = open_prs_by_branch()

    folders, missing = [], []  # (label, rows, truth)
    for branch, pr_folders in CASES.items():
        if branch not in prs:
            missing.append(branch)
            continue
        number, sha = prs[branch]
        for folder, truth in pr_folders.items():
            rows = conn.execute(
                "SELECT * FROM scores WHERE pr_number = ? AND head_sha = ? AND pr_folder = ? ORDER BY score DESC",
                (number, sha, folder),
            ).fetchall()
            if rows:
                folders.append((f"#{number} {folder}", rows, truth))
            else:
                missing.append(f"{branch}:{folder} (not scored)")

    def evaluate(k, floor):
        hit = want = n_cands = neg = neg_silent = 0
        for _, rows, truth in folders:
            cands = candidates(rows, k, floor)
            expected = {repo for repo, rel in truth.items() if set(rel) & set(POSITIVE)}
            hit += len(expected & cands)
            want += len(expected)
            n_cands += len(cands)
            if not expected:
                neg += 1
                neg_silent += not cands
        return hit, want, n_cands / len(folders), neg_silent, neg

    floor = config.MIN_CANDIDATE_SCORE
    print(f"{len(folders)} PR folders | floor {floor} | current K = {config.TOP_K}\n")
    print(f"{'K':>3}  {'recall (no floor)':>18}  {'recall (floor)':>15}  {'cands/folder':>12}  {'silent negs':>11}")
    for k in KS:
        h0, w, _, _, _ = evaluate(k, None)
        h1, _, avg, silent, neg = evaluate(k, floor)
        mark = "  <- current" if k == config.TOP_K else ""
        print(f"{k:>3}  {h0:>8}/{w:<3} = {h0 / w:.2f}  {h1:>5}/{w:<3} = {h1 / w:.2f}  {avg:>12.1f}  {silent:>7}/{neg}{mark}")

    hit, want = Counter(), Counter()
    misses = []
    for label, rows, truth in folders:
        cands = candidates(rows, config.TOP_K, floor)
        for repo, rel in truth.items():
            for r in set(rel) & set(POSITIVE):
                want[r] += 1
                hit[r] += repo in cands
                if repo not in cands:
                    rank = [x["repo_id"] for x in rows if not x["dataflow"]].index(repo) + 1 if repo in {x["repo_id"] for x in rows if not x["dataflow"]} else None
                    score = next((x["score"] for x in rows if x["repo_id"] == repo), None)
                    misses.append(f"{label}: {repo} ({r}) similarity rank {rank}, score {score:.2f}")
    print(f"\nrecall by relation at current settings (K={config.TOP_K}, floor {floor}):")
    for r in POSITIVE:
        if want[r]:
            print(f"  {r:11s} {hit[r]:3d}/{want[r]}")
    if misses and not quiet:
        print("\nmisses:")
        for m in misses:
            print("  " + m)
    if missing:
        print("\nnot evaluated:", ", ".join(missing))


if __name__ == "__main__":
    main()
