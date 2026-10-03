"""Analyses every PR branch of the monorepo into the team view's history, without
posting comments or touching the live pipeline view. Same topics, candidates and
classifier as the watcher; rows are marked source='backfill'.

    python -m watcher.backfill
"""
import time
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from . import classifier, config, db, github, gitrepo, topics
from .history import record_history
from .main import setup_logging


def analyse(conn, base_sha, head_sha):
    head_folders = gitrepo.folders(head_sha)
    touched = [f for f in gitrepo.touched_folders(base_sha, head_sha) if f in head_folders]
    entries = {f: topics.folder_for(conn, head_sha, f, head_folders[f]) for f in touched}
    rows = [{"pr_folder": f, **r} for f, e in entries.items() for r in topics.rank_against_base(conn, base_sha, f, e)]
    cands = [r for r in rows if r["candidate"]]
    base_folders = gitrepo.folders(base_sha)
    others = {r["repo_id"]: db.get_folder(conn, base_folders[r["repo_id"]]) for r in cands}
    jobs = [(r, classifier.as_input(entries[r["pr_folder"]], classifier.topic_by_name(entries[r["pr_folder"]], r["pr_topic"])),
             classifier.as_input(others[r["repo_id"]], classifier.topic_by_name(others[r["repo_id"]], r["repo_topic"])))
            for r in cands]
    with ThreadPoolExecutor(config.LLM_CONCURRENCY) as pool:
        verdicts = list(pool.map(lambda j: classifier.classify(j[1], j[2], j[0]), jobs))
    findings = []
    for (r, _, _), v in zip(jobs, verdicts):
        if v["relation"] != "unrelated" or r["dataflow"]:
            relation = v["relation"] if v["relation"] != "unrelated" else r["dataflow"].split(":")[0]
            findings.append({**r, **v, "relation": relation, "owner": gitrepo.owner(base_sha, r["repo_id"])})
    return entries, rows, findings


def main():
    setup_logging()
    conn = db.connect()
    gitrepo.ensure_clone()
    base_sha = gitrepo.fetch_base()
    topics.ensure_folders(conn, base_sha, gitrepo.folders(base_sha))
    branches = {b: sha for b, sha in gitrepo.fetch_branches().items() if b != config.BASE_BRANCH}
    latest = {}
    for pr in sorted(github.all_prs(), key=lambda p: p["number"]):
        latest[pr["branch"]] = pr
    started = time.time()
    for branch, head in tqdm(sorted(branches.items()), desc="backfill", unit="branch", disable=not config.PROGRESS):
        try:
            entries, rows, findings = analyse(conn, base_sha, head)
        except Exception as e:  # e.g. a branch already merged into main: nothing to compare
            print(f"skip {branch}: {e}")
            continue
        if not entries:
            continue
        pr = latest.get(branch)
        record_history(conn, base_sha, {"pr_number": pr["number"] if pr else None, "head_sha": head}, entries, rows,
                       findings, None, "backfill", branch=branch, title=pr["title"] if pr else branch)
    db.add_event(conn, "backfill", f"backfilled {len(branches)} PR branches into the team history in {time.time() - started:.0f}s")
    print(f"backfilled {len(branches)} branches in {time.time() - started:.0f}s")


if __name__ == "__main__":
    main()
