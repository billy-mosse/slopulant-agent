"""Always-on PR watcher.

Every POLL_SECONDS:
  1. if the base branch moved, (re)index its folders (cards are cached by tree sha)
  2. enqueue any open PR whose head commit we haven't seen
  3. for each queued PR, describe+embed the folders it touches (as they are on the
     PR branch) and score them by cosine similarity against every other folder on base

Run:  python -m watcher.main          (loop)
      python -m watcher.main --once   (single pass, for testing)
"""
import argparse
import logging
import time
import traceback

from tqdm import tqdm

from . import cards, config, db, github, gitrepo

log = logging.getLogger("watcher")


def index_base(conn):
    base_sha = gitrepo.fetch_base()
    if db.get_kv(conn, "indexed_base_sha") != base_sha:
        started = time.time()
        folders = gitrepo.folders(base_sha)
        cards.ensure_cards(conn, base_sha, folders)
        log.info("indexed %d folders on %s @ %s in %.1fs", len(folders), config.BASE_BRANCH, base_sha[:7], time.time() - started)
        db.set_kv(conn, "indexed_base_sha", base_sha)
    return base_sha


def poll(conn, base_sha):
    prs = github.open_prs()
    db.sync_prs(conn, prs)
    for pr in prs:
        if db.enqueue(conn, pr["number"], pr["head_sha"], base_sha):
            log.info("queued PR #%s @ %s (%s)", pr["number"], pr["head_sha"][:7], pr["title"])


def score_pr(conn, base_sha, pr_number, head_sha):
    gitrepo.fetch_pr(pr_number)
    head_folders = gitrepo.folders(head_sha)
    rows = []
    for pr_folder in gitrepo.touched_folders(base_sha, head_sha):
        if pr_folder not in head_folders:  # folder deleted by the PR
            continue
        pr_card = cards.card_for(conn, head_sha, pr_folder, head_folders[pr_folder])
        for r in cards.rank_against_base(conn, base_sha, pr_folder, pr_card):
            rows.append({
                "ts": time.time(), "pr_number": pr_number, "head_sha": head_sha,
                "pr_folder": pr_folder, "base_sha": base_sha, **r,
            })
    db.put_scores(conn, rows)
    for r in [r for r in rows if r["candidate"]]:
        log.info(
            "PR #%s %s ~ %s: %.2f (desc %.2f, kw %.2f [%s], code %.2f, dataflow %s)", pr_number, r["pr_folder"],
            r["repo_id"], r["score"], r["card_score"], r["kw_score"], r["kw_match"], r["code_score"], r["dataflow"],
        )


def tick(conn):
    base_sha = index_base(conn)
    poll(conn, base_sha)
    pending = db.pending(conn)
    for item in tqdm(pending, desc="scoring PRs", unit="pr", disable=len(pending) < 2 or not config.PROGRESS):
        try:
            score_pr(conn, base_sha, item["pr_number"], item["head_sha"])
            db.mark(conn, item, "done")
        except Exception:
            log.exception("PR #%s failed", item["pr_number"])
            db.mark(conn, item, "error", traceback.format_exc())


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("sentence_transformers", "numexpr", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run(once=False):
    conn = db.connect()
    gitrepo.ensure_clone()
    while True:
        try:
            tick(conn)
            db.set_kv(conn, "last_tick", str(time.time()))
        except Exception:
            log.exception("tick failed")
            db.set_kv(conn, "last_error", traceback.format_exc())
        if once:
            break
        time.sleep(config.POLL_SECONDS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    setup_logging()
    run(once=args.once)


if __name__ == "__main__":
    main()
