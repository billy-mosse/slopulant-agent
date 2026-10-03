"""Daily re-index: regenerate the topics of every folder on main with Qwen, bypassing the
topic cache. Scheduled by the OpenClaw job `slopulant-reindex` at 06:00 America/New_York.

The watcher's normal path already re-indexes a folder whenever its contents change (topics
are keyed by git tree sha). This job also picks up changes that don't touch the code: a
new topic prompt, a model upgrade, or drift in what the model writes. It takes the
watcher's tick lock, so no tick runs while it rewrites topics.

    python -m watcher.reindex                   # all folders, fresh LLM calls (~minutes)
    python -m watcher.reindex --from-cache      # dry run: same flow and records, cached topics, no LLM
    python -m watcher.reindex --folders a,b     # only some folders
"""
import argparse
import fcntl
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import config, db, gitrepo

log = logging.getLogger("watcher.reindex")


def stage(conn, status, message, **data):
    db.add_event(conn, "stage", message, None, stage="reindex", status=status, daily=True, **data)


def reindex(conn, from_cache=False, only=None):
    base_sha = gitrepo.fetch_base()
    folders = {f: t for f, t in gitrepo.folders(base_sha).items() if not only or f in only}
    started = time.time()
    mode = "from cache (dry run, no LLM calls)" if from_cache else f"fresh LLM extraction ({config.llm()['model']})"
    stage(conn, "start", f"daily re-index of main @ {base_sha[:7]}: {len(folders)} folders, {mode}",
          base_sha=base_sha, to_extract=sorted(folders), from_cache=from_cache)
    done, failed = [], {}
    if from_cache:
        for folder, tree in folders.items():
            entry = db.get_folder(conn, tree)
            if entry is None:
                failed[folder] = "not in cache"
                continue
            db.put_folder(conn, tree, entry)  # rewrite in place: same records and timestamps as a real run
            done.append(folder)
    else:
        from . import topics  # sentence-transformers + LLM client: only needed for a real run
        config.LLM_CACHE = False  # the point is fresh topics
        with ThreadPoolExecutor(config.LLM_CONCURRENCY) as pool:
            futures = {pool.submit(lambda f=f: topics.build_folder(f, gitrepo.folder_files(base_sha, f))): f for f in folders}
            for fut in as_completed(futures):  # DB writes stay on this thread
                folder = futures[fut]
                try:
                    db.put_folder(conn, folders[folder], fut.result())
                    done.append(folder)
                except Exception as e:  # keep the old topics for this folder
                    failed[folder] = str(e)[:200]
                    log.exception("re-index of %s failed", folder)
    seconds = round(time.time() - started, 1)
    msg = (f"daily re-index of main @ {base_sha[:7]}: {len(done)}/{len(folders)} folders regenerated in {seconds:.0f}s"
           + (" (from cache)" if from_cache else "") + (f"; failed: {', '.join(sorted(failed))}" if failed else ""))
    stage(conn, "end", msg, base_sha=base_sha, extracted=sorted(done), reused=[], failed=failed, seconds=seconds,
          from_cache=from_cache)
    db.add_event(conn, "indexed", msg, base_sha=base_sha, daily=True, from_cache=from_cache)
    db.set_kv(conn, "last_daily_reindex", json.dumps({"at": time.time(), "base_sha": base_sha, "folders": len(folders),
                                                       "regenerated": len(done), "failed": failed, "seconds": seconds,
                                                       "from_cache": from_cache, "model": config.llm()["model"]}))
    log.info(msg)
    return msg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--from-cache", action="store_true", help="dry run: re-store cached topics, no LLM calls")
    parser.add_argument("--folders", help="comma-separated subset of folders")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    conn = db.connect()
    gitrepo.ensure_clone()
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.DATA_DIR / "tick.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # wait for a running tick; ticks skip while we hold it
        print(reindex(conn, args.from_cache, set(args.folders.split(",")) if args.folders else None))


if __name__ == "__main__":
    main()
