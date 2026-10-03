"""The watcher: one pass ("tick") over the repo, run by OpenClaw's scheduler.

Each tick:
  1. poll open + recently closed PRs; record opened / merged / closed events
  2. if main moved (e.g. a merge), re-index its folders. Topics are cached by git tree
     sha, so folders merged from a PR reuse the topic sets computed for that PR.
  3. enqueue every open PR at (head commit, base commit, versions) not yet processed
  4. for each queued PR: topics -> candidates (scores). watcher.clm_worker then judges the
     candidates with CLM, the OpenClaw agent writes the note, and discord_worker posts it.
     CLASSIFIER=1 instead runs the dummy LLM classifier inline (+ [oc] PR comment if
     POST_GITHUB_COMMENTS=1), for a laptop without the CLM stack.

Ticks never overlap: a tick takes an exclusive lock and a tick that finds the lock
held exits immediately, so the next one starts at most POLL_SECONDS after the last
ends. Queue items are claimed before processing; a crashed run's items are retried.

Run:  python -m watcher.main --once   (one tick; what the OpenClaw job runs)
      python -m watcher.main          (loop; local development)
"""
import argparse
import fcntl
import logging
import time
import traceback
from concurrent.futures import ThreadPoolExecutor

from tqdm import tqdm

from . import alerts, classifier, config, db, github, gitrepo, topics
from .history import finish

log = logging.getLogger("watcher")


def topic_brief(t):
    """Name, first sentence of the description and two keywords, for the timeline."""
    first = t["description"].split(". ")[0].rstrip(".")
    return {"name": t["name"], "summary": first[:180] + ("…" if len(first) > 180 else ""), "keywords": t["keywords"][:2]}


def candidate_brief(r):
    """Why a candidate scored: the matched topic pair and each signal, for the timeline."""
    return {"folder": r["repo_id"], "score": round(r["score"], 2), "pr_topic": r["pr_topic"], "repo_topic": r["repo_topic"],
            "desc": round(r["desc_score"], 2), "kw": round(r["kw_score"], 2), "kw_match": r["kw_match"],
            "dataflow": r["dataflow"]}


def stage(conn, pr_number, name, status, message="", **data):
    """Pipeline progress for the live view: one event per stage start/end."""
    db.add_event(conn, "stage", message or f"{name} {status}", pr_number, stage=name, status=status, **data)


def index_base(conn):
    base_sha = gitrepo.fetch_base()
    previous = db.get_kv(conn, db.index_key())
    if previous != base_sha:
        started = time.time()
        folders = gitrepo.folders(base_sha)
        missing = [f for f, t in folders.items() if db.get_folder(conn, t) is None]
        stage(conn, None, "reindex", "start", f"re-indexing main @ {base_sha[:7]}: {len(missing)} folder(s) to extract",
              base_sha=base_sha, to_extract=missing)
        topics.ensure_folders(conn, base_sha, folders)
        before = gitrepo.folders(previous) if previous else {}
        changed = sorted(f for f, t in folders.items() if previous and t != before.get(f))
        db.set_kv(conn, db.index_key(), base_sha)
        msg = (f"indexed main @ {base_sha[:7]}: {len(folders)} folders, {len(missing)} extracted, "
               f"{len(folders) - len(missing)} from cache in {time.time() - started:.0f}s")
        if changed:
            reused = [f for f in changed if f not in missing]
            msg += f"; changed: {', '.join(changed)} (topics reused: {', '.join(reused) or 'none'})"
        log.info(msg)
        db.add_event(conn, "indexed", msg, base_sha=base_sha, changed=changed, extracted=missing)
        stage(conn, None, "reindex", "end", msg, base_sha=base_sha, changed=changed, extracted=missing,
              reused=[f for f in changed if f not in missing], seconds=round(time.time() - started, 1))
    return base_sha


def poll(conn, base_sha):
    open_prs = github.open_prs()
    closed = github.recently_closed_prs()
    changed = db.upsert_prs(conn, closed + open_prs)
    for pr in closed + open_prs:
        if pr["number"] not in changed:
            continue
        before = changed[pr["number"]]
        if pr["state"] == "open":
            db.add_event(conn, "pr_opened" if before is None or before == "closed" else "pr_open",
                         f"PR #{pr['number']} open: {pr['title']}", pr["number"], branch=pr["branch"])
        elif pr["state"] == "merged" and before is not None:
            db.add_event(conn, "pr_merged", f"PR #{pr['number']} merged into {config.BASE_BRANCH}: {pr['title']}",
                         pr["number"], merge_sha=pr["merge_sha"])
        elif pr["state"] == "closed" and before is not None:
            db.add_event(conn, "pr_closed", f"PR #{pr['number']} closed: {pr['title']}", pr["number"])
    for pr in open_prs:
        if db.enqueue(conn, pr["number"], pr["head_sha"], base_sha):
            log.info("queued PR #%s @ %s (%s)", pr["number"], pr["head_sha"][:7], pr["title"])
            stage(conn, pr["number"], "detect", "end", f"PR #{pr['number']} @ {pr['head_sha'][:7]} queued",
                  head_sha=pr["head_sha"], title=pr["title"], url=pr["url"])


def score_pr(conn, base_sha, pr_number, head_sha):
    gitrepo.fetch_pr(pr_number)
    head_folders = gitrepo.folders(head_sha)
    touched = [f for f in gitrepo.touched_folders(base_sha, head_sha) if f in head_folders]  # skip deleted folders

    t0 = time.time()
    if config.REEXTRACT_PR_TOPICS:
        # The PR's own folders always get a fresh extraction; main's folders stay cached.
        # The result is stored under the folder's tree sha, so a merge still reuses it.
        cached = []
        stage(conn, pr_number, "topics", "start", f"extracting topics for {', '.join(touched) or 'no folders'}",
              folders=touched, cached=cached)
        entries = {}
        for f in touched:
            db.put_folder(conn, head_folders[f], topics.build_folder(f, gitrepo.folder_files(head_sha, f)))
            entries[f] = db.get_folder(conn, head_folders[f])
    else:
        cached = [f for f in touched if db.get_folder(conn, head_folders[f]) is not None]
        stage(conn, pr_number, "topics", "start", f"extracting topics for {', '.join(touched) or 'no folders'}",
              folders=touched, cached=cached)
        entries = {f: topics.folder_for(conn, head_sha, f, head_folders[f]) for f in touched}
    stage(conn, pr_number, "topics", "end", "; ".join(f"{f}: {len(e['topics'])} topic(s)" for f, e in entries.items()),
          seconds=round(time.time() - t0, 1), cached=cached,
          topics={f: [t["name"] for t in e["topics"]] for f, e in entries.items()},
          topic_details={f: [topic_brief(t) for t in e["topics"]] for f, e in entries.items()})

    t0 = time.time()
    stage(conn, pr_number, "candidates", "start", "comparing topics with every folder on main")
    rows = []
    for pr_folder, entry in entries.items():
        for r in topics.rank_against_base(conn, base_sha, pr_folder, entry):
            rows.append({"ts": time.time(), "pr_number": pr_number, "head_sha": head_sha,
                         "pr_folder": pr_folder, "base_sha": base_sha, **r})
    db.put_scores(conn, rows)
    cands = [r for r in rows if r["candidate"]]
    stage(conn, pr_number, "candidates", "end", f"{len(cands)} candidate(s) of {len(rows)} folder pairs",
          seconds=round(time.time() - t0, 1), n=len(cands),
          candidates=[{"folder": r["repo_id"], "score": round(r["score"], 2), "dataflow": r["dataflow"]} for r in cands],
          top=[candidate_brief(r) for r in sorted(cands, key=lambda r: -r["score"])[:2]])
    return rows, entries


def classify_candidates(conn, base_sha, pr_number, head_sha, rows, entries):
    """Runs the (dummy) classifier on every candidate; returns findings for the alert."""
    cands = [r for r in rows if r["candidate"]]
    if not cands:
        stage(conn, pr_number, "classifier", "end", "skipped: no candidates to judge", seconds=0, skipped=True, verdicts=[])
        return []
    base_folders = gitrepo.folders(base_sha)

    # DB reads stay on this thread (sqlite); only the LLM calls run in parallel.
    others = {r["repo_id"]: db.get_folder(conn, base_folders[r["repo_id"]]) for r in cands}
    jobs = []
    for r in cands:
        pr_entry, other = entries[r["pr_folder"]], others[r["repo_id"]]
        a = classifier.as_input(pr_entry, classifier.topic_by_name(pr_entry, r["pr_topic"]))
        b = classifier.as_input(other, classifier.topic_by_name(other, r["repo_topic"]))
        jobs.append((r, a, b))
    t0 = time.time()
    stage(conn, pr_number, "classifier", "start", f"judging {len(jobs)} candidate(s)")
    if config.CLASSIFIER:
        with ThreadPoolExecutor(config.LLM_CONCURRENCY) as pool:
            verdicts = list(pool.map(lambda j: classifier.classify(j[1], j[2], j[0]), jobs))
    else:
        verdicts = [{"relation": "upstream" if r["dataflow"] else "unrelated", "is_duplicate": False,
                     "confidence": 0.0, "reason": "classifier disabled"} for r, _, _ in jobs]
    now = time.time()
    db.put_decisions(conn, [
        {"pr_number": pr_number, "head_sha": head_sha, "base_sha": base_sha, "pr_folder": r["pr_folder"],
         "repo_id": r["repo_id"], **{**v, "is_duplicate": int(v["is_duplicate"])}, "classifier": classifier.NAME, "ts": now}
        for (r, _, _), v in zip(jobs, verdicts)
    ])
    stage(conn, pr_number, "classifier", "end", ", ".join(f"{r['repo_id']}: {v['relation']}" for (r, _, _), v in zip(jobs, verdicts)),
          seconds=round(time.time() - t0, 1), classifier=classifier.NAME,
          verdicts=[{"folder": r["repo_id"], "relation": v["relation"], "confidence": v["confidence"]} for (r, _, _), v in zip(jobs, verdicts)])
    owners = {repo: gitrepo.owner(base_sha, repo) for repo in {r["repo_id"] for r in cands}}
    findings = []
    for (r, _, _), v in zip(jobs, verdicts):
        # Report anything the classifier didn't reject, plus every shared-table link.
        if v["relation"] != "unrelated" or r["dataflow"]:
            relation = v["relation"] if v["relation"] != "unrelated" else r["dataflow"].split(":")[0]
            findings.append({**r, **v, "relation": relation, "owner": owners[r["repo_id"]]})
    order = {"duplicate": 0, "partial": 1, "upstream": 2, "downstream": 3}
    findings.sort(key=lambda f: (order.get(f["relation"], 9), -f["score"]))
    return findings


def alert(conn, base_sha, pr_number, head_sha, entries, findings):
    pr = dict(conn.execute("SELECT * FROM prs WHERE number = ?", (pr_number,)).fetchone())
    pr["head_sha"] = head_sha
    pr["commit_author"] = gitrepo.commit_author(head_sha)
    t0 = time.time()
    stage(conn, pr_number, "agent", "start", f"OpenClaw agent '{config.OPENCLAW_AGENT}' writing the note")
    summary, author_by = alerts.agent_summary(pr, sorted(entries), findings)
    stage(conn, pr_number, "agent", "end", summary[:160], seconds=round(time.time() - t0, 1), author_by=author_by)
    body = alerts.comment_body(pr, base_sha, summary, findings, author_by)
    url = None
    if config.POST_GITHUB_COMMENTS:
        t0 = time.time()
        stage(conn, pr_number, "comment", "start", "posting the [oc] comment")
        url, changed = github.upsert_alert_comment(pr_number, body)
        stage(conn, pr_number, "comment", "end", "comment " + ("updated" if changed else "unchanged"),
              seconds=round(time.time() - t0, 1), url=url, changed=changed)
    db.put_alert(conn, {"pr_number": pr_number, "head_sha": head_sha, "base_sha": base_sha, "summary": summary,
                        "body": body, "author_by": author_by, "comment_url": url, "ts": time.time()})
    return url


def process(conn, base_sha, item):
    started = time.time()
    rows, entries = score_pr(conn, base_sha, item["pr_number"], item["head_sha"])
    n_cands = sum(r["candidate"] for r in rows)
    if not config.CLASSIFIER and n_cands:
        # Hand-off: watcher.clm_worker judges the candidates, then writes the note and history.
        stage(conn, item["pr_number"], "classifier", "start", f"waiting for CLM: {n_cands} candidate(s)",
              head_sha=item["head_sha"], base_sha=base_sha)
        return
    findings = classify_candidates(conn, base_sha, item["pr_number"], item["head_sha"], rows, entries)
    url = alert(conn, base_sha, item["pr_number"], item["head_sha"], entries, findings) if config.CLASSIFIER else None
    finish(conn, base_sha, item, entries, rows, findings, url, "live", started)


def tick(conn):
    if requeued := db.requeue_stale(conn):
        db.add_event(conn, "requeued", f"{requeued} stale queue item(s) retried")
    base_sha = index_base(conn)
    poll(conn, base_sha)
    pending = db.pending(conn)
    for item in tqdm(pending, desc="scoring PRs", unit="pr", disable=len(pending) < 2 or not config.PROGRESS):
        if not db.claim(conn, item):
            continue
        try:
            process(conn, base_sha, item)
            db.mark(conn, item, "done")
        except Exception as e:
            log.exception("PR #%s failed", item["pr_number"])
            db.mark(conn, item, "error", traceback.format_exc())
            db.add_event(conn, "error", f"PR #{item['pr_number']} failed: {e}", item["pr_number"])


def setup_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("sentence_transformers", "numexpr", "httpx", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def run_once(conn):
    """One tick under the lock. Returns False if another tick was already running."""
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(config.DATA_DIR / "tick.lock", "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log.info("previous tick still running; skipping")
            return False
        started = time.time()
        db.set_kv(conn, "tick_started", str(started))
        try:
            tick(conn)
            db.set_kv(conn, "last_tick", str(time.time()))
            db.set_kv(conn, "last_tick_seconds", f"{time.time() - started:.1f}")
        except Exception as e:
            log.exception("tick failed")
            db.set_kv(conn, "last_error", traceback.format_exc())
            db.add_event(conn, "error", f"tick failed: {e}")
        finally:
            db.set_kv(conn, "tick_started", "")
        return True


def run(once=False):
    conn = db.connect()
    gitrepo.ensure_clone()
    while True:
        run_once(conn)
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
