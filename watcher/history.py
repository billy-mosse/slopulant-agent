"""The team view's history: one row per analysed PR commit. Shared by the watcher, the
backfill and the CLM worker (which runs in the CLM venv, so keep imports light)."""
import json
import logging
import time

from . import db, gitrepo, teams

log = logging.getLogger("watcher")


def record_history(conn, base_sha, item, entries, rows, findings, url, source, branch=None, title=None):
    """One row per analysed PR commit for the team view (kept across demo resets)."""
    item = dict(item)  # queue items are sqlite3.Row (no .get)
    info = teams.load(base_sha)
    pr = conn.execute("SELECT * FROM prs WHERE number = ?", (item["pr_number"],)).fetchone() if item.get("pr_number") else None
    author = gitrepo.commit_author(item["head_sha"])
    # latest detection of this commit: a retry after a failure restarts the clock
    detected = conn.execute("SELECT max(ts) FROM events WHERE kind = 'stage' AND pr_number = ? "
                            "AND json_extract(data, '$.stage') = 'detect' AND json_extract(data, '$.head_sha') = ?",
                            (item.get("pr_number"), item["head_sha"])).fetchone()[0] if item.get("pr_number") else None
    db.put_history(conn, {
        "branch": branch or (pr["branch"] if pr else "?"), "head_sha": item["head_sha"], "base_sha": base_sha,
        "pr_number": item.get("pr_number"), "title": title or (pr["title"] if pr else None), "author": author,
        "author_team": info["person_team"].get(author),
        "folders": json.dumps({f: [t["name"] for t in e["topics"]] for f, e in entries.items()}),
        "findings": json.dumps([{
            "repo_id": f["repo_id"], "repo_topic": f["repo_topic"], "owner": f["owner"],
            "owner_team": info["folder_team"].get(f["repo_id"]), "relation": f["relation"],
            "confidence": f["confidence"], "reason": f["reason"]} for f in findings]),
        "n_candidates": sum(r["candidate"] for r in rows), "comment_url": url, "detected_at": detected,
        "alerted_at": time.time(), "source": source,
    })


def finish(conn, base_sha, item, entries, rows, findings, url, source, started):
    """Last step for a PR commit, once its verdicts are in: history row + "scored" event."""
    dupes = [f["repo_id"] for f in findings if f["relation"] in ("duplicate", "partial")]
    deps = [f["repo_id"] for f in findings if f["relation"] in ("upstream", "downstream")]
    record_history(conn, base_sha, item, entries, rows, findings, url, source)
    msg = (f"PR #{item['pr_number']} scored in {time.time() - started:.0f}s: "
           f"{sum(r['candidate'] for r in rows)} candidates, duplicates: {', '.join(dupes) or 'none'}, "
           f"connected: {', '.join(deps) or 'none'}")
    log.info(msg)
    db.add_event(conn, "scored", msg, item["pr_number"], head_sha=item["head_sha"], duplicates=dupes,
                 connected=deps, comment_url=url, topics={f: [t["name"] for t in e["topics"]] for f, e in entries.items()})
