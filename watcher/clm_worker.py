"""CLM worker: judges the watcher's duplicate candidates with CLM-v0.1-8B.

Polls the shared database for candidate pairs (scores.candidate = 1, latest run of each
PR) that have no verdict yet, and writes one `decisions` row per pair:

  - pair with a shared-table link -> upstream/downstream from the tables, no CLM call
  - every other pair              -> clm_dupe.decision.decide(): is_duplicate, score, threshold

When every candidate of a PR commit has a verdict, the OpenClaw agent writes the note for
the author (stored in `alerts`), the commit's history row and "scored" event are written,
and the commit is marked finished in `kv` (clm_finished:<head>:<base> = CLM version). The
Discord worker alerts on finished commits only.

Runs in the CLM venv (contrastive-lm), not the watcher's: it never imports watcher.topics.
Pairs are scored one at a time on purpose (see clm_dupe/README.md).

    HF_HUB_OFFLINE=1 CLM_DEVICE=cpu CLM_CKPT=.../CLM_v0.1-8B.pt ~/clm-venv/bin/python -m watcher.clm_worker
"""
import argparse
import logging
import time
import traceback

import requests

from clm_dupe import decision as D

from . import alerts, config, db, gitrepo
from .history import finish

log = logging.getLogger("watcher.clm")

# Candidate pairs without a current verdict. Verdicts from another classifier (the old
# inline dummy, or an earlier CLM version) are re-judged.
UNDECIDED = """
SELECT s.* FROM latest_scores s
LEFT JOIN decisions d USING (pr_number, head_sha, base_sha, pr_folder, repo_id)
WHERE s.candidate = 1 AND (d.pr_number IS NULL OR d.classifier NOT IN (:version, 'dataflow'))
ORDER BY s.ts, s.pr_number, s.pr_folder, s.rank
"""

# PR commits whose candidates all have current verdicts but that haven't been finished yet.
UNFINISHED = """
SELECT s.pr_number, s.head_sha, s.base_sha FROM latest_scores s
LEFT JOIN decisions d USING (pr_number, head_sha, base_sha, pr_folder, repo_id)
WHERE s.candidate = 1
GROUP BY s.pr_number, s.head_sha, s.base_sha
HAVING count(CASE WHEN d.classifier IN (:version, 'dataflow') THEN 1 END) = count(*)
   AND NOT EXISTS (SELECT 1 FROM kv WHERE key = 'clm_finished:' || s.head_sha || ':' || s.base_sha AND value = :version)
"""


def finished_key(head_sha, base_sha):
    """kv marker: this PR commit is fully judged and its note written (value = CLM version).
    The Discord worker alerts only on finished commits."""
    return f"clm_finished:{head_sha}:{base_sha}"


def stage(conn, pr_number, name, status, message, **data):
    db.add_event(conn, "stage", message, pr_number, stage=name, status=status, **data)


def topic(entry, name):
    """{"folder", "name", "description"} of the named topic (first topic if the name is gone)."""
    t = next((t for t in entry["topics"] if t["name"] == name), entry["topics"][0])
    return {"folder": entry["folder"], "name": t["name"], "description": t["description"]}


class Worker:
    def __init__(self, conn):
        self.conn = conn
        self.engine = None
        self.started = {}  # (pr, head, base) -> when this worker first saw it

    def encoder_up(self):
        try:
            return requests.get(config.CLM_EMB_URL.split("/v1/")[0] + "/health", timeout=5).ok
        except requests.RequestException:
            return False

    def judge(self, row):
        """Verdict for one candidate pair, as a `decisions` row."""
        base = {k: row[k] for k in ("pr_number", "head_sha", "base_sha", "pr_folder", "repo_id")}
        if row["dataflow"]:
            direction, tables = row["dataflow"].split(":", 1)
            return {**base, "relation": direction, "is_duplicate": 0, "confidence": 1.0, "threshold": None,
                    "reason": f"shared tables: {tables}", "classifier": "dataflow", "ts": time.time()}
        pr_entry = db.get_folder(self.conn, gitrepo.folders(row["head_sha"])[row["pr_folder"]])
        main_entry = db.get_folder(self.conn, gitrepo.folders(row["base_sha"])[row["repo_id"]])
        if pr_entry is None or main_entry is None:
            raise LookupError(f"topics missing for {row['pr_folder']} / {row['repo_id']}")
        if self.engine is None:
            from clm import Engine
            self.engine = Engine(emb_url=config.CLM_EMB_URL)
        v = D.decide(self.engine, topic(main_entry, row["repo_topic"]), topic(pr_entry, row["pr_topic"]))
        return {**base, "relation": "duplicate" if v["is_dupe"] else "unrelated", "is_duplicate": int(v["is_dupe"]),
                "confidence": v["score"], "threshold": v["threshold"], "classifier": v["model_version"],
                "reason": f"CLM: same problem p={v['score']:.3f} (threshold {v['threshold']})", "ts": time.time()}

    def decide_pending(self):
        rows = self.conn.execute(UNDECIDED, {"version": D.VERSION}).fetchall()
        for row in rows:
            key = (row["pr_number"], row["head_sha"], row["base_sha"])
            if key not in self.started:
                self.started[key] = time.time()
                n = sum((r["pr_number"], r["head_sha"], r["base_sha"]) == key for r in rows)
                stage(self.conn, row["pr_number"], "classifier", "start", f"CLM judging {n} candidate(s)",
                      head_sha=row["head_sha"], base_sha=row["base_sha"], classifier=D.VERSION)
            try:
                db.put_decisions(self.conn, [self.judge(row)])
            except LookupError as e:
                log.warning("PR #%s: %s; will retry", row["pr_number"], e)
        return len(rows)

    def finish_ready(self):
        for pr_number, head_sha, base_sha in self.conn.execute(UNFINISHED, {"version": D.VERSION}).fetchall():
            try:
                self.finish_one(pr_number, head_sha, base_sha)
            except Exception as e:
                log.exception("PR #%s: finishing failed", pr_number)
                db.add_event(self.conn, "error", f"PR #{pr_number}: CLM worker failed to finish: {e}", pr_number)

    def finish_one(self, pr_number, head_sha, base_sha):
        key = (pr_number, head_sha, base_sha)
        started = self.started.pop(key, time.time())
        rows = [dict(r) for r in self.conn.execute(
            "SELECT * FROM latest_scores WHERE pr_number = ? AND head_sha = ? AND base_sha = ?", key)]
        verdicts = {r["repo_id"] + "\0" + r["pr_folder"]: dict(r) for r in self.conn.execute(
            "SELECT * FROM decisions WHERE pr_number = ? AND head_sha = ? AND base_sha = ?", key)}
        head_folders = gitrepo.folders(head_sha)
        entries = {f: db.get_folder(self.conn, head_folders[f]) for f in {r["pr_folder"] for r in rows} if f in head_folders}
        entries = {f: e for f, e in entries.items() if e is not None}
        findings = []
        for r in rows:
            v = verdicts.get(r["repo_id"] + "\0" + r["pr_folder"])
            if r["candidate"] and v and (v["is_duplicate"] or r["dataflow"]):
                findings.append({**r, "relation": v["relation"], "confidence": v["confidence"], "reason": v["reason"],
                                 "owner": gitrepo.owner(base_sha, r["repo_id"])})
        order = {"duplicate": 0, "partial": 1, "upstream": 2, "downstream": 3}
        findings.sort(key=lambda f: (order.get(f["relation"], 9), -f["score"]))
        dupes = [f for f in findings if f["relation"] in ("duplicate", "partial")]
        stage(self.conn, pr_number, "classifier", "end",
              ", ".join(f"{f['repo_id']}: {f['relation']}" for f in findings) or "no duplicates",
              seconds=round(time.time() - started, 1), classifier=D.VERSION, head_sha=head_sha,
              verdicts=[{"folder": v["repo_id"], "relation": v["relation"], "confidence": v["confidence"]}
                        for v in verdicts.values()])
        if dupes:
            self.write_note(pr_number, head_sha, base_sha, sorted(entries), findings)
        finish(self.conn, base_sha, {"pr_number": pr_number, "head_sha": head_sha}, entries, rows, findings,
               None, "live", started)
        db.set_kv(self.conn, finished_key(head_sha, base_sha), D.VERSION)

    def write_note(self, pr_number, head_sha, base_sha, folders, findings):
        """The OpenClaw agent's note for the author; the Discord worker uses it as the alert text."""
        pr = self.conn.execute("SELECT * FROM prs WHERE number = ?", (pr_number,)).fetchone()
        pr = {"number": pr_number, "title": f"PR #{pr_number}", "author": "unknown", **(dict(pr) if pr else {}),
              "head_sha": head_sha, "commit_author": gitrepo.commit_author(head_sha)}
        t0 = time.time()
        stage(self.conn, pr_number, "agent", "start", f"OpenClaw agent '{config.OPENCLAW_AGENT}' writing the note",
              head_sha=head_sha)
        summary, author_by = alerts.agent_summary(pr, folders, findings)
        stage(self.conn, pr_number, "agent", "end", summary[:160], seconds=round(time.time() - t0, 1),
              author_by=author_by, head_sha=head_sha)
        db.put_alert(self.conn, {"pr_number": pr_number, "head_sha": head_sha, "base_sha": base_sha, "summary": summary,
                                 "body": summary, "author_by": author_by, "comment_url": None, "ts": time.time()})

    def tick(self):
        if not self.encoder_up():
            if self.conn.execute(UNDECIDED, {"version": D.VERSION}).fetchone():
                log.warning("CLM encoder not reachable at %s; candidates wait", config.CLM_EMB_URL)
                db.add_event(self.conn, "error", f"CLM encoder not reachable at {config.CLM_EMB_URL}")
                return False
        else:
            self.decide_pending()
        self.finish_ready()
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="one pass, then exit")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    conn = db.connect()
    worker = Worker(conn)
    log.info("CLM worker: %s, encoder %s, db %s", D.VERSION, config.CLM_EMB_URL, config.DB_PATH)
    backoff = config.CLM_POLL_SECONDS
    while True:
        try:
            ok = worker.tick()
        except Exception as e:
            log.exception("CLM worker tick failed")
            db.add_event(conn, "error", f"CLM worker tick failed: {e}", data=traceback.format_exc()[-2000:])
            ok = False
        if args.once:
            break
        backoff = config.CLM_POLL_SECONDS if ok else min(backoff * 2, 120)
        time.sleep(backoff)


if __name__ == "__main__":
    main()
