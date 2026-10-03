"""Read the watcher's tables (data/watcher.db) without changing or reclassifying its verdicts.

One shared SQLite database holds the whole pipeline: the watcher writes PRs, topics and
candidates (`prs`, `topics`, `scores`); the CLM worker writes verdicts (`decisions`), the
OpenClaw note (`alerts`) and a `kv` marker `clm_finished:<head>:<base>` once a PR commit is
fully judged. This adapter turns the latest analysed commit of each PR into one decision
snapshot for the Discord worker's own tables (`discord_*`, same file).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3

from .models import validate_decision
from .store import Store

_TABLES = ("prs", "scores", "decisions", "alerts", "kv", "topics", "folders", "history")


def _rows(connection: sqlite3.Connection, sql: str, parameters=()) -> list[dict]:
    cursor = connection.execute(sql, parameters)
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _check_schema(connection: sqlite3.Connection) -> None:
    """Fail clearly when pointed at a database the watcher hasn't created."""
    present = {row["name"] for row in _rows(connection, "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
    missing = [t for t in (*_TABLES, "latest_scores") if t not in present]
    if missing:
        raise ValueError(f"Not a watcher database (missing {', '.join(missing)}); point DUPCHECK_DB_PATH at data/watcher.db")


def _clip(value: str, limit: int) -> str:
    value = str(value or "").strip()
    return value if len(value) <= limit else value[:limit - 1].rstrip() + "…"


def _topic_description(connection: sqlite3.Connection, folder: str, name: str) -> str:
    """Most recent extraction of this folder's topic (topics are keyed by git tree sha)."""
    row = connection.execute(
        """SELECT t.description FROM topics AS t
           JOIN folders AS f USING (tree_sha, llm_model, embed_model)
           WHERE f.folder = ? AND t.name = ? ORDER BY f.created_at DESC LIMIT 1""", (folder, name)).fetchone()
    return row[0] if row else ""


def _build_decisions(connection: sqlite3.Connection, repository: str) -> list[dict]:
    _check_schema(connection)
    finished = {row["key"].split(":", 1)[1]: row["value"]
                for row in _rows(connection, "SELECT key, value FROM kv WHERE key LIKE 'clm_finished:%'")}
    runs = {}
    for row in _rows(connection, "SELECT * FROM latest_scores ORDER BY pr_number, pr_folder, rank"):
        runs.setdefault(row["pr_number"], []).append(row)
    decisions = []
    for pr in _rows(connection, "SELECT * FROM prs ORDER BY number"):
        rows = runs.get(pr["number"])
        if not rows:
            continue  # not analysed yet
        head_sha, base_sha = rows[0]["head_sha"], rows[0]["base_sha"]
        candidates = [r for r in rows if r["candidate"]]
        verdicts = {(d["pr_folder"], d["repo_id"]): d for d in _rows(
            connection, "SELECT * FROM decisions WHERE pr_number = ? AND head_sha = ? AND base_sha = ?",
            (pr["number"], head_sha, base_sha))}
        version = finished.get(f"{head_sha}:{base_sha}")
        current = [verdicts.get((r["pr_folder"], r["repo_id"])) for r in candidates]
        done = not candidates or (version is not None and all(
            v is not None and v["classifier"] in (version, "dataflow") for v in current))
        positive = [(r, v) for r, v in zip(candidates, current) if done and v and v["is_duplicate"]]
        note = connection.execute(
            "SELECT summary, body, author_by FROM alerts WHERE pr_number = ? AND head_sha = ? AND base_sha = ?",
            (pr["number"], head_sha, base_sha)).fetchone()
        if positive and note is None:
            done = False  # the OpenClaw note is written right after the verdicts
        history = connection.execute("SELECT author, findings FROM history WHERE head_sha = ? AND base_sha = ?",
                                     (head_sha, base_sha)).fetchone()
        owners = {f["repo_id"]: f.get("owner") for f in json.loads(history["findings"])} if history else {}

        matches = []
        for r, v in positive[:25]:
            main_desc = _topic_description(connection, r["repo_id"], r["repo_topic"])
            evidence = [f"Owner: {owners.get(r['repo_id']) or 'unknown'}",
                        f"CLM same-problem score {v['confidence']:.2f} (threshold {v['threshold']})",
                        f"Similarity {r['score']:.2f} (description {r['desc_score']:.2f}, keywords {r['kw_score']:.2f})"]
            if r["kw_match"]:
                evidence.append(f"Shared keywords: {r['kw_match']}")
            if r["dataflow"]:
                evidence.append(f"Shared tables: {r['dataflow'].split(':', 1)[1]}")
            evidence.append(f"Matched PR topic: {r['pr_folder']} · {r['pr_topic']}")
            matches.append({
                "project_name": _clip(r["repo_id"], 256),
                # Discord shows ~800 characters per match: keep the description short so the evidence fits.
                "topic": _clip(f"{r['repo_topic']}" + (f" — {main_desc}" if main_desc else ""), 300),
                "url": f"https://github.com/{repository}/tree/{base_sha}/{r['repo_id']}",
                "evidence": _clip("\n".join(evidence), 6000),
            })
        folders = sorted({r["pr_folder"] for r in rows})
        flagged = sorted({r["pr_topic"] for r, _ in positive}) or sorted({r["pr_topic"] for r in candidates})
        keys = sorted(f"{r['pr_folder']}/{r['repo_id']}" for r, _ in positive) if positive else \
            sorted(f"{r['pr_folder']}/{r['repo_id']}" for r in candidates)

        # The revision covers everything shown, so a changed verdict or note is a new snapshot.
        snapshot = {"pr": pr, "rows": candidates, "verdicts": [v for v in current if v], "note": dict(note) if note else None,
                    "finished": version}
        revision = hashlib.sha256(json.dumps(snapshot, sort_keys=True, default=str).encode("utf-8")).hexdigest()
        payload = {
            "decision_id": f"pr-{pr['number']}-{head_sha[:12]}-{base_sha[:12]}",
            "repository": repository,
            "pr_number": pr["number"],
            "pr_title": _clip(pr["title"], 500) or f"PR #{pr['number']}",
            "pr_url": pr["url"] or None,
            "head_sha": head_sha,
            "author_login": _clip(pr["author"], 256) or "unknown",
            "topic": _clip("; ".join(flagged) or f"PR #{pr['number']}", 256),
            "status": "completed" if done else "pending",
            "pr_state": pr["state"],
            "is_duplicate": bool(positive),
            "duplicate_kind": "partial",
            "reason": _clip(note["summary"], 6000) if note and positive else "",
            "matches": matches,
            "source_revision": revision,
            "source_decision_keys": keys,
            "base_sha": base_sha,
            "folder_names": folders,
        }
        if history and history["author"] and history["author"] != pr["author"]:
            payload["author_name"] = _clip(history["author"], 256)
        if version:
            payload["model_version"] = _clip(version, 256)
        if note and positive:  # Qwen's two parts of the alert (watcher/alerts.py draft_alert)
            try:
                parts = json.loads(note["body"])
            except (TypeError, json.JSONDecodeError):
                parts = {}
            if isinstance(parts, dict) and parts.get("overlap") and parts.get("next_step"):
                payload["overlap"], payload["next_step"] = _clip(parts["overlap"], 1000), _clip(parts["next_step"], 500)
                payload["note_by"] = _clip(note["author_by"], 64)
        descriptions = []
        for folder, name in sorted({(r["pr_folder"], r["pr_topic"]) for r in rows}):
            if description := _topic_description(connection, folder, name):
                descriptions.append(f"{folder} · {name}: {description}")
        if descriptions:
            payload["pr_description"] = _clip("\n".join(descriptions), 6000)
        decisions.append(validate_decision(payload))
    return decisions


def build_decisions(connection: sqlite3.Connection, repository: str) -> list[dict]:
    """One snapshot per analysed PR (its latest commit), read in one consistent transaction."""
    if not isinstance(repository, str) or not repository.strip():
        raise ValueError("Source repository must be a nonempty string")
    connection.row_factory = sqlite3.Row
    owns_transaction = not connection.in_transaction
    if owns_transaction:
        connection.execute("BEGIN")
    try:
        return _build_decisions(connection, repository.strip())
    finally:
        if owns_transaction:
            connection.rollback()


def sync_source(store: Store, repository: str) -> int:
    """Copy changed snapshots into the worker's tables; never edit the watcher's rows.

    A PR that GitHub reports closed or merged stops being eligible for alerts. A new
    snapshot gets a new versioned decision_id so the newest one always supersedes.
    """
    connection = sqlite3.connect(store.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        connection.execute("PRAGMA busy_timeout = 5000")
        decisions = build_decisions(connection, repository)
        latest, counts = {}, {}
        for row in _rows(connection, "SELECT repository, pr_number, payload_json, pr_state FROM discord_decisions ORDER BY rowid"):
            key = (row["repository"], row["pr_number"])
            counts[key] = counts.get(key, 0) + 1
            latest[key] = {"payload": json.loads(row["payload_json"]), "pr_state": row["pr_state"]}
    finally:
        connection.close()
    ingested = 0
    for payload in decisions:
        key = (payload["repository"], payload["pr_number"])
        previous = latest.get(key)
        if payload["pr_state"] != "open":
            if previous and previous["pr_state"] != payload["pr_state"]:
                store.close_pr(*key, payload["pr_state"])
            continue
        if previous and previous["payload"].get("source_revision") == payload["source_revision"] \
                and previous["pr_state"] == "open":
            continue
        payload["decision_id"] += f"-v{counts.get(key, 0) + 1}"
        if store.add_decision(payload):
            ingested += 1
    return ingested
