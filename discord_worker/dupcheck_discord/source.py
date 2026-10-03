"""Read the team's flagger tables without changing or reclassifying its verdicts.

The source schema identifies PRs and incumbent topics, but has no Git SHA, PR
URL, PR state, or detailed code comparison. Snapshot hashes are source database
revisions, never synthetic Git commits. Worker tables retain source IDs so a
review can be traced back to the flagger's original decisions.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
import sqlite3

from .models import validate_decision
from .store import Store


_COLUMNS = {
    "Commited": {"topic_id", "commited_topic", "topic_desc", "owner"},
    "dup_cg": {"dup_id", "pr_id", "t_id", "incumbent_topic", "pr_topic"},
    "dupe_decision": {"dup_id", "boolean"},
    "new_prs": {"pr_id", "topic_id", "folder_name", "topic_desc", "owner", "timestamp"},
}


def _rows(connection: sqlite3.Connection, sql: str, parameters=()) -> list[dict]:
    cursor = connection.execute(sql, parameters)
    columns = [column[0] for column in cursor.description]
    return [dict(zip(columns, row)) for row in cursor.fetchall()]


def _check_schema(connection: sqlite3.Connection) -> None:
    """Fail clearly when configured against a different or unfinished schema."""
    for table, required in _COLUMNS.items():
        # Identifiers come from the fixed schema above, not external values.
        columns = {row["name"].lower() for row in _rows(connection, f'PRAGMA table_info("{table}")')}
        if not columns:
            raise ValueError(f"Team database is missing the {table!r} table")
        missing = required - columns
        if missing:
            raise ValueError(f"Team table {table!r} is missing columns: {', '.join(sorted(missing))}")


def _text(value) -> str:
    return str(value).strip() if value is not None else ""


def _clip(value: str, limit: int) -> str:
    return value if len(value) <= limit else value[:limit - 1].rstrip() + "…"


def _identifier(value, description: str, *, positive=False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError(f"{description} must be {'positive' if positive else 'nonnegative'} integer")
    return value


def _build_decisions(connection: sqlite3.Connection, repository: str) -> list[dict]:
    _check_schema(connection)
    pr_topics = defaultdict(list)
    for row in _rows(connection, """SELECT pr_id, topic_id, folder_name, topic_desc, owner, timestamp
                                    FROM new_prs ORDER BY pr_id, topic_id"""):
        pr_topics[row["pr_id"]].append(row)

    comparisons = defaultdict(list)
    for row in _rows(connection, """SELECT cg.dup_id, cg.pr_id, cg.t_id, cg.incumbent_topic,
                                          cg.pr_topic, d.boolean AS verdict,
                                          c.topic_id AS incumbent_id,
                                          c.commited_topic AS committed_topic,
                                          c.topic_desc AS committed_description,
                                          c.owner AS incumbent_owner
                                   FROM dup_cg AS cg
                                   LEFT JOIN dupe_decision AS d ON d.dup_id = cg.dup_id
                                   LEFT JOIN Commited AS c ON c.topic_id = cg.t_id
                                   ORDER BY cg.pr_id, cg.dup_id"""):
        comparisons[row["pr_id"]].append(row)

    decisions = []
    for pr_number, candidates in comparisons.items():
        topics = pr_topics.get(pr_number)
        # Source tables may be filled in separate transactions. Wait until the
        # PR's author metadata exists rather than assigning an arbitrary owner.
        if not topics:
            continue
        _identifier(pr_number, "new_prs.pr_id", positive=True)
        owners = {_text(row["owner"]) for row in topics}
        if len(owners) != 1 or "" in owners:
            raise ValueError(f"PR #{pr_number} must have exactly one nonempty author in new_prs.owner")
        for row in topics:
            _identifier(row["topic_id"], "new_prs.topic_id")
        for row in candidates:
            _identifier(row["dup_id"], "dup_cg.dup_id")
            _identifier(row["t_id"], "dup_cg.t_id")
            if row["verdict"] is not None and (type(row["verdict"]) is not int or row["verdict"] not in (0, 1)):
                raise ValueError(f"dupe_decision.boolean for dup_id={row['dup_id']} must be integer 0 or 1")
        positive = [row for row in candidates if row["verdict"] == 1]
        pending = any(row["verdict"] is None for row in candidates)
        folders = sorted({_text(row["folder_name"]) for row in topics if _text(row["folder_name"])})
        flagged_topics = sorted({_text(row["pr_topic"]) for row in (positive or candidates) if _text(row["pr_topic"])})
        if not flagged_topics:
            flagged_topics = sorted({_text(row["topic_desc"]) for row in topics if _text(row["topic_desc"])})

        # Include every source candidate, including clean/pending rows, and the
        # actual metadata in the revision. Repeated polling is a stable no-op.
        snapshot = {"repository": repository, "pr_topics": topics, "comparisons": candidates}
        revision = hashlib.sha256(json.dumps(
            snapshot, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
        ).encode("utf-8")).hexdigest()
        reason = ""
        matches = []
        if positive:
            reason = (
                "The detector flagged possible overlap between this PR's topics and the existing topics shown below. "
                "Compare the descriptions and review whether the existing implementation can be reused or extended."
            )
            if len(positive) > 25:
                reason += f" Showing 25 of {len(positive)} flagged comparisons; all source decision IDs are retained."
            for row in positive[:25]:
                incumbent = _text(row["incumbent_topic"]) or _text(row["committed_topic"])
                if not incumbent:
                    raise ValueError(f"Duplicate decision {row['dup_id']} has no existing topic name")
                evidence = [f"Flagged PR topic: {_text(row['pr_topic']) or 'not supplied'}",
                            f"Stored existing topic: {incumbent}"]
                description = _text(row["committed_description"])
                if description:
                    evidence.append(f"Stored existing description: {description}")
                matches.append({
                    "project_name": _clip(incumbent, 256),
                    "topic": _clip(_text(row["committed_topic"]) or incumbent, 6000),
                    "evidence": _clip("\n".join(evidence), 6000),
                    "source_dup_id": row["dup_id"],
                    "source_topic_id": row["t_id"],
                })
        payload = {
            "decision_id": f"team-pr-{pr_number}-{revision[:32]}",
            "repository": repository,
            "pr_number": pr_number,
            "pr_title": _clip(f"PR #{pr_number}" + (": " + ", ".join(folders) if folders else ""), 500),
            "pr_url": None,
            "head_sha": None,
            "author_login": next(iter(owners)),
            "topic": _clip("; ".join(flagged_topics) or f"PR #{pr_number} topics", 256),
            "status": "pending" if pending else "completed",
            "is_duplicate": bool(positive),
            "duplicate_kind": "unspecified",
            "reason": reason,
            "matches": matches,
            "source_revision": revision,
            "source_dup_ids": [row["dup_id"] for row in (positive or candidates) if row["verdict"] is not None],
            "source_pr_topic_ids": [row["topic_id"] for row in topics],
            "folder_names": folders,
        }
        descriptions = sorted({_text(row["topic_desc"]) for row in topics if _text(row["topic_desc"])})
        if descriptions:
            payload["pr_description"] = _clip("\n".join(descriptions), 6000)
        decisions.append(validate_decision(payload))
    return decisions


def build_decisions(connection: sqlite3.Connection, repository: str = "team repository") -> list[dict]:
    """Build one snapshot per source PR in a consistent read transaction.

    ``dup_cg.t_id`` identifies an incumbent Commited topic; it must never be
    joined to ``new_prs.topic_id``. Candidate PR metadata is grouped by pr_id.
    Missing decision rows make the aggregate pending; explicit all-zero
    verdicts produce a completed clean snapshot that supersedes older flags.
    """
    if not isinstance(repository, str) or not repository.strip():
        raise ValueError("Source repository must be a nonempty string")
    owns_transaction = not connection.in_transaction
    if owns_transaction:
        connection.execute("BEGIN")
    try:
        return _build_decisions(connection, repository.strip())
    finally:
        if owns_transaction:
            connection.rollback()


def sync_source(store: Store, repository: str = "team repository") -> int:
    """Copy changed team snapshots into the worker inbox; never edit source rows.

    Reverting a source verdict also creates a new immutable snapshot. Merely
    reusing a content hash as decision_id would leave the earlier verdict newer
    than the replay and could allow a stale duplicate alert to remain eligible.
    """
    connection = sqlite3.connect(store.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("BEGIN")
        decisions = build_decisions(connection, repository)
        latest = {}
        counts = defaultdict(int)
        for row in _rows(connection, """SELECT repository, pr_number, payload_json, pr_state
                                        FROM discord_decisions ORDER BY rowid"""):
            key = (row["repository"], row["pr_number"])
            counts[key] += 1
            latest[key] = {"payload": json.loads(row["payload_json"]), "pr_state": row["pr_state"]}
    finally:
        connection.close()
    ingested = 0
    active = {(payload["repository"], payload["pr_number"]) for payload in decisions}
    for key, previous in latest.items():
        if (key[0] == repository.strip() and key not in active
                and previous["payload"].get("source_revision")
                and previous["pr_state"] == "open"):
            # Absence from the team's active PR/candidate tables disables
            # delivery. This is not a claim about the PR's GitHub close state.
            store.close_pr(key[0], key[1], "closed")
    for payload in decisions:
        key = (payload["repository"], payload["pr_number"])
        previous = latest.get(key, {})
        if (previous.get("payload", {}).get("source_revision") == payload["source_revision"]
                and previous.get("pr_state") == "open"):
            continue
        payload["decision_id"] += f"-v{counts[key] + 1}"
        if store.add_decision(payload):
            ingested += 1
    return ingested
