"""SQLite inbox, delivery tracking, and exact-message reaction feedback."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import time
from typing import Iterator


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class Store:
    """Use short connections so the flagger and Discord worker can share a DB.

    ``add_decision`` expects a payload already checked by ``validate_decision``.
    Removing a reaction leaves an audit row with a null vote. Two opposing
    reactions from one person are also a null vote until one is removed.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connection() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            # Upgrade existing notification tables before schema.sql installs
            # triggers that reference the new columns. Fresh databases get
            # these columns from CREATE TABLE instead.
            connection.execute("BEGIN IMMEDIATE")
            notification_columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(discord_notifications)")
            }
            if notification_columns:
                for column in ("like_count", "dislike_count"):
                    if column not in notification_columns:
                        connection.execute(
                            f"ALTER TABLE discord_notifications ADD COLUMN {column} "
                            f"INTEGER NOT NULL DEFAULT 0 CHECK ({column} >= 0)"
                        )
            connection.commit()
            connection.executescript(Path(__file__).with_name("schema.sql").read_text())
            connection.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in connection.execute("PRAGMA table_info(discord_decisions)")}
            if "revision_key" not in columns:
                connection.execute("ALTER TABLE discord_decisions ADD COLUMN revision_key TEXT NOT NULL DEFAULT ''")
                connection.execute("UPDATE discord_decisions SET revision_key = head_sha")
            # Backfill preexisting reviews and repair totals on reinitializing.
            # A writer lock keeps this consistent with incoming feedback.
            connection.execute(
                """UPDATE discord_notifications
                   SET like_count = (
                       SELECT COALESCE(SUM(f.thumbs_up), 0) FROM discord_feedback AS f
                       WHERE f.decision_id = discord_notifications.decision_id
                         AND f.channel_id = discord_notifications.channel_id
                   ), dislike_count = (
                       SELECT COALESCE(SUM(f.thumbs_down), 0) FROM discord_feedback AS f
                       WHERE f.decision_id = discord_notifications.decision_id
                         AND f.channel_id = discord_notifications.channel_id
                   )"""
            )

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def add_decision(self, payload: dict) -> bool:
        """Return True for a new decision, False for replay; reject changed data."""
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        head_sha = payload.get("head_sha") or ""
        revision_key = head_sha if head_sha else "source:" + payload["source_revision"]
        with self._connection() as connection:
            # Acquire the writer before checking so concurrent replays are safe.
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT payload_json FROM discord_decisions WHERE decision_id = ?",
                (payload["decision_id"],),
            ).fetchone()
            if existing:
                if existing["payload_json"] != encoded:
                    raise ValueError(f"Decision {payload['decision_id']!r} already exists with different data")
                return False
            connection.execute(
                """INSERT INTO discord_decisions
                   (decision_id, repository, pr_number, head_sha, revision_key, status,
                    is_duplicate, pr_state, model_version, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    payload["decision_id"], payload["repository"], payload["pr_number"],
                    head_sha, revision_key, payload.get("status", "completed"),
                    int(payload["is_duplicate"]), payload.get("pr_state", "open"),
                    payload.get("model_version"), encoded,
                ),
            )
        return True

    def ready(self, channel_id: str, limit: int = 10) -> list[dict]:
        """Return eligible latest decisions whose delivery retry is due."""
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT d.payload_json, d.pr_state FROM discord_decisions AS d
                   LEFT JOIN discord_notifications AS n
                     ON n.decision_id = d.decision_id AND n.channel_id = ?
                   WHERE d.status = 'completed' AND d.is_duplicate = 1 AND d.pr_state = 'open'
                     AND NOT EXISTS (
                         SELECT 1 FROM discord_decisions AS newer
                         WHERE newer.repository = d.repository AND newer.pr_number = d.pr_number
                           AND newer.rowid > d.rowid
                     )
                     AND (n.status IS NULL OR n.status != 'sent')
                     AND (n.next_retry_at IS NULL OR n.next_retry_at <= ?)
                     AND NOT EXISTS (
                         SELECT 1 FROM discord_notifications AS sent
                         JOIN discord_decisions AS previous ON previous.decision_id = sent.decision_id
                         WHERE sent.channel_id = ? AND sent.status = 'sent'
                           AND previous.repository = d.repository
                           AND previous.pr_number = d.pr_number AND previous.revision_key = d.revision_key
                     )
                   ORDER BY d.rowid LIMIT ?""",
                (str(channel_id), time.time(), str(channel_id), limit),
            ).fetchall()
        decisions = []
        for row in rows:
            payload = json.loads(row["payload_json"])
            payload["pr_state"] = row["pr_state"]
            decisions.append(payload)
        return decisions

    def mark_attempt(self, decision_id: str, channel_id: str) -> None:
        now = _now_iso()
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO discord_notifications
                   (decision_id, channel_id, attempts, last_attempt_at, updated_at)
                   VALUES (?, ?, 1, ?, ?)
                   ON CONFLICT (decision_id, channel_id) DO UPDATE SET
                     attempts = discord_notifications.attempts + 1,
                     last_attempt_at = excluded.last_attempt_at,
                     updated_at = excluded.updated_at
                   WHERE discord_notifications.status != 'sent'""",
                (decision_id, str(channel_id), now, now),
            )

    def mark_sent(self, decision_id: str, channel_id: str, message_id: str) -> None:
        now = _now_iso()
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT message_id FROM discord_notifications WHERE decision_id = ? AND channel_id = ?",
                (decision_id, str(channel_id)),
            ).fetchone()
            if existing and existing["message_id"] is not None:
                if existing["message_id"] != str(message_id):
                    raise ValueError("This decision already has a different Discord message")
                return
            connection.execute(
                """INSERT INTO discord_notifications
                   (decision_id, channel_id, status, message_id, sent_at, updated_at)
                   VALUES (?, ?, 'sent', ?, ?, ?)
                   ON CONFLICT (decision_id, channel_id) DO UPDATE SET
                     status = 'sent', message_id = excluded.message_id,
                     sent_at = excluded.sent_at, updated_at = excluded.updated_at,
                     next_retry_at = NULL, last_error = NULL""",
                (decision_id, str(channel_id), str(message_id), now, now),
            )

    def mark_failed(self, decision_id: str, channel_id: str, error: str, retry_after: float = 10) -> None:
        with self._connection() as connection:
            connection.execute(
                """INSERT INTO discord_notifications
                   (decision_id, channel_id, status, last_error, next_retry_at, updated_at)
                   VALUES (?, ?, 'failed', ?, ?, ?)
                   ON CONFLICT (decision_id, channel_id) DO UPDATE SET
                     status = 'failed', last_error = excluded.last_error,
                     next_retry_at = excluded.next_retry_at, updated_at = excluded.updated_at
                   WHERE discord_notifications.status != 'sent'""",
                (decision_id, str(channel_id), str(error), time.time() + max(0, retry_after), _now_iso()),
            )

    def get_notification(self, message_id: str, channel_id: str) -> dict | None:
        with self._connection() as connection:
            row = connection.execute(
                """SELECT * FROM discord_notifications
                   WHERE message_id = ? AND channel_id = ? AND status = 'sent'""",
                (str(message_id), str(channel_id)),
            ).fetchone()
        return dict(row) if row else None

    def record_reaction(self, message_id: str, channel_id: str, user_id: str, emoji: str, added: bool) -> bool:
        """Return whether this is a known notification and supported reaction."""
        if emoji not in ("👍", "👎"):
            return False
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            notification = connection.execute(
                """SELECT decision_id FROM discord_notifications
                   WHERE message_id = ? AND channel_id = ? AND status = 'sent'""",
                (str(message_id), str(channel_id)),
            ).fetchone()
            if notification is None:
                return False
            key = (notification["decision_id"], str(channel_id), str(user_id))
            previous = connection.execute(
                "SELECT * FROM discord_feedback WHERE decision_id = ? AND channel_id = ? AND user_id = ?", key,
            ).fetchone()
            up = bool(previous["thumbs_up"]) if previous else False
            down = bool(previous["thumbs_down"]) if previous else False
            if emoji == "👍":
                if up == bool(added):
                    return True
                up = bool(added)
            else:
                if down == bool(added):
                    return True
                down = bool(added)
            vote = (1 if up else -1) if up != down else None
            now = _now_iso()
            connection.execute(
                """INSERT INTO discord_feedback
                   (decision_id, channel_id, user_id, thumbs_up, thumbs_down, vote, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT (decision_id, channel_id, user_id) DO UPDATE SET
                     thumbs_up = excluded.thumbs_up, thumbs_down = excluded.thumbs_down,
                     vote = excluded.vote, updated_at = excluded.updated_at""",
                (*key, int(up), int(down), vote, now, now),
            )
        return True

    def clear_reactions(self, message_id: str, channel_id: str, emoji: str | None = None) -> None:
        if emoji not in (None, "👍", "👎"):
            return
        if emoji is None:
            assignment = "thumbs_up = 0, thumbs_down = 0, vote = NULL"
            condition = "(thumbs_up = 1 OR thumbs_down = 1)"
        elif emoji == "👍":
            assignment = "thumbs_up = 0, vote = CASE WHEN thumbs_down = 1 THEN -1 ELSE NULL END"
            condition = "thumbs_up = 1"
        else:
            assignment = "thumbs_down = 0, vote = CASE WHEN thumbs_up = 1 THEN 1 ELSE NULL END"
            condition = "thumbs_down = 1"
        with self._connection() as connection:
            # SQL fragments above are fixed constants, never interpolated user input.
            connection.execute(
                f"""UPDATE discord_feedback SET {assignment}, updated_at = ?
                    WHERE channel_id = ? AND {condition} AND decision_id IN (
                        SELECT decision_id FROM discord_notifications
                        WHERE message_id = ? AND channel_id = ? AND status = 'sent'
                    )""",
                (_now_iso(), str(channel_id), str(message_id), str(channel_id)),
            )

    def replace_reactions(self, message_id: str, channel_id: str, votes: list[tuple[str, str]]) -> None:
        """Atomically reconcile a complete Discord reaction snapshot.

        Only changed flags update a row's timestamp. Reviewers absent from the
        snapshot retain their audit row with both flags cleared and a null vote.
        Callers must fetch the full snapshot successfully before calling this.
        """
        snapshot: dict[str, set[str]] = {}
        for user_id, emoji in votes:
            if emoji in ("👍", "👎"):
                snapshot.setdefault(str(user_id), set()).add(emoji)
        with self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            notification = connection.execute(
                """SELECT decision_id FROM discord_notifications
                   WHERE message_id = ? AND channel_id = ? AND status = 'sent'""",
                (str(message_id), str(channel_id)),
            ).fetchone()
            if notification is None:
                return
            decision_id = notification["decision_id"]
            previous = {
                row["user_id"]: row for row in connection.execute(
                    "SELECT * FROM discord_feedback WHERE decision_id = ? AND channel_id = ?",
                    (decision_id, str(channel_id)),
                ).fetchall()
            }
            now = _now_iso()
            for user_id in sorted(snapshot.keys() | previous.keys()):
                flags = snapshot.get(user_id, set())
                up, down = int("👍" in flags), int("👎" in flags)
                old = previous.get(user_id)
                if old and (old["thumbs_up"], old["thumbs_down"]) == (up, down):
                    continue
                vote = (1 if up else -1) if up != down else None
                connection.execute(
                    """INSERT INTO discord_feedback
                       (decision_id, channel_id, user_id, thumbs_up, thumbs_down, vote, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                       ON CONFLICT (decision_id, channel_id, user_id) DO UPDATE SET
                         thumbs_up = excluded.thumbs_up, thumbs_down = excluded.thumbs_down,
                         vote = excluded.vote, updated_at = excluded.updated_at""",
                    (decision_id, str(channel_id), user_id, up, down, vote, now, now),
                )

    def feedback_rows(self) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT f.*, d.repository, d.pr_number, NULLIF(d.head_sha, '') AS head_sha, d.model_version,
                          n.message_id, d.payload_json
                   FROM discord_feedback AS f
                   JOIN discord_decisions AS d ON d.decision_id = f.decision_id
                   JOIN discord_notifications AS n
                     ON n.decision_id = f.decision_id AND n.channel_id = f.channel_id
                   ORDER BY f.created_at, f.decision_id, f.channel_id, f.user_id"""
            ).fetchall()
        feedback = []
        for row in rows:
            item = dict(row)
            item["decision"] = json.loads(item.pop("payload_json"))
            feedback.append(item)
        return feedback

    def notifications(self, channel_id: str, limit: int = 100) -> list[dict]:
        with self._connection() as connection:
            rows = connection.execute(
                """SELECT * FROM discord_notifications
                   WHERE channel_id = ? AND status = 'sent'
                   ORDER BY sent_at DESC, rowid DESC LIMIT ?""",
                (str(channel_id), limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def close_pr(self, repository: str, pr_number: int, state: str) -> None:
        """Update delivery eligibility without rewriting the model's evidence."""
        if state not in ("open", "closed", "merged"):
            raise ValueError("PR state must be open, closed, or merged")
        with self._connection() as connection:
            connection.execute(
                "UPDATE discord_decisions SET pr_state = ? WHERE repository = ? AND pr_number = ?",
                (state, repository, pr_number),
            )
