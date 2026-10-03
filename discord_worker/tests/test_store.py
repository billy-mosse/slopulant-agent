from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from dupcheck_discord.store import Store


def decision(**overrides):
    payload = {
        "decision_id": "decision-1",
        "repository": "team/projects",
        "pr_number": 42,
        "pr_title": "Add receipt parser",
        "pr_url": "https://github.com/team/projects/pull/42",
        "head_sha": "commit-1",
        "author_login": "maya",
        "topic": "Receipt extraction",
        "status": "completed",
        "is_duplicate": True,
        "duplicate_kind": "partial",
        "reason": "Receipt field extraction exists in Invoice Parser.",
        "matches": [{
            "project_name": "Invoice Parser",
            "pr_files": ["receipt.py"],
            "existing_files": ["invoice.py"],
            "evidence": "Both extract merchant, date, and total.",
        }],
        "pr_state": "open",
        "model_version": "qwen-demo",
    }
    payload.update(overrides)
    return payload


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "team.sqlite3"
        self.store = Store(self.path)

    def add_and_send(self, payload=None, message_id="900", channel_id="100"):
        payload = payload or decision()
        self.store.add_decision(payload)
        self.store.mark_attempt(payload["decision_id"], channel_id)
        self.store.mark_sent(payload["decision_id"], channel_id, message_id)

    def assert_counts(self, likes, dislikes, message_id="900", channel_id="100", store=None):
        notification = (store or self.store).get_notification(message_id, channel_id)
        self.assertIsNotNone(notification)
        self.assertEqual((notification["like_count"], notification["dislike_count"]), (likes, dislikes))

    def test_only_completed_duplicate_open_prs_are_ready(self):
        variants = [
            {"is_duplicate": False},
            {"status": "pending"},
            {"status": "failed"},
            {"pr_state": "closed"},
            {"pr_state": "merged"},
            {},
        ]
        for index, variant in enumerate(variants):
            self.store.add_decision(decision(
                decision_id=f"decision-{index}", pr_number=index + 1, **variant,
            ))
        ready = self.store.ready("100")
        self.assertEqual([p["decision_id"] for p in ready], ["decision-5"])

    def test_newest_decision_supersedes_old_even_if_pending_or_clean(self):
        self.store.add_decision(decision())
        self.store.add_decision(decision(
            decision_id="pending-2", head_sha="commit-2", status="pending",
        ))
        self.assertEqual(self.store.ready("100"), [])
        self.store.add_decision(decision(decision_id="completed-2", head_sha="commit-2"))
        self.assertEqual(self.store.ready("100")[0]["decision_id"], "completed-2")
        self.store.add_decision(decision(
            decision_id="clean-3", head_sha="commit-3", is_duplicate=False,
        ))
        self.assertEqual(self.store.ready("100"), [])

    def test_other_repository_with_same_pr_number_is_independent(self):
        self.store.add_decision(decision())
        self.store.add_decision(decision(
            decision_id="other-repo", repository="team/other", is_duplicate=False,
        ))
        self.assertEqual([p["decision_id"] for p in self.store.ready("100")], ["decision-1"])

    def test_sent_notification_is_not_sent_again_and_survives_restart(self):
        self.add_and_send()
        self.assert_counts(0, 0)
        self.store.add_decision(dict(reversed(list(decision().items()))))
        restarted = Store(self.path)
        self.assertEqual(restarted.ready("100"), [])
        self.assertEqual(len(restarted.ready("different-channel")), 1)
        mapping = restarted.get_notification("900", "100")
        self.assertEqual(mapping["decision_id"], "decision-1")
        self.assertEqual(mapping["attempts"], 1)
        self.assertIsNone(restarted.get_notification("900", "wrong-channel"))
        restarted.mark_failed("decision-1", "100", "late failure")
        restarted.mark_attempt("decision-1", "100")
        self.assertEqual(restarted.get_notification("900", "100")["attempts"], 1)
        self.assertEqual(len(restarted.notifications("100")), 1)
        self.assertEqual(restarted.notifications("100")[0]["like_count"], 0)
        self.assertEqual(restarted.notifications("100")[0]["dislike_count"], 0)

    def test_retry_waits_until_due_and_attempt_count_is_retained(self):
        self.store.add_decision(decision())
        self.store.mark_attempt("decision-1", "100")
        with patch("dupcheck_discord.store.time.time", return_value=100):
            self.store.mark_failed("decision-1", "100", "Discord temporarily unavailable", retry_after=10)
            self.assertEqual(self.store.ready("100"), [])
        with patch("dupcheck_discord.store.time.time", return_value=109.9):
            self.assertEqual(self.store.ready("100"), [])
        with patch("dupcheck_discord.store.time.time", return_value=110):
            self.assertEqual(len(self.store.ready("100")), 1)
        self.store.mark_attempt("decision-1", "100")
        self.store.mark_sent("decision-1", "100", "900")
        mapping = self.store.get_notification("900", "100")
        self.assertEqual(mapping["attempts"], 2)
        self.assertIsNone(mapping["next_retry_at"])
        self.assertIsNone(mapping["last_error"])

    def test_decisions_are_immutable_but_identical_replay_is_idempotent(self):
        self.assertTrue(self.store.add_decision(decision()))
        self.assertFalse(self.store.add_decision(decision()))
        with self.assertRaisesRegex(ValueError, "different data"):
            self.store.add_decision(decision(reason="A rewritten explanation"))
        self.assertEqual(len(self.store.ready("100")), 1)
        self.assertEqual(self.store.ready("100")[0]["reason"], decision()["reason"])

    def test_mapping_cannot_be_replaced_by_another_message(self):
        self.add_and_send()
        self.store.mark_sent("decision-1", "100", "900")
        with self.assertRaisesRegex(ValueError, "different Discord message"):
            self.store.mark_sent("decision-1", "100", "901")
        self.assertIsNotNone(self.store.get_notification("900", "100"))
        self.assertIsNone(self.store.get_notification("901", "100"))

    def test_reanalysis_of_sent_commit_is_suppressed_in_same_channel(self):
        self.add_and_send()
        self.store.add_decision(decision(
            decision_id="reanalysis-1", reason="New explanation for the same commit",
        ))
        self.assertEqual(self.store.ready("100"), [])
        self.assertEqual(self.store.ready("200")[0]["decision_id"], "reanalysis-1")
        self.store.add_decision(decision(decision_id="decision-2", head_sha="commit-2"))
        self.assertEqual(self.store.ready("100")[0]["decision_id"], "decision-2")

    def test_optional_git_sha_uses_source_revision_for_delivery_dedup(self):
        source = decision(
            head_sha=None, source_revision="a" * 64, source_dup_ids=[11, 12],
            source_pr_topic_ids=[41, 42], folder_names=["receipt", "billing"],
        )
        self.add_and_send(source)
        self.assertFalse(self.store.add_decision(source))
        self.store.add_decision({**source, "decision_id": "reanalysis"})
        self.assertEqual(self.store.ready("100"), [])
        self.assertIsNone(self.store.ready("200")[0]["head_sha"])
        changed = {**source, "decision_id": "changed-source", "source_revision": "b" * 64}
        self.store.add_decision(changed)
        self.assertEqual(self.store.ready("100")[0], changed)
        with closing(sqlite3.connect(self.path)) as connection:
            head_sha, revision_key = connection.execute(
                "SELECT head_sha, revision_key FROM discord_decisions WHERE decision_id = 'decision-1'"
            ).fetchone()
        self.assertEqual(head_sha, "")
        self.assertEqual(revision_key, "source:" + "a" * 64)

    def test_feedback_view_bridges_source_ids_and_nullable_labels(self):
        source = decision(
            head_sha=None, source_revision="a" * 64, source_dup_ids=[11, 12],
            source_pr_topic_ids=[41, 42], folder_names=["receipt", "billing"],
        )
        self.add_and_send(source)
        self.store.replace_reactions("900", "100", [
            ("correct", "👍"), ("incorrect", "👎"), ("ambiguous", "👍"), ("ambiguous", "👎"),
        ])
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            rows = {
                row["user_id"]: dict(row)
                for row in connection.execute("SELECT * FROM discord_flagger_feedback")
            }
        self.assertEqual(rows["correct"]["is_good"], 1)
        self.assertEqual(rows["incorrect"]["is_good"], 0)
        self.assertIsNone(rows["ambiguous"]["is_good"])
        self.assertEqual(rows["correct"]["decision_id"], "decision-1")
        self.assertEqual(rows["correct"]["message_id"], "900")
        self.assertEqual(json.loads(rows["correct"]["source_dup_ids"]), [11, 12])
        self.assertEqual(rows["correct"]["source_revision"], "a" * 64)
        self.assertIsNone(rows["correct"]["head_sha"])
        self.assertEqual(json.loads(rows["correct"]["payload_json"]), source)
        self.assertTrue(all(row["head_sha"] is None for row in self.store.feedback_rows()))

    def test_legacy_worker_database_migrates_revision_keys_without_losing_rows(self):
        legacy_path = Path(self.temp.name) / "legacy.sqlite3"
        source = decision()
        with closing(sqlite3.connect(legacy_path)) as connection, connection:
            connection.execute(
                """CREATE TABLE discord_decisions (
                    decision_id TEXT PRIMARY KEY, repository TEXT NOT NULL,
                    pr_number INTEGER NOT NULL, head_sha TEXT NOT NULL,
                    status TEXT NOT NULL, is_duplicate INTEGER NOT NULL,
                    pr_state TEXT NOT NULL, model_version TEXT, payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
                )"""
            )
            connection.execute(
                """INSERT INTO discord_decisions
                   (decision_id, repository, pr_number, head_sha, status,
                    is_duplicate, pr_state, model_version, payload_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (source["decision_id"], source["repository"], source["pr_number"], source["head_sha"],
                 source["status"], 1, source["pr_state"], source["model_version"],
                 json.dumps(source, sort_keys=True, ensure_ascii=False, separators=(",", ":"))),
            )
        migrated = Store(legacy_path)
        self.assertEqual(migrated.ready("100"), [source])
        self.assertFalse(migrated.add_decision(source))
        with closing(sqlite3.connect(legacy_path)) as connection:
            self.assertEqual(connection.execute("SELECT revision_key FROM discord_decisions").fetchone()[0], "commit-1")
        migrated.mark_sent("decision-1", "100", "900")
        self.assertTrue(migrated.add_decision({**source, "decision_id": "new-analysis"}))
        self.assertEqual(migrated.ready("100"), [])
        self.assertEqual(Store(legacy_path).ready("100"), [])

    def test_feedback_maps_to_exact_decision_and_commit(self):
        self.add_and_send()
        self.add_and_send(decision(decision_id="decision-2", head_sha="commit-2"), message_id="901")
        self.assertTrue(self.store.record_reaction("900", "100", "maya", "👍", True))
        self.assertTrue(self.store.record_reaction("901", "100", "sam", "👎", True))
        rows = {row["decision_id"]: row for row in Store(self.path).feedback_rows()}
        self.assertEqual(rows["decision-1"]["head_sha"], "commit-1")
        self.assertEqual(rows["decision-1"]["vote"], 1)
        self.assertEqual(rows["decision-2"]["head_sha"], "commit-2")
        self.assertEqual(rows["decision-2"]["vote"], -1)
        self.assertEqual(rows["decision-2"]["message_id"], "901")
        self.assertEqual(rows["decision-2"]["decision"]["model_version"], "qwen-demo")

    def test_unknown_messages_and_unsupported_reactions_are_ignored(self):
        self.add_and_send()
        self.assertFalse(self.store.record_reaction("unknown", "100", "maya", "👍", True))
        self.assertFalse(self.store.record_reaction("900", "wrong-channel", "maya", "👍", True))
        self.assertFalse(self.store.record_reaction("900", "100", "maya", "🔥", True))
        self.assertTrue(self.store.record_reaction("900", "100", "maya", "👍", False))
        self.assertEqual(self.store.feedback_rows(), [])

    def test_dual_reactions_and_removals_are_idempotent(self):
        self.add_and_send()
        with patch("dupcheck_discord.store._now_iso", return_value="first"):
            self.store.record_reaction("900", "100", "maya", "👍", True)
        with patch("dupcheck_discord.store._now_iso", return_value="second"):
            self.store.record_reaction("900", "100", "maya", "👍", True)
        row = self.store.feedback_rows()[0]
        self.assertEqual(row["updated_at"], "first")
        self.assertEqual(row["vote"], 1)
        self.assert_counts(1, 0)
        self.store.record_reaction("900", "100", "maya", "👎", True)
        row = self.store.feedback_rows()[0]
        self.assertEqual((row["thumbs_up"], row["thumbs_down"], row["vote"]), (1, 1, None))
        self.assert_counts(1, 1)
        self.assertEqual(row["created_at"], "first")
        self.store.record_reaction("900", "100", "maya", "👍", False)
        self.assertEqual(self.store.feedback_rows()[0]["vote"], -1)
        self.assert_counts(0, 1)
        self.store.record_reaction("900", "100", "maya", "👎", False)
        row = self.store.feedback_rows()[0]
        self.assertEqual((row["thumbs_up"], row["thumbs_down"], row["vote"]), (0, 0, None))
        self.assert_counts(0, 0)

    def test_clearing_one_reaction_or_all_reactions_updates_all_users(self):
        self.add_and_send()
        self.store.record_reaction("900", "100", "maya", "👍", True)
        self.store.record_reaction("900", "100", "maya", "👎", True)
        self.store.record_reaction("900", "100", "sam", "👍", True)
        self.assert_counts(2, 1)
        self.store.clear_reactions("unknown", "100")
        self.store.clear_reactions("900", "100", "🔥")
        self.assertEqual(len(self.store.feedback_rows()), 2)
        self.assert_counts(2, 1)
        self.store.clear_reactions("900", "100", "👍")
        rows = {row["user_id"]: row for row in self.store.feedback_rows()}
        self.assertEqual(rows["maya"]["vote"], -1)
        self.assertIsNone(rows["sam"]["vote"])
        self.assert_counts(0, 1)
        self.store.clear_reactions("900", "100")
        self.assertTrue(all(row["vote"] is None for row in self.store.feedback_rows()))
        self.assertTrue(all(row["thumbs_down"] == 0 for row in self.store.feedback_rows()))
        self.assert_counts(0, 0)

    def test_snapshot_reconciliation_preserves_unchanged_timestamps(self):
        self.add_and_send()
        with patch("dupcheck_discord.store._now_iso", return_value="first"):
            self.store.replace_reactions("900", "100", [
                ("maya", "👍"), ("sam", "👎"), ("alex", "👍"), ("unsupported", "🔥"),
            ])
        self.assert_counts(2, 1)
        with patch("dupcheck_discord.store._now_iso", return_value="second"):
            self.store.replace_reactions("900", "100", [
                ("maya", "👍"), ("maya", "👍"),
                ("sam", "👍"), ("sam", "👎"), ("new-reviewer", "👎"),
            ])
        rows = {row["user_id"]: row for row in self.store.feedback_rows()}
        self.assertEqual(set(rows), {"maya", "sam", "alex", "new-reviewer"})
        self.assertEqual((rows["maya"]["vote"], rows["maya"]["updated_at"]), (1, "first"))
        self.assertEqual((rows["sam"]["vote"], rows["sam"]["updated_at"]), (None, "second"))
        self.assertEqual(rows["sam"]["created_at"], "first")
        self.assertEqual((rows["alex"]["thumbs_up"], rows["alex"]["thumbs_down"], rows["alex"]["vote"]), (0, 0, None))
        self.assertEqual(rows["alex"]["updated_at"], "second")
        self.assertEqual(rows["alex"]["created_at"], "first")
        self.assertEqual((rows["new-reviewer"]["vote"], rows["new-reviewer"]["created_at"]), (-1, "second"))
        self.assert_counts(2, 2)
        with patch("dupcheck_discord.store._now_iso", return_value="third"):
            self.store.replace_reactions("900", "100", [
                ("maya", "👍"), ("sam", "👍"), ("sam", "👎"), ("new-reviewer", "👎"),
            ])
        self.assertEqual(self.store.feedback_rows(), list(rows.values()))
        self.assert_counts(2, 2)

    def test_empty_snapshot_clears_flags_and_unknown_snapshot_is_ignored(self):
        self.add_and_send()
        self.store.replace_reactions("900", "100", [("maya", "👍")])
        self.store.replace_reactions("unknown", "100", [])
        self.store.replace_reactions("900", "wrong-channel", [])
        self.assertEqual(self.store.feedback_rows()[0]["vote"], 1)
        self.assert_counts(1, 0)
        self.store.replace_reactions("900", "100", [])
        row = self.store.feedback_rows()[0]
        self.assertEqual((row["thumbs_up"], row["thumbs_down"], row["vote"]), (0, 0, None))
        self.assert_counts(0, 0)

    def test_snapshot_rolls_back_all_changes_when_any_write_fails(self):
        self.add_and_send()
        self.store.replace_reactions("900", "100", [("maya", "👍")])
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                """CREATE TRIGGER test_snapshot_failure BEFORE INSERT ON discord_feedback
                   WHEN NEW.user_id = 'z-failed'
                   BEGIN SELECT RAISE(ABORT, 'injected snapshot failure'); END"""
            )
        with self.assertRaisesRegex(sqlite3.IntegrityError, "injected snapshot failure"):
            self.store.replace_reactions("900", "100", [("maya", "👎"), ("z-failed", "👎")])
        rows = self.store.feedback_rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["user_id"], rows[0]["vote"]), ("maya", 1))
        self.assert_counts(1, 0)

    def test_feedback_delete_and_key_update_refresh_only_affected_messages(self):
        self.add_and_send()
        self.add_and_send(decision(decision_id="decision-2", head_sha="commit-2"), message_id="901")
        self.add_and_send(decision(), message_id="902", channel_id="200")
        self.store.record_reaction("900", "100", "maya", "👍", True)
        self.store.record_reaction("900", "100", "maya", "👎", True)
        self.store.record_reaction("900", "100", "alex", "👍", True)
        self.store.record_reaction("901", "100", "sam", "👎", True)
        self.store.record_reaction("902", "200", "other-channel", "👍", True)
        self.assert_counts(2, 1)
        self.assert_counts(0, 1, message_id="901")
        self.assert_counts(1, 0, message_id="902", channel_id="200")
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("DELETE FROM discord_feedback WHERE user_id = 'maya'")
        self.assert_counts(1, 0)
        self.assert_counts(0, 1, message_id="901")
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(
                "UPDATE discord_feedback SET decision_id = 'decision-2' WHERE user_id = 'alex'"
            )
        self.assert_counts(0, 0)
        self.assert_counts(1, 1, message_id="901")
        self.assert_counts(1, 0, message_id="902", channel_id="200")

    def test_reinitialization_preserves_counts_and_repairs_outdated_totals(self):
        self.add_and_send()
        self.store.replace_reactions("900", "100", [
            ("maya", "👍"), ("maya", "👎"), ("sam", "👍"),
        ])
        self.assert_counts(2, 1)
        before = self.store.feedback_rows()
        restarted = Store(self.path)
        self.assert_counts(2, 1, store=restarted)
        self.assertEqual(restarted.feedback_rows(), before)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("UPDATE discord_notifications SET like_count = 99, dislike_count = 88")
        restarted = Store(self.path)
        self.assert_counts(2, 1, store=restarted)
        self.assertEqual(restarted.feedback_rows(), before)
        restarted.mark_sent("decision-1", "100", "900")
        restarted.record_reaction("900", "100", "sam", "👍", True)
        self.assert_counts(2, 1, store=restarted)

    def test_existing_shared_database_adds_and_backfills_count_columns(self):
        legacy_path = Path(self.temp.name) / "legacy-counts.sqlite3"
        with closing(sqlite3.connect(legacy_path)) as connection, connection:
            connection.executescript("""
                CREATE TABLE teammate_decisions (id INTEGER, note TEXT);
                INSERT INTO teammate_decisions VALUES (1, 'keep this');
                CREATE TABLE discord_decisions (
                    decision_id TEXT PRIMARY KEY, repository TEXT NOT NULL,
                    pr_number INTEGER NOT NULL, head_sha TEXT NOT NULL,
                    revision_key TEXT NOT NULL, status TEXT NOT NULL,
                    is_duplicate INTEGER NOT NULL, pr_state TEXT NOT NULL,
                    model_version TEXT, payload_json TEXT NOT NULL, created_at TEXT
                );
                CREATE TABLE discord_notifications (
                    decision_id TEXT NOT NULL, channel_id TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER DEFAULT 0,
                    message_id TEXT, last_error TEXT, next_retry_at REAL,
                    last_attempt_at TEXT, sent_at TEXT, created_at TEXT, updated_at TEXT,
                    PRIMARY KEY (decision_id, channel_id), UNIQUE (channel_id, message_id)
                );
                CREATE TABLE discord_feedback (
                    decision_id TEXT NOT NULL, channel_id TEXT NOT NULL, user_id TEXT NOT NULL,
                    thumbs_up INTEGER NOT NULL, thumbs_down INTEGER NOT NULL, vote INTEGER,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY (decision_id, channel_id, user_id),
                    FOREIGN KEY (decision_id, channel_id)
                        REFERENCES discord_notifications(decision_id, channel_id)
                );
            """)
            for index in range(1, 3):
                payload = decision(decision_id=f"decision-{index}", head_sha=f"commit-{index}")
                connection.execute(
                    "INSERT INTO discord_decisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (payload["decision_id"], payload["repository"], payload["pr_number"],
                     payload["head_sha"], payload["head_sha"], payload["status"], 1,
                     payload["pr_state"], payload["model_version"], json.dumps(payload), "original"),
                )
                connection.execute(
                    "INSERT INTO discord_notifications (decision_id, channel_id, status, message_id) "
                    "VALUES (?, '100', 'sent', ?)",
                    (payload["decision_id"], str(899 + index)),
                )
            connection.executemany(
                "INSERT INTO discord_feedback VALUES ('decision-1', '100', ?, ?, ?, ?, 'created', 'updated')",
                [("maya", 1, 1, None), ("sam", 1, 0, 1), ("removed", 0, 0, None)],
            )
        migrated = Store(legacy_path)
        self.assert_counts(2, 1, store=migrated)
        self.assert_counts(0, 0, message_id="901", store=migrated)
        self.assertEqual(len(migrated.feedback_rows()), 3)
        with closing(sqlite3.connect(legacy_path)) as connection:
            self.assertEqual(connection.execute("SELECT * FROM teammate_decisions").fetchall(), [(1, "keep this")])
            self.assertEqual(connection.execute("SELECT DISTINCT created_at, updated_at FROM discord_feedback").fetchall(), [("created", "updated")])
            triggers = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'trigger' AND name LIKE 'discord_feedback_counts_%'"
            ).fetchall()
        self.assertEqual(len(triggers), 3)
        migrated.record_reaction("900", "100", "maya", "👍", False)
        self.assert_counts(1, 1, store=migrated)
        self.assert_counts(1, 1, store=Store(legacy_path))

    def test_close_pr_stops_notifications_without_rewriting_decision(self):
        self.store.add_decision(decision())
        self.store.close_pr("team/projects", 42, "merged")
        self.assertEqual(self.store.ready("100"), [])
        self.store.add_decision(decision())  # replay cannot silently reopen the PR
        self.assertEqual(self.store.ready("100"), [])
        with closing(sqlite3.connect(self.path)) as connection, connection:
            payload = json.loads(connection.execute("SELECT payload_json FROM discord_decisions").fetchone()[0])
        self.assertEqual(payload["pr_state"], "open")
        self.store.close_pr("team/projects", 42, "open")
        self.assertEqual(len(self.store.ready("100")), 1)

    def test_existing_team_tables_are_preserved_and_foreign_keys_are_enforced(self):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("CREATE TABLE teammate_decisions (id INTEGER, note TEXT)")
            connection.execute("INSERT INTO teammate_decisions VALUES (1, 'keep this')")
        Store(self.path)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            self.assertEqual(connection.execute("PRAGMA journal_mode").fetchone()[0], "wal")
            self.assertEqual(connection.execute("SELECT note FROM teammate_decisions").fetchone()[0], "keep this")
        with self.assertRaises(sqlite3.IntegrityError):
            self.store.mark_sent("unknown-decision", "100", "900")


if __name__ == "__main__":
    unittest.main()
