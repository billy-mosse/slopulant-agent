from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from dupcheck_discord.source import build_decisions, sync_source
from dupcheck_discord.store import Store


TEAM_SCHEMA = """
CREATE TABLE Commited (
    topic_id INTEGER PRIMARY KEY, commited_topic TEXT, topic_desc TEXT, owner TEXT
);
CREATE TABLE new_prs (
    pr_id INTEGER, topic_id INTEGER, folder_name TEXT, topic_desc TEXT,
    owner TEXT, timestamp TEXT DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY (pr_id, topic_id)
);
CREATE TABLE dup_cg (
    dup_id INTEGER PRIMARY KEY, pr_id INTEGER, t_id INTEGER REFERENCES Commited(topic_id),
    incumbent_topic TEXT, pr_topic TEXT
);
CREATE TABLE dupe_decision (
    dup_id INTEGER PRIMARY KEY REFERENCES dup_cg(dup_id), boolean INTEGER
);
"""


class SourceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "my database.db"
        self.store = Store(self.path)
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.executescript(TEAM_SCHEMA)
            connection.execute("INSERT INTO Commited VALUES (1, 'Password reset', 'Send reset links by email.', 'alice')")
            connection.execute("INSERT INTO new_prs VALUES (101, 1001, 'auth/reset', 'Send password recovery links by email.', 'frank', '2026-10-01')")
            connection.execute("INSERT INTO dup_cg VALUES (1, 101, 1, 'Password reset', 'Email password recovery')")
            connection.execute("INSERT INTO dupe_decision VALUES (1, 1)")

    def execute(self, statement, parameters=()):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(statement, parameters)

    def decisions(self):
        with closing(sqlite3.connect(self.path)) as connection:
            return build_decisions(connection)

    def stored_decisions(self):
        with closing(sqlite3.connect(self.path)) as connection:
            return [json.loads(row[0]) for row in connection.execute(
                "SELECT payload_json FROM discord_decisions ORDER BY rowid",
            )]

    def add_comparison(self, *, dup_id=2, topic_id=2, verdict=1, pr_number=101):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute("INSERT INTO Commited VALUES (?, 'CSV export', 'Export rows as CSV.', 'bob')", (topic_id,))
            connection.execute("INSERT INTO dup_cg VALUES (?, ?, ?, 'CSV export', 'Customer CSV download')", (dup_id, pr_number, topic_id))
            if verdict is not None:
                connection.execute("INSERT INTO dupe_decision VALUES (?, ?)", (dup_id, verdict))

    def test_real_schema_uses_incumbent_topic_and_actual_pr_author(self):
        payload = self.decisions()[0]
        self.assertEqual(payload["pr_number"], 101)
        self.assertEqual(payload["author_login"], "frank")
        self.assertEqual(payload["topic"], "Email password recovery")
        self.assertEqual(payload["source_pr_topic_ids"], [1001])
        self.assertEqual(payload["source_dup_ids"], [1])
        self.assertEqual(payload["matches"][0]["source_topic_id"], 1)
        self.assertEqual(payload["matches"][0]["source_dup_id"], 1)
        self.assertIn("Send reset links by email.", payload["matches"][0]["evidence"])
        self.assertIsNone(payload["head_sha"])
        self.assertIsNone(payload["pr_url"])
        self.assertEqual(payload["duplicate_kind"], "unspecified")
        self.assertEqual(payload["pr_description"], "Send password recovery links by email.")

    def test_multiple_matches_and_pr_folders_are_one_snapshot(self):
        self.add_comparison()
        self.execute("INSERT INTO new_prs VALUES (101, 1002, 'exports/customers', 'Export customers.', 'frank', '2026-10-01')")
        payloads = self.decisions()
        self.assertEqual(len(payloads), 1)
        self.assertEqual(len(payloads[0]["matches"]), 2)
        self.assertEqual(payloads[0]["source_dup_ids"], [1, 2])
        self.assertEqual(payloads[0]["source_pr_topic_ids"], [1001, 1002])
        self.assertEqual(payloads[0]["folder_names"], ["auth/reset", "exports/customers"])

    def test_clean_candidates_are_not_attached_to_duplicate_feedback(self):
        self.add_comparison(verdict=0)
        payload = self.decisions()[0]
        self.assertEqual(payload["source_dup_ids"], [1])
        self.assertEqual(len(payload["matches"]), 1)

    def test_clean_verdict_is_ingested_and_never_eligible_for_delivery(self):
        self.execute("UPDATE dupe_decision SET boolean = 0")
        self.assertEqual(sync_source(self.store), 1)
        payload = self.stored_decisions()[0]
        self.assertFalse(payload["is_duplicate"])
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["matches"], [])
        self.assertEqual(self.store.ready("100"), [])

    def test_identical_polls_do_not_add_or_resend_a_snapshot(self):
        self.assertEqual(sync_source(self.store), 1)
        self.assertEqual(sync_source(self.store), 0)
        self.assertEqual(len(self.stored_decisions()), 1)
        payload = self.store.ready("100")[0]
        self.store.mark_sent(payload["decision_id"], "100", "900")
        self.assertEqual(sync_source(self.store), 0)
        self.assertEqual(self.store.ready("100"), [])

    def test_negative_update_invalidates_pending_positive(self):
        sync_source(self.store)
        self.assertEqual(len(self.store.ready("100")), 1)
        self.execute("UPDATE dupe_decision SET boolean = 0")
        self.assertEqual(sync_source(self.store), 1)
        self.assertEqual(self.store.ready("100"), [])

    def test_missing_verdict_holds_the_entire_pr_pending(self):
        sync_source(self.store)
        self.add_comparison(verdict=None)
        self.assertEqual(sync_source(self.store), 1)
        self.assertEqual(self.stored_decisions()[-1]["status"], "pending")
        self.assertEqual(self.store.ready("100"), [])
        self.execute("INSERT INTO dupe_decision VALUES (2, 0)")
        self.assertEqual(sync_source(self.store), 1)
        self.assertEqual(len(self.store.ready("100")), 1)

    def test_reverting_boolean_creates_latest_immutable_snapshot(self):
        sync_source(self.store)
        self.execute("UPDATE dupe_decision SET boolean = 0")
        sync_source(self.store)
        self.execute("UPDATE dupe_decision SET boolean = 1")
        self.assertEqual(sync_source(self.store), 1)
        payloads = self.stored_decisions()
        self.assertEqual(len(payloads), 3)
        self.assertEqual(payloads[0]["source_revision"], payloads[2]["source_revision"])
        self.assertNotEqual(payloads[0]["decision_id"], payloads[2]["decision_id"])
        self.assertEqual(self.store.ready("100")[0]["decision_id"], payloads[2]["decision_id"])

    def test_changed_source_metadata_creates_a_new_revision(self):
        sync_source(self.store)
        self.execute("UPDATE Commited SET topic_desc = 'Use the shared reset email service.'")
        self.assertEqual(sync_source(self.store), 1)
        payloads = self.stored_decisions()
        self.assertNotEqual(payloads[0]["source_revision"], payloads[1]["source_revision"])
        self.assertIn("shared reset email service", payloads[1]["matches"][0]["evidence"])

    def test_source_disappearance_disables_stale_delivery_and_reappearance_restores(self):
        sync_source(self.store)
        self.execute("DELETE FROM new_prs")
        self.assertEqual(sync_source(self.store), 0)
        self.assertEqual(self.store.ready("100"), [])
        self.execute("INSERT INTO new_prs VALUES (101, 1001, 'auth/reset', 'Send password recovery links by email.', 'frank', '2026-10-01')")
        self.assertEqual(sync_source(self.store), 1)
        self.assertEqual(len(self.store.ready("100")), 1)

    def test_candidate_disappearance_disables_stale_delivery(self):
        sync_source(self.store)
        self.execute("DELETE FROM dup_cg")
        sync_source(self.store)
        self.assertEqual(self.store.ready("100"), [])

    def test_orphan_comparison_waits_for_pr_metadata(self):
        self.add_comparison(pr_number=102)
        self.assertEqual([payload["pr_number"] for payload in self.decisions()], [101])

    def test_multiple_authors_are_rejected_instead_of_guessed(self):
        self.execute("INSERT INTO new_prs VALUES (101, 1002, 'other', 'Other topic.', 'someone_else', '2026-10-01')")
        with self.assertRaisesRegex(ValueError, "exactly one nonempty author"):
            self.decisions()

    def test_malformed_boolean_is_not_treated_as_truthy(self):
        self.execute("UPDATE dupe_decision SET boolean = 2")
        with self.assertRaisesRegex(ValueError, "integer 0 or 1"):
            self.decisions()

    def test_missing_team_tables_fail_clearly(self):
        self.execute("DROP TABLE dupe_decision")
        with self.assertRaisesRegex(ValueError, "missing.*dupe_decision"):
            self.decisions()

    def test_missing_required_column_fails_clearly(self):
        self.execute("ALTER TABLE new_prs RENAME COLUMN owner TO something_else")
        with self.assertRaisesRegex(ValueError, "missing columns: owner"):
            self.decisions()

    def test_build_only_reads_source_and_preserves_callers_transaction(self):
        with closing(sqlite3.connect(self.path)) as connection:
            connection.execute("BEGIN")
            before = connection.total_changes
            first = build_decisions(connection)
            second = build_decisions(connection)
            self.assertEqual(first, second)
            self.assertEqual(connection.total_changes, before)
            self.assertTrue(connection.in_transaction)

    def test_repository_config_is_part_of_snapshot_identity(self):
        self.assertEqual(sync_source(self.store, "team/one"), 1)
        self.assertEqual(sync_source(self.store, "team/two"), 1)
        payloads = self.stored_decisions()
        self.assertNotEqual(payloads[0]["decision_id"], payloads[1]["decision_id"])
        self.assertEqual(len(self.store.ready("100")), 2)


if __name__ == "__main__":
    unittest.main()
