import ast
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from dupcheck_discord.source import build_decisions, sync_source
from dupcheck_discord.store import Store


REPO = "acme/monorepo"
HEAD, BASE = "a" * 40, "b" * 40
VERSION = "clm-v0.1-8B/sys-same_problem/th0.19"


def watcher_schema():
    """The watcher's real SCHEMA constant, read without importing its dependencies."""
    source = (Path(__file__).resolve().parents[2] / "watcher" / "db.py").read_text()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "SCHEMA":
            return node.value.value
    raise AssertionError("watcher/db.py has no SCHEMA")


class SourceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "watcher.db"
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.executescript(watcher_schema())
        self.store = Store(self.path)
        self.pr(9, "[TEST] SEO titles", "frank")
        self.folder("tree-pr", "seo_titles", "SEO Titles", "Writes SEO titles for product pages.")
        self.folder("tree-main", "title_rewriter", "Title Rewriter", "Rewrites product titles for search.")
        self.folder("tree-main2", "attribute_extraction", "Attributes", "Extracts product attributes.")
        self.run_scores(9, HEAD, BASE, [("seo_titles", "title_rewriter", 1, None), ("seo_titles", "attribute_extraction", 1, "upstream:attrs"),
                                         ("seo_titles", "churn", 0, None)])

    # ---- fixture helpers (the watcher and CLM worker write these rows in production)
    def sql(self, statement, *params):
        with closing(sqlite3.connect(self.path)) as connection, connection:
            connection.execute(statement, params)

    def pr(self, number, title, author, state="open"):
        self.sql("INSERT OR REPLACE INTO prs VALUES (?, ?, ?, 'branch', ?, ?, ?, NULL, NULL)",
                 number, title, author, f"https://github.com/{REPO}/pull/{number}", HEAD, state)

    def folder(self, tree, folder, topic, description):
        self.sql("INSERT INTO folders VALUES (?, 'm', 'e', ?, 1, '[]', x'', 0)", tree, folder)
        self.sql("INSERT INTO topics VALUES (?, 'm', 'e', 0, ?, ?, x'', '[]', '[]', '[]')", tree, topic, description)

    def run_scores(self, number, head, base, pairs):
        self.sql("INSERT INTO pr_queue VALUES (?, ?, ?, 'done', NULL, 1, 1)", number, head, f"{base}|v")
        topics = {"seo_titles": "SEO Titles", "title_rewriter": "Title Rewriter", "attribute_extraction": "Attributes", "churn": "Churn"}
        for rank, (pr_folder, repo_id, candidate, dataflow) in enumerate(pairs, 1):
            self.sql("INSERT INTO scores VALUES (1, ?, ?, ?, ?, ?, 0.6, 0.7, 0.5, 'titles', ?, ?, 0.1, NULL, ?, ?, ?)",
                     number, head, pr_folder, repo_id, base, topics[pr_folder], topics[repo_id], dataflow, rank, candidate)

    def decide(self, repo_id, is_dupe, score=0.6, classifier=VERSION, number=9, head=HEAD):
        relation = "upstream" if classifier == "dataflow" else ("duplicate" if is_dupe else "unrelated")
        self.sql("INSERT OR REPLACE INTO decisions VALUES (?, ?, ?, 'seo_titles', ?, ?, ?, ?, 'r', ?, 1, ?)",
                 number, head, BASE, repo_id, relation, int(is_dupe), score, classifier, None if classifier == "dataflow" else 0.19)

    def finish(self, note="Overlaps with title_rewriter; talk to its owner.", head=HEAD, number=9):
        if note:
            self.sql("INSERT OR REPLACE INTO alerts VALUES (?, ?, ?, ?, ?, 'openclaw:watcher', NULL, 1)", number, head, BASE, note, note)
        self.sql("INSERT OR REPLACE INTO kv VALUES (?, ?)", f"clm_finished:{head}:{BASE}", VERSION)

    def judge_all(self, dupe=True):
        self.decide("title_rewriter", dupe)
        self.decide("attribute_extraction", False, 1.0, "dataflow")
        self.finish(note="Overlaps with title_rewriter; talk to its owner." if dupe else None)

    def preview(self):
        with closing(sqlite3.connect(self.path)) as connection:
            return build_decisions(connection, REPO)

    def stored(self):
        with closing(sqlite3.connect(self.path)) as connection:
            return [json.loads(row[0]) for row in connection.execute("SELECT payload_json FROM discord_decisions ORDER BY rowid")]

    # ---- tests
    def test_duplicate_snapshot_carries_commit_link_note_and_evidence(self):
        self.judge_all()
        [payload] = self.preview()
        self.assertEqual(payload["status"], "completed")
        self.assertTrue(payload["is_duplicate"])
        self.assertEqual((payload["pr_number"], payload["head_sha"], payload["base_sha"]), (9, HEAD, BASE))
        self.assertEqual(payload["pr_url"], f"https://github.com/{REPO}/pull/9")
        self.assertEqual(payload["author_login"], "frank")
        self.assertEqual(payload["reason"], "Overlaps with title_rewriter; talk to its owner.")
        self.assertEqual(payload["model_version"], VERSION)
        self.assertEqual(payload["source_decision_keys"], ["seo_titles/title_rewriter"])
        [match] = payload["matches"]  # the dataflow link is not a duplicate
        self.assertEqual(match["project_name"], "title_rewriter")
        self.assertEqual(match["url"], f"https://github.com/{REPO}/tree/{BASE}/title_rewriter")
        self.assertIn("Rewrites product titles for search.", match["topic"])
        self.assertIn("CLM same-problem score 0.60 (threshold 0.19)", match["evidence"])
        self.assertIn("Writes SEO titles for product pages.", payload["pr_description"])

    def test_unfinished_commit_is_pending_and_not_delivered(self):
        self.decide("title_rewriter", True)  # one of two candidates judged, no finished marker
        self.assertEqual(self.preview()[0]["status"], "pending")
        self.assertEqual(sync_source(self.store, REPO), 1)
        self.assertEqual(self.store.ready("100"), [])
        self.decide("attribute_extraction", False, 1.0, "dataflow")
        self.finish()
        self.assertEqual(sync_source(self.store, REPO), 1)
        self.assertEqual(len(self.store.ready("100")), 1)

    def test_duplicate_waits_for_the_openclaw_note(self):
        self.decide("title_rewriter", True)
        self.decide("attribute_extraction", False, 1.0, "dataflow")
        self.finish(note=None)
        self.assertEqual(self.preview()[0]["status"], "pending")

    def test_verdicts_from_another_classifier_are_not_final(self):
        self.judge_all()
        self.decide("title_rewriter", True, classifier="llm-dummy-v1")  # stale verdict awaiting re-judging
        self.assertEqual(self.preview()[0]["status"], "pending")

    def test_clean_verdict_is_ingested_and_never_eligible_for_delivery(self):
        self.judge_all(dupe=False)
        self.assertEqual(sync_source(self.store, REPO), 1)
        [payload] = self.stored()
        self.assertFalse(payload["is_duplicate"])
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["matches"], [])
        self.assertEqual(self.store.ready("100"), [])

    def test_pr_without_candidates_is_clean(self):
        self.pr(10, "[TEST] unrelated", "grace")
        self.run_scores(10, "c" * 40, BASE, [("seo_titles", "churn", 0, None)])
        payload = next(p for p in self.preview() if p["pr_number"] == 10)
        self.assertEqual((payload["status"], payload["is_duplicate"]), ("completed", False))

    def test_identical_polls_do_not_add_or_resend_a_snapshot(self):
        self.judge_all()
        self.assertEqual(sync_source(self.store, REPO), 1)
        self.assertEqual(sync_source(self.store, REPO), 0)
        self.assertEqual(len(self.stored()), 1)

    def test_changed_verdict_supersedes_pending_alert(self):
        self.judge_all()
        sync_source(self.store, REPO)
        self.assertEqual(len(self.store.ready("100")), 1)
        self.decide("title_rewriter", False, 0.1)
        self.assertEqual(sync_source(self.store, REPO), 1)
        self.assertEqual(self.store.ready("100"), [])
        payloads = self.stored()
        self.assertNotEqual(payloads[0]["decision_id"], payloads[1]["decision_id"])

    def test_closed_or_merged_pr_stops_delivery_and_reopening_restores(self):
        self.judge_all()
        sync_source(self.store, REPO)
        self.pr(9, "[TEST] SEO titles", "frank", state="merged")
        self.assertEqual(sync_source(self.store, REPO), 0)
        self.assertEqual(self.store.ready("100"), [])
        self.pr(9, "[TEST] SEO titles", "frank", state="open")
        self.assertEqual(sync_source(self.store, REPO), 1)
        self.assertEqual(len(self.store.ready("100")), 1)

    def test_feedback_view_links_back_to_watcher_decisions(self):
        self.judge_all()
        sync_source(self.store, REPO)
        [decision] = self.store.ready("100")
        self.store.mark_attempt(decision["decision_id"], "100")
        self.store.mark_sent(decision["decision_id"], "100", "555")
        self.store.record_reaction("555", "100", "42", "👎", True)
        with closing(sqlite3.connect(self.path)) as connection:
            connection.row_factory = sqlite3.Row
            [row] = connection.execute("SELECT * FROM discord_flagger_feedback").fetchall()
        self.assertEqual((row["pr_number"], row["head_sha"], row["base_sha"], row["is_good"]), (9, HEAD, BASE, 0))
        self.assertEqual(json.loads(row["source_decision_keys"]), ["seo_titles/title_rewriter"])

    def test_wrong_database_fails_clearly(self):
        other = Path(self.temp.name) / "other.db"
        Store(other)
        with closing(sqlite3.connect(other)) as connection, self.assertRaisesRegex(ValueError, "Not a watcher database"):
            build_decisions(connection, REPO)


if __name__ == "__main__":
    unittest.main()
