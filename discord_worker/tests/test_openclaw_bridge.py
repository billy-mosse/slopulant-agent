import unittest
from unittest import mock

from dupcheck_discord import openclaw_bridge as bridge

DECISION = {
    "decision_id": "pr-7-aaaa-bbbb-v1", "pr_number": 7, "pr_title": "[TEST] Add review ring detector",
    "pr_url": "https://github.com/acme/monorepo/pull/7", "author_login": "billy", "author_name": "Leo Martins",
    "head_sha": "849df079361c", "folder_names": ["review_ring_detector"], "reason": "fallback text",
    "overlap": "Both find review rings.", "next_step": "Talk to Ines Duarte.",
    "matches": [{"project_name": "fake_review_detection", "url": "https://github.com/acme/monorepo/tree/b/fake_review_detection",
                 "topic": "ring detection — long description", "evidence": "Owner: Ines Duarte\nCLM same-problem score 0.65 (threshold 0.19)\nMatched PR topic: x"}],
}


class PresentationTest(unittest.TestCase):
    def test_card_has_facts_and_drafted_parts(self):
        texts = [b.get("text", "") for b in bridge.presentation(DECISION)["blocks"]]
        joined = "\n".join(texts)
        self.assertIn("[PR #7: Add review ring detector](https://github.com/acme/monorepo/pull/7)", texts[0])
        self.assertIn("**What overlaps:** Both find review rings.", joined)
        self.assertIn("**Suggested next step:** Talk to Ines Duarte.", joined)
        self.assertIn("Owner: Ines Duarte · CLM same-problem score 0.65 (threshold 0.19)", joined)
        self.assertNotIn("Matched PR topic", joined)
        self.assertIn("pr-7-aaaa-bbbb-v1", texts[-1])

    def test_without_drafted_parts_falls_back_to_reason(self):
        decision = {k: v for k, v in DECISION.items() if k not in ("overlap", "next_step")}
        self.assertEqual(bridge.presentation(decision)["blocks"][1]["text"], "fallback text")


class ReactionsTest(unittest.TestCase):
    def test_bot_and_unlisted_users_are_ignored(self):
        store = mock.Mock()
        store.notifications.return_value = [{"message_id": "555"}]
        snapshot = {"payload": {"reactions": [
            {"emoji": {"name": "👍"}, "users": [{"id": "1", "username": "slop-factory"}, {"id": "42", "username": "ana"}]},
            {"emoji": {"name": "👎"}, "users": [{"id": "43", "username": "ben"}]},
            {"emoji": {"name": "🎉"}, "users": [{"id": "44", "username": "cy"}]}]}}
        with mock.patch.object(bridge, "_cli", return_value=snapshot):
            self.assertEqual(bridge.sync_reactions(store, "100"), 1)
        store.replace_reactions.assert_called_once_with("555", "100", [("42", "👍"), ("43", "👎")])
        with mock.patch.object(bridge, "_cli", return_value=snapshot):
            bridge.sync_reactions(store, "100", reviewer_ids=frozenset({"43"}))
        self.assertEqual(store.replace_reactions.call_args.args[2], [("43", "👎")])


if __name__ == "__main__":
    unittest.main()
