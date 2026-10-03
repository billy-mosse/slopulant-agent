import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

from dupcheck_discord.bot import DiscordWorker, select_named_channel
from dupcheck_discord.config import Config
from dupcheck_discord.messages import build_alert, marker, preview_alert
from dupcheck_discord.models import validate_decision
from dupcheck_discord.store import Store


EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "duplicate.json"


def decision():
    return validate_decision(json.loads(EXAMPLE.read_text()))


class ContractTest(unittest.TestCase):
    def test_strings_cannot_turn_clean_decisions_into_alerts(self):
        payload = decision()
        for value in ("false", "true", 0, 1, None):
            payload["is_duplicate"] = value
            with self.assertRaises(ValueError):
                validate_decision(payload)

    def test_duplicates_require_explanation_and_matching_project(self):
        payload = decision()
        for key, value in (("matches", []), ("reason", ""), ("status", [])):
            with self.subTest(key=key):
                bad = copy.deepcopy(payload)
                bad[key] = value
                with self.assertRaises(ValueError):
                    validate_decision(bad)

    def test_only_explicit_author_can_be_pinged(self):
        payload = decision()
        payload["author_name"] = "@everyone"
        self.assertEqual(preview_alert(payload)["allowed_mentions"], {"parse": []})
        payload["discord_user_id"] = "123456789012345678"
        preview = preview_alert(payload)
        self.assertIn("<@123456789012345678>", preview["content"])
        self.assertEqual(preview["allowed_mentions"]["users"], [123456789012345678])
        self.assertEqual(preview["allowed_mentions"]["parse"], [])

    def test_large_evidence_fits_discord_limits(self):
        payload = decision()
        payload["pr_title"] = "*" * 500
        payload["reason"] = "A" * 6000
        payload["author_name"] = "B" * 256
        payload["topic"] = "C" * 256
        match = payload["matches"][0]
        match["project_name"] = "*" * 256
        match["evidence"] = "D" * 6000
        payload["matches"] = [copy.deepcopy(match) for _ in range(25)]
        embed = build_alert(payload)["embed"]
        self.assertLessEqual(len(embed), 6000)
        self.assertLessEqual(len(embed.title), 256)
        self.assertLessEqual(len(embed.description), 4096)
        self.assertLessEqual(len(embed.fields), 25)
        for field in embed.fields:
            self.assertLessEqual(len(field.name), 256)
            self.assertLessEqual(len(field.value), 1024)

    def test_clean_decision_cannot_render_an_alert(self):
        payload = decision()
        payload["is_duplicate"] = False
        with self.assertRaises(ValueError):
            build_alert(payload)

    def test_source_snapshot_is_not_presented_as_a_git_commit(self):
        payload = decision()
        payload.update(head_sha=None, pr_url=None, source_revision="a" * 64,
                       duplicate_kind="unspecified", source_dup_ids=[4],
                       pr_description="Throttle API requests per user.")
        payload = validate_decision(payload)
        preview = preview_alert(payload)
        embed = preview["embeds"][0]
        self.assertNotIn("url", embed)
        self.assertIn("source revision", embed["footer"]["text"])
        self.assertNotIn("commit ", embed["footer"]["text"])
        self.assertEqual(payload["source_dup_ids"], [4])
        self.assertIn("Scope needs review", [field["value"] for field in embed["fields"]])

    def test_channel_name_must_resolve_uniquely(self):
        channel = SimpleNamespace(name="jfreds_discord", permissions_for=lambda member: SimpleNamespace(view_channel=True))
        guild = SimpleNamespace(me=object(), text_channels=[channel])
        self.assertIs(select_named_channel([guild], "jfreds_discord"), channel)
        for guilds, name in (([guild, guild], "jfreds_discord"), ([guild], "missing")):
            with self.assertRaises(ValueError):
                select_named_channel(guilds, name)

    def test_config_never_prints_token_and_rejects_bad_intervals(self):
        with patch.dict(os.environ, {"DISCORD_BOT_TOKEN": "private-test-token"}, clear=True):
            config = Config.from_env()
            self.assertEqual(config.channel_name, "slop-factory")
            self.assertNotIn("private-test-token", repr(config))
            for value in ("nan", "inf", "0"):
                os.environ["POLL_INTERVAL_SECONDS"] = value
                with self.assertRaises(ValueError):
                    Config.from_env()

    def test_close_pr_cannot_silently_reopen_in_team_source_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unused.sqlite3"
            result = subprocess.run(
                [sys.executable, "-m", "dupcheck_discord", "--env-file", str(Path(directory) / "missing.env"),
                 "--db", str(path), "close-pr", "team repository", "101"],
                env={**os.environ, "DUPCHECK_SOURCE": "team_sqlite"}, capture_output=True, text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("close-pr is for inbox mode", result.stderr)
            self.assertFalse(path.exists())


class WorkerTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.store = Store(Path(self.temp.name) / "test.sqlite3")
        self.decision = decision()
        self.store.add_decision(self.decision)
        self.worker = DiscordWorker(self.store, Config("test", channel_id=777))
        self.worker._connection.user = SimpleNamespace(id=999)
        self.message = SimpleNamespace(id=1234, add_reaction=AsyncMock())
        self.worker.channel = SimpleNamespace(id=777, send=AsyncMock(return_value=self.message))

    async def asyncTearDown(self):
        await self.worker.close()
        self.temp.cleanup()

    def payload(self, emoji="👍", user_id=222, bot=False):
        return SimpleNamespace(channel_id=777, message_id=1234, user_id=user_id,
                               emoji=emoji, member=SimpleNamespace(bot=bot))

    async def test_delivery_then_raw_reactions_records_exact_decision(self):
        await self.worker.deliver(self.decision)
        self.assertEqual(self.store.ready("777"), [])
        self.message.add_reaction.assert_any_await("👍")
        self.message.add_reaction.assert_any_await("👎")
        notification = self.store.get_notification("1234", "777")
        self.assertEqual((notification["like_count"], notification["dislike_count"]), (0, 0))
        await self.worker.on_raw_reaction_add(self.payload())
        row = self.store.feedback_rows()[0]
        self.assertEqual(row["decision_id"], self.decision["decision_id"])
        self.assertEqual(row["head_sha"], self.decision["head_sha"])
        self.assertEqual(row["vote"], 1)
        await self.worker.on_raw_reaction_add(self.payload("👎"))
        self.assertIsNone(self.store.feedback_rows()[0]["vote"])
        await self.worker.on_raw_reaction_remove(self.payload())
        self.assertEqual(self.store.feedback_rows()[0]["vote"], -1)
        notification = self.store.get_notification("1234", "777")
        self.assertEqual((notification["like_count"], notification["dislike_count"]), (0, 1))

    async def test_bot_unrelated_channel_and_unknown_message_reactions_are_ignored(self):
        await self.worker.deliver(self.decision)
        await self.worker.on_raw_reaction_add(self.payload(user_id=999))
        await self.worker.on_raw_reaction_add(self.payload(bot=True))
        payload = self.payload()
        payload.channel_id = 888
        await self.worker.on_raw_reaction_add(payload)
        payload.channel_id = 777
        payload.message_id = 8888
        await self.worker.on_raw_reaction_add(payload)
        self.assertEqual(self.store.feedback_rows(), [])
        notification = self.store.get_notification("1234", "777")
        self.assertEqual((notification["like_count"], notification["dislike_count"]), (0, 0))

    async def test_allowlisted_reviewers_only(self):
        self.worker.config.reviewer_ids = frozenset({333})
        await self.worker.deliver(self.decision)
        await self.worker.on_raw_reaction_add(self.payload(user_id=222))
        await self.worker.on_raw_reaction_add(self.payload(user_id=333))
        self.assertEqual([row["user_id"] for row in self.store.feedback_rows()], ["333"])

    async def test_network_failure_keeps_notification_unsent_for_retry(self):
        self.worker.channel.send.side_effect = OSError("Test network failure")
        await self.worker.deliver(self.decision)
        self.assertEqual(self.store.notifications("777"), [])
        self.assertEqual(self.store.ready("777"), [])  # Retry is delayed.
        self.assertTrue(self.worker._recover_needed)

    async def test_recent_post_is_recovered_after_restart_without_resending(self):
        old_message = SimpleNamespace(id=1234, author=SimpleNamespace(id=999),
                                      embeds=[build_alert(self.decision)["embed"]])
        async def history(limit):
            yield old_message
            yield SimpleNamespace(id=5678, author=old_message.author, embeds=old_message.embeds)
        self.worker.channel.history = history
        await self.worker.recover_recent_alerts([self.decision])
        self.assertEqual(self.store.ready("777"), [])
        self.worker.channel.send.assert_not_awaited()
        self.assertEqual(self.store.notifications("777")[0]["message_id"], "1234")

    async def test_cleared_reactions_withdraw_feedback(self):
        await self.worker.deliver(self.decision)
        await self.worker.on_raw_reaction_add(self.payload())
        await self.worker.on_raw_reaction_clear(self.payload())
        self.assertIsNone(self.store.feedback_rows()[0]["vote"])

    async def test_sync_recovers_reactions_missed_while_disconnected(self):
        await self.worker.deliver(self.decision)
        await self.worker.on_raw_reaction_add(self.payload())
        async def users():
            yield SimpleNamespace(id=333, bot=False)
            yield SimpleNamespace(id=999, bot=True)
        reaction = SimpleNamespace(emoji="👎", users=users)
        self.message.reactions = [reaction]
        self.worker.channel.fetch_message = AsyncMock(return_value=self.message)
        await self.worker.sync_feedback()
        rows = {row["user_id"]: row for row in self.store.feedback_rows()}
        self.assertIsNone(rows["222"]["vote"])
        self.assertEqual(rows["333"]["vote"], -1)
        self.assertNotIn("999", rows)
        notification = self.store.get_notification("1234", "777")
        self.assertEqual((notification["like_count"], notification["dislike_count"]), (0, 1))

    async def test_connection_test_sends_one_message_and_no_pr_alerts(self):
        self.worker.test_only = True
        await self.worker.on_ready()
        await self.worker.on_ready()
        self.worker.channel.send.assert_awaited_once()
        self.assertIn("connection test", self.worker.channel.send.call_args.args[0])
        self.assertEqual(self.worker.sent_test_message_id, 1234)
        self.assertEqual(self.store.notifications("777"), [])


if __name__ == "__main__":
    unittest.main()
