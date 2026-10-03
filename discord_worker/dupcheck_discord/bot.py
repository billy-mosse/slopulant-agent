"""Small Discord bot: deliver decisions and persist explicit review votes."""

import asyncio
import logging
import sqlite3

import aiohttp
import discord
from discord.ext import tasks

from .messages import build_alert, marker

LOG = logging.getLogger(__name__)
EMOJIS = ("👍", "👎")


def select_named_channel(guilds, name):
    matches = [channel for guild in guilds for channel in guild.text_channels
               if channel.name == name and channel.permissions_for(guild.me).view_channel]
    if len(matches) != 1:
        raise ValueError(f"Found {len(matches)} accessible text channels named #{name}. Set DISCORD_CHANNEL_ID to the numeric target channel ID.")
    return matches[0]


class DiscordWorker(discord.Client):
    def __init__(self, store, config, *, test_only=False):
        intents = discord.Intents.none()
        intents.guilds = True
        intents.guild_reactions = True
        super().__init__(intents=intents, allowed_mentions=discord.AllowedMentions.none())
        self.store = store
        self.config = config
        self.channel = None
        self.test_only = test_only
        self._test_started = False
        self.sent_test_message_id = None
        self._channel_lock = asyncio.Lock()
        self._feedback_lock = asyncio.Lock()
        self._recover_needed = True
        self.poll.change_interval(seconds=config.poll_interval)
        self.sync_feedback.change_interval(seconds=config.sync_interval)

    async def setup_hook(self):
        if not self.test_only:
            self.poll.start()
            self.sync_feedback.start()

    async def close(self):
        self.poll.cancel()
        self.sync_feedback.cancel()
        await super().close()

    async def resolve_channel(self):
        async with self._channel_lock:
            if self.channel is not None:
                return self.channel
            channel = await self.fetch_channel(self.config.channel_id) if self.config.channel_id else select_named_channel(self.guilds, self.config.channel_name)
            if not isinstance(channel, discord.TextChannel):
                raise ValueError("The target must be a server text channel")
            permissions = channel.permissions_for(channel.guild.me)
            required = ("view_channel", "send_messages", "embed_links", "add_reactions", "read_message_history")
            missing = [permission for permission in required if not getattr(permissions, permission)]
            if missing:
                raise ValueError("The bot needs these channel permissions: " + ", ".join(missing))
            self.channel = channel
            LOG.info("Target channel: #%s (%s)", channel.name, channel.id)
            return channel

    async def on_ready(self):
        LOG.info("Connected as %s", self.user)
        self._recover_needed = True
        try:
            await self.resolve_channel()
        except (ValueError, discord.HTTPException) as exc:
            LOG.error("Channel setup failed: %s", exc)
            await self.close()
            return
        if self.test_only:
            if self._test_started:
                return
            self._test_started = True
            try:
                message = await self.channel.send(
                    "✅ DupCheck connection test — the bot can post in this channel.\n"
                    "PR duplicate alerts and reviewer feedback will use this connection.",
                    allowed_mentions=discord.AllowedMentions.none(),
                )
                self.sent_test_message_id = message.id
                LOG.info("Connection test sent: channel=%s message=%s", self.channel.id, message.id)
            except (discord.HTTPException, aiohttp.ClientError, OSError) as exc:
                LOG.error("Connection test failed: %s", exc)
            finally:
                await self.close()

    async def recover_recent_alerts(self, decisions):
        """Recover a recent post if Discord accepted it before the worker stopped."""
        wanted = {marker(decision): decision for decision in decisions}
        async for message in self.channel.history(limit=100):
            if message.author.id != self.user.id:
                continue
            for embed in message.embeds:
                decision = wanted.get(embed.footer.text)
                if decision:
                    await asyncio.to_thread(self.store.mark_sent, decision["decision_id"], str(self.channel.id), str(message.id))
                    LOG.info("Recovered Discord message %s for %s", message.id, decision["decision_id"])
                    wanted.pop(embed.footer.text)
        self._recover_needed = False

    async def deliver(self, decision):
        channel_id = str(self.channel.id)
        await asyncio.to_thread(self.store.mark_attempt, decision["decision_id"], channel_id)
        try:
            message = await self.channel.send(**build_alert(decision))
            await asyncio.to_thread(self.store.mark_sent, decision["decision_id"], channel_id, str(message.id))
        except (discord.HTTPException, aiohttp.ClientError, OSError, asyncio.TimeoutError, sqlite3.Error) as exc:
            self._recover_needed = True
            await asyncio.to_thread(self.store.mark_failed, decision["decision_id"], channel_id, str(exc), 10)
            LOG.warning("Delivery failed for %s; retrying: %s", decision["decision_id"], exc)
            return
        LOG.info("Sent decision %s as message %s", decision["decision_id"], message.id)
        for emoji in EMOJIS:
            try:
                await message.add_reaction(emoji)
            except discord.HTTPException as exc:
                LOG.warning("Alert sent but reaction seed failed: %s", exc)

    @tasks.loop(seconds=3)
    async def poll(self):
        try:
            await self.resolve_channel()
            if self.config.source_mode == "team_sqlite":
                from .source import sync_source
                imported = await asyncio.to_thread(sync_source, self.store, self.config.source_repository)
                if imported:
                    LOG.info("Imported %s new database decision snapshots", imported)
            decisions = await asyncio.to_thread(self.store.ready, str(self.channel.id))
            if decisions and self._recover_needed:
                await self.recover_recent_alerts(decisions)
                decisions = await asyncio.to_thread(self.store.ready, str(self.channel.id))
            for decision in decisions:
                await self.deliver(decision)
        except (discord.HTTPException, aiohttp.ClientError, sqlite3.Error, OSError, ValueError) as exc:
            LOG.warning("Polling failed; will retry: %s", exc)

    @poll.before_loop
    async def before_poll(self):
        await self.wait_until_ready()

    def accepts_user(self, user_id):
        return user_id != self.user.id and (not self.config.reviewer_ids or user_id in self.config.reviewer_ids)

    async def save_reaction(self, payload, added):
        if self.channel is None or payload.channel_id != self.channel.id or not self.accepts_user(payload.user_id):
            return
        member = getattr(payload, "member", None)
        if member is not None and member.bot:
            return
        emoji = str(payload.emoji)
        if emoji not in EMOJIS:
            return
        async with self._feedback_lock:
            saved = await asyncio.to_thread(self.store.record_reaction, str(payload.message_id), str(payload.channel_id), str(payload.user_id), emoji, added)
        if saved:
            LOG.info("Recorded feedback: message=%s reviewer=%s emoji=%s added=%s", payload.message_id, payload.user_id, emoji, added)

    async def on_raw_reaction_add(self, payload):
        await self.save_reaction(payload, True)

    async def on_raw_reaction_remove(self, payload):
        await self.save_reaction(payload, False)

    async def on_raw_reaction_clear(self, payload):
        if self.channel is not None and payload.channel_id == self.channel.id:
            async with self._feedback_lock:
                await asyncio.to_thread(self.store.clear_reactions, str(payload.message_id), str(payload.channel_id))

    async def on_raw_reaction_clear_emoji(self, payload):
        if self.channel is not None and payload.channel_id == self.channel.id and str(payload.emoji) in EMOJIS:
            async with self._feedback_lock:
                await asyncio.to_thread(self.store.clear_reactions, str(payload.message_id), str(payload.channel_id), str(payload.emoji))

    @tasks.loop(seconds=60)
    async def sync_feedback(self):
        """Reconcile recent alerts so reconnects/restarts don't lose votes."""
        try:
            await self.resolve_channel()
            notifications = await asyncio.to_thread(self.store.notifications, str(self.channel.id), 100)
            for row in notifications:
                try:
                    async with self._feedback_lock:
                        message = await self.channel.fetch_message(int(row["message_id"]))
                        votes = []
                        for reaction in message.reactions:
                            emoji = str(reaction.emoji)
                            if emoji in EMOJIS:
                                async for user in reaction.users():
                                    if not user.bot and self.accepts_user(user.id):
                                        votes.append((str(user.id), emoji))
                        # Only clear after the full snapshot was fetched successfully.
                        await asyncio.to_thread(self.store.replace_reactions, row["message_id"], str(self.channel.id), votes)
                except discord.NotFound:
                    await asyncio.to_thread(self.store.clear_reactions, row["message_id"], str(self.channel.id))
                except discord.HTTPException as exc:
                    LOG.warning("Could not refresh feedback for message %s: %s", row["message_id"], exc)
        except (sqlite3.Error, OSError, discord.HTTPException, aiohttp.ClientError) as exc:
            LOG.warning("Feedback sync failed; will retry: %s", exc)

    @sync_feedback.before_loop
    async def before_sync(self):
        await self.wait_until_ready()
