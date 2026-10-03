"""Runtime configuration. The bot token is never printed."""

from dataclasses import dataclass, field
import os
import math


@dataclass
class Config:
    token: str = field(repr=False)
    channel_id: int | None = None
    channel_name: str = "slop-factory"
    poll_interval: float = 3.0
    sync_interval: float = 60.0
    reviewer_ids: frozenset[int] = frozenset()
    source_mode: str = "inbox"
    source_repository: str = "team repository"

    @classmethod
    def from_env(cls):
        token = os.environ.get("DISCORD_BOT_TOKEN", "").strip()
        if not token or token in {"your-bot-token", "YOUR_BOT_TOKEN", "replace-me"}:
            raise ValueError("Set DISCORD_BOT_TOKEN in .env on the machine running the worker")
        raw_id = os.environ.get("DISCORD_CHANNEL_ID", "").strip()
        try:
            channel_id = int(raw_id) if raw_id else None
            reviewer_ids = frozenset(int(x.strip()) for x in os.environ.get("DISCORD_REVIEWER_IDS", "").split(",") if x.strip())
            poll = float(os.environ.get("POLL_INTERVAL_SECONDS", "3"))
            sync = float(os.environ.get("FEEDBACK_SYNC_SECONDS", "60"))
        except ValueError as exc:
            raise ValueError("Channel/reviewer IDs must be numeric and intervals must be numbers") from exc
        if not math.isfinite(poll) or not math.isfinite(sync) or poll < 1 or sync < 10 or (channel_id is not None and channel_id <= 0) or any(x <= 0 for x in reviewer_ids):
            raise ValueError("Use positive IDs, a poll interval >=1s and a feedback sync interval >=10s")
        name = os.environ.get("DISCORD_CHANNEL_NAME", "slop-factory").strip().removeprefix("#")
        if not channel_id and not name:
            raise ValueError("Set DISCORD_CHANNEL_ID or DISCORD_CHANNEL_NAME")
        mode = os.environ.get("DUPCHECK_SOURCE", "inbox").strip()
        if mode not in {"inbox", "team_sqlite"}:
            raise ValueError("DUPCHECK_SOURCE must be inbox or team_sqlite")
        repository = os.environ.get("DUPCHECK_REPOSITORY", "team repository").strip()
        if not repository or len(repository) > 256:
            raise ValueError("DUPCHECK_REPOSITORY must be a nonempty repository name")
        return cls(token, channel_id, name, poll, sync, reviewer_ids, mode, repository)
