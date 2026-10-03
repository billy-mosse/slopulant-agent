"""Run with python -m dupcheck_discord. Preview never connects to Discord."""

import argparse
import json
import logging
import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

from .config import repository_from_env
from .models import validate_decision
from .store import Store


def main():
    parser = argparse.ArgumentParser(description="DupCheck Discord alerts and review feedback")
    parser.add_argument("--db", help="SQLite path (default: DUPCHECK_DB_PATH or data/dupcheck.sqlite3)")
    parser.add_argument("--env-file", default=".env", help="Optional local configuration file")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init-db", help="Create the worker's tables; preserve the watcher's tables")
    commands.add_parser("run", help="Connect the bot and start delivering alerts/recording reactions")
    commands.add_parser("test-connection", help="Send one labelled connection-test message, then exit")
    commands.add_parser("feedback", help="Export recorded review feedback as JSON")
    commands.add_parser("sync-source", help="Import current decisions from the watcher's tables")
    commands.add_parser("source-preview", help="Preview the watcher database adapter without writing or sending")
    commands.add_parser("openclaw-sync", help="One pass via OpenClaw: import verdicts, post new alerts, sync 👍/👎")
    commands.add_parser("openclaw-preview", help="Print the OpenClaw alert cards that would be posted (no sending)")
    for command in ("ingest", "preview"):
        sub = commands.add_parser(command, help="Read a decision JSON file" if command == "ingest" else "Print the alert without sending it")
        sub.add_argument("file", type=Path)
    close = commands.add_parser("close-pr", help="Stop pending alerts for a closed/merged PR")
    close.add_argument("repository")
    close.add_argument("pr_number", type=int)
    close.add_argument("--state", choices=("closed", "merged"), default="closed")
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    try:
        if args.command == "close-pr" and os.environ.get("DUPCHECK_SOURCE") == "watcher":
            raise ValueError("close-pr is for inbox mode. Watcher mode follows the PR state the watcher reads from GitHub.")
        if args.command in {"ingest", "preview"}:
            decision = validate_decision(json.loads(args.file.read_text(encoding="utf-8")))
            if args.command == "preview":
                if not decision["is_duplicate"] or decision["status"] != "completed" or decision["pr_state"] != "open":
                    print(json.dumps({"will_send": False, "reason": "Only completed duplicates on open PRs are sent."}))
                else:
                    from .messages import preview_alert
                    print(json.dumps(preview_alert(decision), indent=2, ensure_ascii=False))
                return
        path = Path(args.db or os.environ.get("DUPCHECK_DB_PATH", "data/dupcheck.sqlite3")).expanduser()
        if args.command == "source-preview":
            from .source import build_decisions
            with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
                print(json.dumps(build_decisions(connection, repository_from_env()), indent=2, ensure_ascii=False))
            return
        if args.command in {"run", "sync-source"} and os.environ.get("DUPCHECK_SOURCE") == "watcher" and not path.is_file():
            raise ValueError(f"Watcher database does not exist: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        store = Store(path)
        if args.command == "init-db":
            print(f"Worker tables ready in {path.resolve()}")
        elif args.command == "ingest":
            store.add_decision(decision)
            print(f"Saved {decision['decision_id']}; duplicate={decision['is_duplicate']}")
        elif args.command == "feedback":
            print(json.dumps(store.feedback_rows(), indent=2, ensure_ascii=False))
        elif args.command == "sync-source":
            from .source import sync_source
            print(f"Imported {sync_source(store, repository_from_env())} new decision snapshots")
        elif args.command in {"openclaw-sync", "openclaw-preview"}:
            from .openclaw_bridge import presentation, run_once
            channel = os.environ.get("DISCORD_CHANNEL_ID", "").strip()
            if not channel.isdigit():
                raise ValueError("Set DISCORD_CHANNEL_ID (numeric) for OpenClaw delivery")
            if args.command == "openclaw-preview":
                from .source import build_decisions
                with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as connection:
                    cards = [presentation(d) for d in build_decisions(connection, repository_from_env())
                             if d["is_duplicate"] and d["status"] == "completed"]
                print(json.dumps(cards, indent=2, ensure_ascii=False))
            else:
                logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
                reviewers = frozenset(x.strip() for x in os.environ.get("DISCORD_REVIEWER_IDS", "").split(",") if x.strip())
                print(json.dumps(run_once(store, repository_from_env(), channel, reviewers)))
        elif args.command == "close-pr":
            store.close_pr(args.repository, args.pr_number, args.state)
            print(f"Marked {args.repository}#{args.pr_number} {args.state}")
        elif args.command in {"run", "test-connection"}:
            from .bot import DiscordWorker
            from .config import Config
            config = Config.from_env()
            logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
            worker = DiscordWorker(store, config, test_only=args.command == "test-connection")
            worker.run(config.token, log_handler=None)
            if args.command == "test-connection" and worker.sent_test_message_id is None:
                parser.exit(2, "Connection test did not send a message. Check the bot's channel access.\n")
    except (ValueError, OSError, sqlite3.Error) as exc:
        parser.exit(2, f"Error: {exc}\n")


if __name__ == "__main__":
    main()
