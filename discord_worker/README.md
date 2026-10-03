# DupCheck Discord worker

This worker connects the team's OpenClaw/Qwen duplicate detector to Discord. It reads completed decisions from SQLite, posts duplicate alerts in `#slop-factory`, and records reviewers' reactions against the exact analyzed snapshot.

OpenClaw/Qwen makes the duplication decision. This worker handles delivery and feedback. It requires internet access to reach Discord.

## Start from this repository

From the repository root, enter this component before running its commands:

```sh
cd discord_worker
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
mkdir -p data
cp snapshots/my_database.db data/my_database.db
```

Paste your own bot token into the local `.env`, configure the intended Discord channel, and set:

```dotenv
DUPCHECK_SOURCE=team_sqlite
DUPCHECK_DB_PATH=data/my_database.db
```

Then preview the source decisions and start the worker:

```sh
python -m dupcheck_discord source-preview
python -m dupcheck_discord run
```

`snapshots/my_database.db` is a consistent backup of the GB10's live shared database. At capture it contains five source verdicts, three Discord delivery records, and three human-review rows with per-message like/dislike totals. It also retains the source topics, PR metadata, and immutable decision snapshots. The copy under `data/` is ignored by Git and gives the worker a writable database without changing the committed snapshot.

The capture timestamp, SHA-256, integrity checks, and table row counts are recorded in [`snapshots/manifest.json`](snapshots/manifest.json). [`snapshots/schema.sql`](snapshots/schema.sql) provides the snapshot's schema for review without a SQLite browser.

The snapshot already records delivered posts and their original Discord message/channel IDs. With the original channel configured, those delivered snapshots are not automatically sent again. Use the fictional inbox examples below for a fresh notification test. Keep the GB10's current `/home/dell/my_database.db` as the live database; do not overwrite it with this historical snapshot. No bot token or `.env` is included in the snapshot or this repository.

This repository's watcher uses a different `data/watcher.db` schema, defined in `../watcher/db.py`. The current adapter targets `Commited`, `new_prs`, `dup_cg`, and `dupe_decision`; it does not automatically read the watcher's `decisions` or `alerts` tables. A detector with richer metadata can integrate through the JSON inbox API documented below. The worker has its own dependencies and runs separately from the watcher.

## Set up

Use Python 3.10 or newer. From `discord_worker/` in this repository, or `/home/dell/dupcheck-discord` for the existing GB10 deployment, these commands work on the GB10 or your laptop:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `DISCORD_BOT_TOKEN`. The default channel name is `slop-factory`. If the bot can see more than one text channel with that name, set `DISCORD_CHANNEL_ID` to the intended channel's numeric ID. A channel ID takes priority over a name.

The example configuration uses `data/dupcheck.sqlite3` with `DUPCHECK_SOURCE=inbox`. The GB10's shared database uses the configuration below. `--db` overrides `DUPCHECK_DB_PATH`; `--env-file` selects a configuration file instead of `.env`.

### Create the Discord bot

1. Open the [Discord Developer Portal](https://discord.com/developers/applications), create an application, and open its **Bot** page.
2. Generate/reset the bot token and paste it into your local `.env`. Keep that file out of Git; the included `.gitignore` excludes it.
3. Under **OAuth2 → URL Generator**, select the `bot` scope and these permissions: **View Channels**, **Send Messages**, **Embed Links**, **Add Reactions**, and **Read Message History**. Open the generated URL and invite the bot to your server.
4. Ensure those permissions also apply in `#slop-factory`. You do not need Administrator, Message Content Intent, or Server Members Intent.
5. For a numeric channel ID, enable Discord's **User Settings → Advanced → Developer Mode**, then right-click the channel and choose **Copy Channel ID**.

The bot listens for guild message reactions. It does not read conversation text.

## Try it

These fictional examples use a fresh setup with the inbox settings from `.env.example`. For the configured GB10, follow **Shared database on the GB10** instead.

First verify the bot can post one labelled test message:

```sh
python -m dupcheck_discord test-connection
```

This exits after the connection test and does not deliver PR decisions.

```sh
python -m dupcheck_discord init-db
python -m dupcheck_discord preview examples/duplicate.json
python -m dupcheck_discord ingest examples/duplicate.json
python -m dupcheck_discord run
```

`preview` renders an alert without contacting Discord. The examples contain fictional PRs. `run` connects the bot and sends eligible alerts; leave it running to collect reactions. Stop it with Ctrl+C.

In another terminal with the virtual environment activated:

```sh
python -m dupcheck_discord ingest examples/clean.json
python -m dupcheck_discord feedback
```

A clean decision creates no Discord notification. `feedback` prints a JSON list of saved reviews and the related decision metadata.

In inbox mode, stop new alerts for a closed or merged PR with:

```sh
python -m dupcheck_discord close-pr OWNER/REPOSITORY 42 --state closed
```

Use `--state merged` for a merged PR. This command is only available in inbox mode, where the producer must keep PR state current. `team_sqlite` rejects `close-pr` and follows the detector's active source rows instead.

## Shared database on the GB10

The deployed worker directory is `/home/dell/dupcheck-discord`. Its `.env` selects the team's database:

```dotenv
DISCORD_CHANNEL_ID=1555978700686889111
DUPCHECK_DB_PATH=/home/dell/my_database.db
DUPCHECK_SOURCE=team_sqlite
DUPCHECK_REPOSITORY=team repository
```

Replace `team repository` with the real `OWNER/REPOSITORY` when known. The adapter reads `Commited`, `new_prs`, `dup_cg`, and `dupe_decision`. It uses `new_prs.owner` for the PR author and `dupe_decision.boolean` for the verdict, joins existing topics through `dup_cg.t_id`, and combines positive comparisons into one alert per PR snapshot. Missing verdicts leave the snapshot pending; all-zero verdicts produce no alert.

`run` polls these source tables automatically. The original tables remain unchanged; the worker creates its namespaced delivery/feedback tables and a `discord_flagger_feedback` view in the same SQLite file.

```sh
cd /home/dell/dupcheck-discord
source .venv/bin/activate
python -m dupcheck_discord source-preview
python -m dupcheck_discord sync-source
```

`source-preview` only reads the database and prints adapted decisions. `sync-source` imports snapshots into the worker's tables without sending to Discord. The source schema has no Git SHA, PR URL, GitHub open/closed state, detailed code comparison, or whole-project/partial classification. Alerts show the available stored topics and descriptions, label the scope as unspecified, and identify a **database revision**, not a Git commit. No missing evidence or links are invented.

The adapter treats `new_prs` and its comparison rows as the active PR set. Removing a PR from that set suppresses future delivery. This is a local eligibility rule; it does not establish the PR's GitHub state.

The GB10 uses a user service for continuous operation:

[`deploy/dupcheck-discord.service`](deploy/dupcheck-discord.service) contains the deployed unit. For a fresh deployment, adjust `WorkingDirectory` and `ExecStart` to the worker directory and virtual environment, then install the unit:

```sh
mkdir -p ~/.config/systemd/user
cp deploy/dupcheck-discord.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now dupcheck-discord.service
```

```sh
systemctl --user status dupcheck-discord.service
systemctl --user restart dupcheck-discord.service
systemctl --user stop dupcheck-discord.service
journalctl --user -u dupcheck-discord.service -n 30 --no-pager
```

Do not start a second `run` process while this service is active. The service restarts on failure and depends on the user's systemd session; lingering is not enabled. The pre-integration backup is `/home/dell/dupcheck-discord/backups/my_database-before-discord-20261003T174654Z.db`.

## Optional inbox integration

For a detector that can emit richer JSON directly, keep `DUPCHECK_SOURCE=inbox`. See `examples/duplicate.json` for the full contract. Ingest the JSON through the CLI or call the Python API when an analysis finishes:

```python
from dupcheck_discord.models import validate_decision
from dupcheck_discord.store import Store

# payload follows examples/duplicate.json.
Store("data/dupcheck.sqlite3").add_decision(validate_decision(payload))
```

The input contract is:

| Field | Meaning |
| --- | --- |
| `decision_id` | Unique ID for this analysis; reuse only for an identical replay. |
| `repository`, `pr_number`, `pr_title` | Identify the PR; optional `pr_url` links it. |
| `head_sha` or `source_revision` | Analyzed Git commit hash, or a 64-character SHA-256 fingerprint of the database snapshot. |
| `author_login`, optional `author_name` | Actual PR author; optional display name. |
| `topic` | What the PR implements. |
| `is_duplicate` | JSON boolean `true` or `false`. |
| `status` | `pending`, `completed`, or `failed`; defaults to `completed`. |
| `pr_state` | `open`, `closed`, or `merged`; defaults to `open`. |
| `duplicate_kind` | `partial`, `whole_project`, or `unspecified`; defaults to `partial`. |
| `reason`, `matches` | Completed duplicate decisions require an explanation and at least one match. Each match has `project_name` and can include `topic`, `url`, `pr_files`, `existing_files`, and `evidence`. |
| Optional `model_version`, `discord_user_id` | Model provenance and an explicit Discord author mention. |

Give every analysis a new `decision_id`. Decisions are immutable: replaying the same ID and payload is a no-op; changing its payload raises an error. A newer ingested decision supersedes the older snapshot for that PR, so the producer must ingest analyses in chronological order and discard obsolete results. Supply the actual PR author and available evidence; a database fingerprint must not be presented as a Git SHA.

The worker sends completed, current, duplicate decisions eligible for delivery. Each alert shows the PR author, topic, explanation, matches, and any supplied file paths or evidence links. The database records the resulting Discord message ID.

## Feedback

Each alert explains the two reactions:

- **👍** — the duplication flag is correct.
- **👎** — the flag is a false positive.

Votes are stored per decision and reviewer, with the Discord message ID linking the response to the original alert. The bot ignores its own reactions and other bots. Removing a reaction clears that vote; reacting with both thumbs leaves the review unset until the reviewer chooses one. Reaction clearing is handled too.

By default, all human reviewers who can see the channel may vote. Set `DISCORD_REVIEWER_IDS` to comma-separated Discord user IDs to restrict accepted votes.

Each Discord post also has persisted `like_count` and `dislike_count` columns in `discord_notifications`. In the shared database `/home/dell/my_database.db`, read the totals with:

```sql
SELECT n.message_id, d.pr_number, n.like_count, n.dislike_count
FROM discord_notifications AS n
JOIN discord_decisions AS d ON d.decision_id = n.decision_id
WHERE n.status = 'sent';
```

Counts update when reactions are added, removed, or cleared and include accepted human reviewers only; the bot's initial 👍/👎 reactions do not count. A reviewer who uses both emojis contributes one to each total, while their per-user vote remains unset until they choose one.

Saving feedback does not automatically train Qwen. The detector must consume these reviewed examples in future prompts/retrieval, or the team must curate them for a later training step.

In the shared database, the detector can read:

```sql
SELECT source_dup_ids, pr_number, user_id, is_good, updated_at
FROM discord_flagger_feedback
WHERE is_good IS NOT NULL;
```

The underlying `discord_feedback` table stores `vote` as `1`, `-1`, or `NULL`; the view maps that to `is_good` as `1`, `0`, or `NULL`. It also exposes `source_dup_ids` (a JSON array of the original positive comparison IDs), `source_revision`, `decision_id`, Discord message/channel IDs, and `payload_json`. Each review concerns the **whole alert**, not separate labels for every listed comparison. Feedback is stored alongside the source tables and does not rewrite the detector's verdicts.

## Delivery behavior

Run one worker for a database. Successful deliveries are recorded and ordinary polling does not send them again. Reanalysis of the same PR commit or database revision does not create another alert in the same channel. Failed sends are retried. After a restart or uncertain send, the worker looks for matching decision markers in the newest 100 channel messages before retrying. Delivery is not exactly once: a crash or timeout after Discord accepts a message but before the database records it can still produce a duplicate outside that recovery window.

While running, the bot processes reaction add/remove/clear events. Every `FEEDBACK_SYNC_SECONDS` seconds, it also reconciles the newest 100 delivered messages with Discord's current reactions, recovering recent votes changed while it was offline. Older messages are outside this periodic recovery window.

## Verify locally

```sh
python -m unittest discover -s tests -v
python -m dupcheck_discord preview examples/duplicate.json
```

These checks do not send Discord messages. A live test needs your bot token and channel configuration, followed by an ingested duplicate decision and a 👍/👎 reaction in Discord.

## Copy to the GB10

If you built this folder on your laptop, create a destination on the machine and copy the source files:

```sh
ssh dell@172.20.65.127 'mkdir -p ~/dupcheck-discord'
scp -r dupcheck_discord examples requirements.txt README.md .env.example dell@172.20.65.127:~/dupcheck-discord/
```

Then SSH into the GB10, enter `~/dupcheck-discord`, and follow **Set up**. The copy command excludes your token and local database. Run the worker on the machine that has access to the detector's SQLite file.
