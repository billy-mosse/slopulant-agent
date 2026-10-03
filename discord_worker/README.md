# DupCheck Discord worker

This worker connects the duplicate pipeline to Discord. It reads finished verdicts from the pipeline's shared SQLite database (`../data/watcher.db`), posts duplicate alerts in `#slop-factory`, and records reviewers' reactions against the exact analyzed commit.

The watcher (Qwen3-Coder-Next topics + candidates) and the CLM worker (CLM-v0.1-8B verdicts + the OpenClaw agent's draft) make the duplication decision. This package turns finished verdicts into alerts and records feedback; in production OpenClaw delivers them (see **On the GB10**).

## Start from this repository

```sh
cd discord_worker
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env     # paste DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID
.venv/bin/python -m dupcheck_discord source-preview   # what would be sent, read-only
.venv/bin/python -m dupcheck_discord run
```

`.env.example` points at the watcher's database (`DUPCHECK_SOURCE=watcher`, `DUPCHECK_DB_PATH=../data/watcher.db`). The worker adds its own `discord_*` tables and the `discord_flagger_feedback` view to that file and never writes the watcher's tables.

## Set up

Use Python 3.10 or newer. From `discord_worker/` in this repository, these commands work on the GB10 or your laptop:

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `DISCORD_BOT_TOKEN`. The default channel name is `slop-factory`. If the bot can see more than one text channel with that name, set `DISCORD_CHANNEL_ID` to the intended channel's numeric ID. A channel ID takes priority over a name.

The example configuration reads the watcher's database (`DUPCHECK_SOURCE=watcher`, `DUPCHECK_DB_PATH=../data/watcher.db`). For the fictional JSON examples below, use `DUPCHECK_SOURCE=inbox` with `DUPCHECK_DB_PATH=data/dupcheck.sqlite3`. `--db` overrides `DUPCHECK_DB_PATH`; `--env-file` selects a configuration file instead of `.env`.

### Create the Discord bot

1. Open the [Discord Developer Portal](https://discord.com/developers/applications), create an application, and open its **Bot** page.
2. Generate/reset the bot token and paste it into your local `.env`. Keep that file out of Git; the included `.gitignore` excludes it.
3. Under **OAuth2 → URL Generator**, select the `bot` scope and these permissions: **View Channels**, **Send Messages**, **Embed Links**, **Add Reactions**, and **Read Message History**. Open the generated URL and invite the bot to your server.
4. Ensure those permissions also apply in `#slop-factory`. You do not need Administrator, Message Content Intent, or Server Members Intent.
5. For a numeric channel ID, enable Discord's **User Settings → Advanced → Developer Mode**, then right-click the channel and choose **Copy Channel ID**.

The bot listens for guild message reactions. It does not read conversation text.

## Try it

These fictional examples use a fresh setup with `DUPCHECK_SOURCE=inbox` and `DUPCHECK_DB_PATH=data/dupcheck.sqlite3`. For the pipeline, follow **The watcher source** instead.

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

Use `--state merged` for a merged PR. This command is only available in inbox mode, where the producer must keep PR state current. Watcher mode rejects `close-pr` and follows the PR state the watcher reads from GitHub.

## The watcher source (`DUPCHECK_SOURCE=watcher`)

For each PR the watcher has analysed (`prs` + its latest run in `latest_scores`), the adapter builds one snapshot of that commit:

- **pending** until every candidate has a verdict from the current CLM version (`decisions.classifier`) and the CLM worker has marked the commit finished (`kv` `clm_finished:<head>:<base>`), and, for a duplicate, until the OpenClaw note is in `alerts`;
- **completed duplicate** when any candidate has `decisions.is_duplicate = 1`: the note is the alert text, and each match carries a link to the folder on `main`, its owner, the CLM score and threshold, the similarity scores, shared keywords/tables and the matched PR topic;
- **completed clean** otherwise (also for a PR with no candidates): never sent.

The real head commit, PR URL, author and GitHub state come from the watcher. A closed or merged PR stops being eligible; reopening restores it. A newer snapshot (new push, new verdict, new note) supersedes the older one, and a commit that was already posted is not posted again.

```sh
.venv/bin/python -m dupcheck_discord source-preview   # read-only
.venv/bin/python -m dupcheck_discord sync-source      # import snapshots, don't send
```

### On the GB10: delivery through OpenClaw

OpenClaw owns the Discord connection; this package supplies the data side. The OpenClaw job
`slopulant-discord` runs one pass every 30 s:

```sh
.venv/bin/python -m dupcheck_discord openclaw-sync      # import, post new alerts, sync 👍/👎
.venv/bin/python -m dupcheck_discord openclaw-preview   # print the alert cards, send nothing
```

[`openclaw_bridge.py`](dupcheck_discord/openclaw_bridge.py) posts each eligible alert as a
card with `openclaw message send --presentation` (records the message id with the same
delivery bookkeeping as the bot), seeds 👍/👎, and reconciles reactions on the 10 most recent
alerts with `openclaw message reactions` (per reviewer; the bot's own reactions don't count).
Thread questions (@slop-factory) are answered by OpenClaw's `watcher` agent; see the
repo README. Setup: `../deploy/openclaw_setup.sh`.

The standalone bot (`python -m dupcheck_discord run`, discord.py) still works for the inbox
examples, but must not run next to OpenClaw with the same token.

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

Each Discord post also has persisted `like_count` and `dislike_count` columns in `discord_notifications`. In the shared database (`data/watcher.db`), read the totals with:

```sql
SELECT n.message_id, d.pr_number, n.like_count, n.dislike_count
FROM discord_notifications AS n
JOIN discord_decisions AS d ON d.decision_id = n.decision_id
WHERE n.status = 'sent';
```

Counts update when reactions are added, removed, or cleared and include accepted human reviewers only; the bot's initial 👍/👎 reactions do not count. A reviewer who uses both emojis contributes one to each total, while their per-user vote remains unset until they choose one.

Saving feedback does not automatically retune anything. `python -m eval.export_feedback` (repo root) joins the reviews back to the judged pairs, with the CLM score and threshold, so the threshold can be re-picked in `clm_dupe/calibration/`.

In the shared database, the reviews link back to the watcher's verdicts:

```sql
SELECT pr_number, head_sha, base_sha, source_decision_keys, user_id, is_good, updated_at
FROM discord_flagger_feedback
WHERE is_good IS NOT NULL;
```

The underlying `discord_feedback` table stores `vote` as `1`, `-1`, or `NULL`; the view maps that to `is_good` as `1`, `0`, or `NULL`. `source_decision_keys` is a JSON array of `<pr folder>/<existing folder>` pairs: with `pr_number`, `head_sha` and `base_sha` they are the primary keys of the reviewed `decisions` rows. The view also exposes `source_revision`, `decision_id`, Discord message/channel IDs, and `payload_json`. Each review concerns the **whole alert**, not separate labels for every listed comparison. Feedback is stored alongside the source tables and does not rewrite the detector's verdicts.

## Delivery behavior

Run one worker for a database. Successful deliveries are recorded and ordinary polling does not send them again. Reanalysis of the same PR commit or database revision does not create another alert in the same channel. Failed sends are retried. After a restart or uncertain send, the worker looks for matching decision markers in the newest 100 channel messages before retrying. Delivery is not exactly once: a crash or timeout after Discord accepts a message but before the database records it can still produce a duplicate outside that recovery window.

While running, the bot processes reaction add/remove/clear events. Every `FEEDBACK_SYNC_SECONDS` seconds, it also reconciles the newest 100 delivered messages with Discord's current reactions, recovering recent votes changed while it was offline. Older messages are outside this periodic recovery window.

## Verify locally

```sh
python -m unittest discover -s tests -v
python -m dupcheck_discord preview examples/duplicate.json
```

These checks do not send Discord messages. A live test needs your bot token and channel configuration, followed by an ingested duplicate decision and a 👍/👎 reaction in Discord.
