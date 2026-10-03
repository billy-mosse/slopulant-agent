# slopulant-agent — duplicate-effort agent

This repo is **the tool**: an always-on agent that watches PRs and flags work that
duplicates (or could reuse) an existing system. It runs fully on local models.

The code it analyses lives in a **separate repo**:
[`billy-mosse/slopulant-monorepo`](https://github.com/billy-mosse/slopulant-monorepo) —
the mock "Slopulent Living" company ML monorepo. Each top-level folder there is one
system ("repo"). Nothing in this repo should be committed there, and vice versa.

## How it works

Architecture diagram: [`docs/architecture.html`](docs/architecture.html). Pitch deck:
[`docs/deck/`](docs/deck/).

Everything shares **one SQLite database**, `data/watcher.db`. OpenClaw runs the schedule,
the agent and the Discord connection:

```
GitHub PR ─► watcher tick (OpenClaw job, every 30 s)      Qwen3-Coder-Next: topics · MiniLM: candidates
          ─► CLM worker (watcher/clm_worker.py)            CLM-v0.1-8B: is it a duplicate?
                                                           + OpenClaw agent `watcher` (Qwen) drafts the alert
          ─► Discord sync (OpenClaw job, every 30 s)       posts the alert card to #slop-factory via OpenClaw,
                                                           syncs 👍/👎 reviews
          ─► @slop-factory in a thread                     OpenClaw routes it to `watcher`, which answers
                                                           with read-only tools (watcher/mcp_server.py)
          ─► daily re-index (OpenClaw job, 06:00 NY)       regenerates topics for every folder on main
          ─► dashboards (watcher/web.py)                   live pipeline + team view, read-only
```

| Stage | Writes | Process |
|---|---|---|
| topics, PRs, candidates | `folders`, `topics`, `prs`, `pr_queue`, `scores` (`candidate = 1`) | watcher tick |
| verdicts (`is_duplicate`), the note | `decisions` (+ CLM score, threshold, version), `alerts`, `history`, `kv` `clm_finished:<head>:<base>` | CLM worker |
| delivery, reviews | `discord_decisions`, `discord_notifications`, `discord_feedback`, view `discord_flagger_feedback` | Discord sync (`dupcheck_discord openclaw-sync`) |

One **tick** (`python -m watcher.main --once`), run every 30 s by an OpenClaw scheduled
job. Ticks never overlap: each takes an exclusive lock, and queue items are claimed, so
a crashed run's items are retried.

1. **Poll** open and recently closed PRs (GitHub API with ETags). Opened, merged and
   closed PRs are recorded as events. Each PR is queued per (head commit, base commit,
   version), so new pushes and merges to `main` trigger a re-score.
2. **Index `main`** when it moves. Each top-level folder becomes **1..N topics**, one per
   model, pipeline or tool (`watcher/topics.py`). Each topic has:
   - a description, embedded with all-MiniLM-L6-v2
   - keywords from a separate LLM pass, with generic terms removed in code
   - the tables it reads and writes

   Folders over 40k characters are chunked, and the per-chunk topics merged. Topics are
   cached by git tree sha + model, so a merged PR's folders reuse the topics computed
   for the PR. A PR's own folders are always re-extracted (`REEXTRACT_PR_TOPICS`).
3. **Candidates.** Each PR topic is compared with every topic on `main`; the best topic
   pair gives the score, `mean(description cosine, keyword TF-IDF)`. Candidates are
   every shared-table link (upstream/downstream), plus the top `TOP_K` (5) folders
   scoring ≥ `MIN_CANDIDATE_SCORE` (0.38).
4. **CLM classifier** (`watcher/clm_worker.py`, its own process) judges each candidate
   one pair at a time: shared-table links are upstream/downstream from the tables;
   every other pair asks CLM-v0.1-8B "do these two systems solve the same problem?" and
   is a duplicate at p ≥ 0.19. The rule and its calibration live in
   [`clm_dupe/`](clm_dupe/README.md). Verdicts written by any other classifier (or an
   older CLM version) are re-judged.
5. **Draft.** When a PR commit has a duplicate, the OpenClaw agent `watcher` (Qwen) gets
   both systems' descriptions, tables, owners and scores and writes two short parts: *what
   overlaps* and a *suggested next step* (`watcher/alerts.py` `ALERT_BRIEF`; template
   fallback without OpenClaw).
6. **Discord alert.** The OpenClaw job `slopulant-discord`
   ([`openclaw_bridge.py`](discord_worker/dupcheck_discord/openclaw_bridge.py)) posts a card
   through OpenClaw's Discord channel: the drafted parts plus the facts from data (PR link,
   author, matched system with link, owner, CLM and similarity scores, shared keywords and
   tables), and seeds 👍/👎. It also reads the reactions back as per-reviewer votes.
   **Ask about it:** @mention slop-factory in the alert's thread. OpenClaw routes the
   message to the same `watcher` agent, which looks the alert up and answers from data. Its
   only tools are the read-only `slopulant` MCP tools (`watcher/mcp_server.py`): alert, PR
   analysis, system topics, list/read files. No shell, file-write, web or config tools.
7. **Feedback.** `python -m eval.export_feedback` turns the reviews into labelled pairs
   (label, CLM score, threshold, similarity) for re-picking the threshold.

`CLASSIFIER=1` brings back the old in-tick dummy LLM classifier (and, with
`POST_GITHUB_COMMENTS=1`, the `[oc]` PR comment) for a laptop without the CLM stack.

Every stage records start/end events for the live view. Each analysis is also written
to a `history` table for the team view. State lives in SQLite at `data/watcher.db`
(browsable in the dashboard's **Data** tab). Team ownership comes from `teams.yaml` in
the company monorepo.

## Dashboards

`python -m watcher.web` serves both on port 8765 (`--watcher` also runs the loop in
the same process, for laptop use).

- **`/` Team view:** Overview (KPIs, recent analyses, most re-built systems, which team
  re-builds which), Teams, Pull requests (Merge/Close), Systems (topics per folder),
  History (git graph), Activity, Data (every table; save the demo baseline), Try it
  (score pasted code without a PR), and a model switch between local Ollama and
  OpenRouter.
- **`/demo` Live pipeline:** an architecture diagram whose boxes light up as each
  stage runs, plus a timeline and the result. Open a test PR, merge it, or **Reset
  demo**: that force-pushes `main` back to the saved baseline and restores the
  baseline's PRs. **Present side by side** opens GitHub on the right half of the
  screen; it follows the pipeline (PR → Discord alert → merge → commits). Chrome may
  ask to allow pop-ups the first time.

## Evaluation

Ground truth lives in `eval/company_manifest.json` (tool repo only; the watcher never
sees it): every system on the company repo's `main` with the exact tables it reads and
writes, the duplicate clusters and partial duplicates planted among them, and every
`[TEST]` PR with its labels. `python -m eval.build_cases` derives `eval/cases.json`
from it: upstream/downstream relations are computed from the tables, duplicate /
partial / related ones are declared. "related" is ignored by scoring.

Company repo: 39 systems on `main` across 8 teams (~5.5k lines of new code), 25 test
PRs: duplicates (incl. different technique / modality), partial duplicates,
upstream consumers, downstream producers, unrelated systems (incl. hard negatives
like "invoice *matching*"), edits to existing folders, and a PR touching two folders.

All numbers on Qwen3-Coder-Next (OpenRouter), embeddings all-MiniLM-L6-v2.

**End to end** (`python -m eval.run_eval`, 25 PRs / 26 PR folders). The watcher is a
candidate generator, so the eval measures recall and the size of the candidate set,
not precision. Candidates(K) = every dataflow-linked folder + the top-K folders by
similarity above the floor (0.38):

| K | recall | candidates / PR folder | unrelated PRs with no candidates |
|---|---|---|---|
| 1 | 28/35 = 0.80 | 1.3 | 5/5 |
| 3 | 34/35 = 0.97 | 2.0 | 5/5 |
| **5 (in use)** | **35/35 = 1.00** | **2.3** | **5/5** |
| 10 | 35/35 = 1.00 | 2.3 | 5/5 |

The hardest case: image-based `visual_similarity` vs text-based `email_product_recs`
(similarity rank 5, score 0.39). The floor, not K, is what keeps unrelated PRs silent.

**Similarity signals** (`python -m eval.prompt_lab signals --by-group`): 185 queries /
160 positive: all PR folders, duplicates planted on `main`, demo snippets,
`eval/snippets/` (hand-written) and `eval/snippets/gen/` (LLM rewrites of every
`main` system in 3 styles + 12 unrelated systems, `eval/gen_snippets.py`).

| signal | R@3 | MRR | AUC |
|---|---|---|---|
| description only | 0.938 | 0.842 | 0.891 |
| keywords only (TF-IDF) | 0.943 | 0.844 | 0.907 |
| code only (functions) | 0.670 | 0.603 | 0.502 |
| mean(description, keywords, code) (previous) | 0.923 | 0.825 | 0.847 |
| **mean(description, keywords) (in use)** | **0.947** | **0.850** | **0.922** |

On the earlier toy code, function matching helped; on realistic code it matches shared
boilerplate (SQL loading, argparse mains) and is no better than chance, so it's kept
as evidence only. Floor 0.38: keeps 97% of true top-3 matches, drops 62% of wrong
ones, leaves 80% of unrelated systems with no candidate.

`python -m eval.prompt_lab prompts [variant ...]` compares description prompts.

## Run it end to end

Settings come from environment variables or a gitignored `.env` in the repo root (see
`watcher/config.py`). You need a GitHub token that can read/write the company monorepo
(contents, pull requests, issues): `GITHUB_TOKEN`, or a file at `GITHUB_TOKEN_FILE`.

### Quick start on a laptop (OpenRouter, no OpenClaw)

```sh
pip install -r requirements.txt
echo "sk-or-..." > .api_key                # OpenRouter key (gitignored)
python -m watcher.web --watcher            # dashboards + watcher loop: http://localhost:8765
```

The first run indexes `main` (41 folders, ~40 s on OpenRouter). Without the CLM stack,
run with `CLASSIFIER=1` (dummy LLM classifier in the tick; add `POST_GITHUB_COMMENTS=1`
for the `[oc]` PR comment). Without OpenClaw the note falls back to a template.

### On the GB10 box (local models + OpenClaw), as deployed

Everything runs from this checkout, `~/slopulant-agent`.

- **Models:** Ollama serving `coder-next:latest` (Qwen3-Coder-Next Q6) on :11434; the
  CLM encoder (Qwen3-8B GGUF, llama.cpp) on :8090, started by `scripts/start_clm.sh`.
- **Venvs:** `.venv` (watcher, `requirements.txt`), `~/clm-venv` (CLM worker,
  `clm_dupe/requirements.txt`), `discord_worker/.venv` (`discord_worker/requirements.txt`).
- **`.env`** (repo root, gitignored):
  ```
  LLM_PROFILE=local
  LLM_CACHE=0
  LLM_CONCURRENCY=4
  GITHUB_TOKEN_FILE=~/.slopulant/github_token
  ALLOW_GH_FALLBACK=0
  OPENCLAW_BIN=/home/dell/.local/opt/node-v24.21.0-linux-arm64/bin/openclaw
  OPENCLAW_AGENT=watcher
  PROGRESS=0
  DISCORD_GUILD_ID=<server id, for "View in Discord" links>
  ```
- **OpenClaw:** `deploy/openclaw_setup.sh` (idempotent) installs the `watcher` agent's
  instructions (`deploy/openclaw/watcher/*.md`), the Discord channel plugin, the config from
  `deploy/openclaw_config.py` (Discord mention-only in #slop-factory → `watcher`; read-only
  MCP tools; tool limits) and three jobs: `slopulant-watcher` and `slopulant-discord`
  (every 30 s) and `slopulant-reindex` (06:00 America/New_York, registered **disabled**;
  `--enable-reindex` turns it on). Secrets come from `discord_worker/.env`
  (`DISCORD_BOT_TOKEN`, `DISCORD_CHANNEL_ID`) and `.env` (`DISCORD_GUILD_ID`).
- **CLM worker:** systemd user unit `deploy/slopulant-clm.service` (starts the CLM
  servers first, so they come back after a reboot).
  ```sh
  cp deploy/slopulant-clm.service ~/.config/systemd/user/
  systemctl --user daemon-reload && systemctl --user enable --now slopulant-clm
  ```
- **Dashboards:** `python -m watcher.web --host 127.0.0.1 --port 8765`. From a laptop,
  tunnel with `ssh -L 8765:127.0.0.1:8765 dell@<gb10>` and open
  http://127.0.0.1:8765/demo.
- **Demo setup:**
  - Open test PRs from `/demo`; the test branches are listed there.
  - Save the baseline once from the **Data** tab, so **Reset demo** knows where to return.
  - `python -m watcher.backfill` analyses all test branches into the team view's history
    (it still uses the dummy LLM classifier).

### Other commands

```sh
python -m watcher.show               # latest candidates per PR (--all for every pair)
python -m eval.run_eval              # candidate-generator recall@K on the test PRs
python -m eval.export_for_jesse      # eval SQLite for the classifier (see for_jesse_readme.md)
python -m eval.export_feedback       # Discord 👍/👎 reviews as labelled CLM pairs (CSV)
python -m watcher.reindex --from-cache   # the daily re-index as a dry run (cached topics, no LLM)
(cd discord_worker && .venv/bin/python -m dupcheck_discord openclaw-preview)   # alert cards, not sent
~/clm-venv/bin/python -m watcher.clm_worker --once   # one CLM pass (env vars: clm_dupe/README.md)
```

| Setting | Default | What it does |
|---|---|---|
| `GITHUB_REPO` | `billy-mosse/slopulant-monorepo` | repo to watch |
| `LLM_PROFILE` | `openrouter` | `local` (Ollama, `LLM_LOCAL_URL` / `LLM_LOCAL_MODEL`) or `openrouter` (key in `.api_key`); switchable in the dashboard |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | topic embeddings |
| `POLL_SECONDS` | `30` | loop interval (laptop mode) |
| `TOP_K`, `MIN_CANDIDATE_SCORE` | `5`, `0.38` | candidate generation |
| `CHUNK_CHARS` | `40000` | folders bigger than this are chunked |
| `LLM_CONCURRENCY` | `8` | parallel LLM calls |
| `LLM_CACHE` | `1` | answer identical LLM requests from `data/llm_cache.json` (`0` for real timings) |
| `REEXTRACT_PR_TOPICS` | `1` | always re-extract a PR's folders (`main` stays cached) |
| `CLASSIFIER` | `0` | `1`: dummy LLM classifier inside the tick instead of the CLM worker |
| `CLM_EMB_URL`, `CLM_POLL_SECONDS` | `http://127.0.0.1:8090/v1/embeddings`, `5` | CLM worker: encoder, poll interval |
| `DISCORD_GUILD_ID` | | Discord server id, for links to alerts in the dashboards |
| `OPENCLAW_BIN`, `OPENCLAW_AGENT` | `openclaw`, `main` | agent that writes the note |
| `POST_GITHUB_COMMENTS`, `ALERT_COMMENT_MODE` | `0`, `repost` | with `CLASSIFIER=1`: post the `[oc]` PR comment; `repost` keeps it at the bottom, `edit` updates in place |

## Discord alerts and feedback

OpenClaw owns the Discord connection (bot `@slop-factory`, mention-only, #slop-factory
only). The `slopulant-discord` job posts one alert card per duplicate PR commit once the
CLM worker has finished it, and records 👍/👎 per reviewer. Reviews link back to the judged
pairs through the `discord_flagger_feedback` view (`pr_number`, `head_sha`, `base_sha`,
`source_decision_keys`); `python -m eval.export_feedback` exports them. See
[`discord_worker/README.md`](discord_worker/README.md) for the adapter and the standalone
bot (inbox mode) it grew out of.
