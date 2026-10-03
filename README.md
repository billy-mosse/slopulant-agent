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
4. **Classifier** (`watcher/classifier.py`) judges each candidate:
   duplicate / partial / upstream / downstream / unrelated. It's a dummy LLM
   classifier now; the CLM classifier replaces `classify()`. Eval data for it:
   [`for_jesse_readme.md`](for_jesse_readme.md).
5. **Alert.** The OpenClaw agent `watcher` writes a short note to the PR author. The
   watcher posts it as a single `[oc]` comment on the PR, together with links, owners
   and verdicts built from data. The comment is replaced with a fresh one on every
   re-score (`ALERT_COMMENT_MODE=repost`), so it's always at the bottom of the PR.

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
  screen; it follows the pipeline (PR → `[oc]` comment → merge → commits). Chrome may
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

The first run indexes `main` (41 folders, ~40 s on OpenRouter). Without OpenClaw the
note in the PR comment falls back to a template; everything else works.

### On the GB10 box (local model + OpenClaw), as deployed

- **Code:** `~/slopulant-watcher` (synced from this repo), Python from a venv with
  `requirements.txt`.
- **Model:** Ollama serving `coder-next:latest` (Qwen3-Coder-Next Q6) on :11434.
- **`.env`:**
  ```
  LLM_PROFILE=local
  LLM_CACHE=0
  LLM_CONCURRENCY=4
  GITHUB_TOKEN_FILE=~/.slopulant/github_token
  ALLOW_GH_FALLBACK=0
  OPENCLAW_BIN=/home/dell/.local/opt/node-v24.21.0-linux-arm64/bin/openclaw
  OPENCLAW_AGENT=watcher
  PROGRESS=0
  ```
- **OpenClaw:**
  - Create the agent once: `openclaw agents add watcher --non-interactive --workspace ~/.openclaw/workspace-watcher --model ollama/coder-next:latest`
  - Register the scheduled job (it runs one tick every 30 s; ticks never overlap):
    ```sh
    openclaw cron add --name slopulant-watcher --every 30s --no-deliver --timeout-seconds 3600 \
      --command "cd ~/slopulant-watcher && <venv>/bin/python -m watcher.main --once"
    ```
- **Dashboards:** `python -m watcher.web --host 127.0.0.1 --port 8765`. From a laptop,
  tunnel with `ssh -L 8765:127.0.0.1:8765 dell@<gb10>` and open
  http://127.0.0.1:8765/demo.
- **Demo setup:**
  - Open test PRs from `/demo`; the test branches are listed there.
  - Save the baseline once from the **Data** tab, so **Reset demo** knows where to return.
  - `python -m watcher.backfill` analyses all test branches into the team view's history.

### Other commands

```sh
python -m watcher.show               # latest candidates per PR (--all for every pair)
python -m eval.run_eval              # candidate-generator recall@K on the test PRs
python -m eval.export_for_jesse      # eval SQLite for the classifier (see for_jesse_readme.md)
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
| `CLASSIFIER` | `1` | run the classifier on candidates |
| `OPENCLAW_BIN`, `OPENCLAW_AGENT` | `openclaw`, `main` | agent that writes the note |
| `POST_GITHUB_COMMENTS`, `ALERT_COMMENT_MODE` | `1`, `repost` | post the `[oc]` comment; `repost` keeps it at the bottom, `edit` updates in place |

## Discord alerts and feedback

The separate [Discord worker](discord_worker/README.md) posts duplicate alerts, records 👍/👎 reviews, and stores per-message like/dislike totals in SQLite. It includes its own dependencies, tests, examples, and a snapshot of the GB10's shared database; see its README for setup.

The current `team_sqlite` adapter reads `Commited`, `new_prs`, `dup_cg`, and `dupe_decision`. This differs from the watcher's `data/watcher.db` schema, so the watcher is not automatically connected to it. A detector can also send structured decisions through the worker's JSON inbox API. The watcher commands above remain the way to run the watcher; run the Discord worker as a separate component.
