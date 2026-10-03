# slopulant-agent — duplicate-effort agent

This repo is **the tool**: an always-on agent that watches PRs and flags work that
duplicates (or could reuse) an existing system. It runs fully on local models.

The code it analyses lives in a **separate repo**:
[`billy-mosse/slopulant-monorepo`](https://github.com/billy-mosse/slopulant-monorepo) —
the mock "Slopulent Living" company ML monorepo. Each top-level folder there is one
system ("repo"). Nothing in this repo should be committed there, and vice versa.

## How the watcher works

Every `POLL_SECONDS` (30):

1. **Index base branch.** If `main` moved, build a *card* for each top-level folder,
   in parallel (`LLM_CONCURRENCY`, default 8), cached by git tree sha + model + prompt
   version so unchanged folders are never re-processed. Three LLM calls per card, run
   concurrently:
   - **description**: capability card, embedded
   - **keywords**: 10-15 technical keywords, then generic terms (ML, AI, model, data,
     code, ...) and stop words removed in code (`watcher/keywords.py`)
   - **inputs/outputs**: tables the code reads/writes
   plus **functions**: every non-trivial function's source, embedded.
2. **Poll** open PRs (GitHub API with ETags, so idle polls cost no rate limit). A PR
   is (re)queued per `(head sha, base sha, card version)`: new pushes, merges to
   `main` and prompt changes all trigger a re-score.
3. **Score.** For each folder the PR touches, build its card *from the PR branch* and
   compare it with every other folder on `main`:
   - `card_score`: cosine of description embeddings
   - `kw_score`: TF-IDF cosine of keyword tokens (IDF fitted on `main`), with
     `kw_match` listing the shared keywords, rarest first
   - `code_score`: best cosine between any two functions, with `code_match`
   - `dataflow`: `upstream:<table>` / `downstream:<table>` producer-consumer links
   - `score` = **mean(card, kw)**; `code_score` is evidence only
   - `candidate` = every dataflow link, plus the top `TOP_K` (5) non-dataflow
     folders above `MIN_CANDIDATE_SCORE` (0.38).

State is SQLite at `data/watcher.db`: `pr_queue`, `folder_cards`, `scores`, and the
`latest_scores` view (most recent run per PR).

## Demo UI

`python -m watcher.web` serves a local dashboard and runs the watcher loop in the
same process:

- **Pull requests**: every open PR with the LLM's card for each folder it touches,
  and the ranked folders on `main` (candidates first, with why: card/code scores,
  the matching function pair, or the data-flow link). Refreshes every 4s.
- **Try it**: paste code for a new system and score it against `main` without
  opening a PR (same code path as the watcher). Comes with example snippets.
- **Index (main)**: the card for every folder on `main`.

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

## Run

```sh
pip install -r requirements.txt
# LLM: Qwen3-Coder-Next on OpenRouter by default; key read from .api_key (gitignored)
python -m watcher.web             # demo UI on http://localhost:8765 (runs the loop too)
python -m watcher.main            # loop only; add --once for a single pass
python -m watcher.show            # latest candidates per PR (--all for every pair)
```

At the hackathon, serve Qwen3-Coder-Next locally (vLLM) and point the watcher at it:
`LLM_BASE_URL=http://localhost:8000/v1 LLM_MODEL=<served name>`.
`scripts/serve_llm.sh` still runs a small MLX model on a Mac for offline dev.

Everything is configured via env vars (see `watcher/config.py`). On the hackathon
box, serve the big model with vLLM and set `LLM_BASE_URL` / `LLM_MODEL`.

| Env var | Default |
|---|---|
| `GITHUB_REPO` | `billy-mosse/slopulant-monorepo` |
| `LLM_BASE_URL` | `https://openrouter.ai/api/v1` |
| `LLM_MODEL` | `qwen/qwen3-coder-next` |
| `LLM_API_KEY` | from `.api_key` when using OpenRouter |
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` |
| `POLL_SECONDS` | `30` |
| `TOP_K` | `5` |
| `MIN_CANDIDATE_SCORE` | `0.38` |
| `LLM_CONCURRENCY` | `8` |
| `LLM_CACHE` | `1`: identical LLM requests are answered from `data/llm_cache.json`; `0` disables |

## Discord alerts and feedback

The separate [Discord worker](discord_worker/README.md) posts duplicate alerts, records 👍/👎 reviews, and stores per-message like/dislike totals in SQLite. It includes its own dependencies, tests, examples, and a snapshot of the GB10's shared database; see its README for setup.

The current `team_sqlite` adapter reads `Commited`, `new_prs`, `dup_cg`, and `dupe_decision`. This differs from the watcher's `data/watcher.db` schema, so the watcher is not automatically connected to it. A detector can also send structured decisions through the worker's JSON inbox API. The watcher commands above remain the way to run the watcher; run the Discord worker as a separate component.
