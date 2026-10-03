# CLM duplicate detector: how to use it

CLM decides whether a new system in a PR **duplicates** an existing system on `main`.
It is the classifier stage of the pipeline:

```
PR → watcher topics → candidate generator (CG) → CLM decision (is_dupe) → Discord
```

CLM only ranks and judges; it cannot generate text. It needs two servers running (below).

## 1. Start the servers

```bash
scripts/start_clm.sh
```

Safe to re-run: it starts only what is down. Nothing restarts automatically after a reboot.

| Service | Address | Runs on | Log |
|---|---|---|---|
| Encoder: Qwen3-8B bf16 GGUF, llama.cpp, last-token pooling | `http://127.0.0.1:8090/v1/embeddings` | GPU, ~17 GB | `~/models/encoder.log` |
| CLM heads + web playground (`clm-serve`) | `http://127.0.0.1:8700/` | CPU | `~/models/clm-serve.log` |

Health check: `curl localhost:8090/health` should return `{"status":"ok"}`.

The encoder runs with `-np 1` (one request at a time) on purpose: parallel slots change the
batching and shift scores by about 0.01, enough to flip decisions near the threshold.

## 2. Make a decision

Everything about the decision (prompt, layout, threshold, version tag) lives in
**`clm_dupe/decision.py`**. Import it; don't copy the prompt or threshold anywhere else.

```bash
export HF_HUB_OFFLINE=1 CLM_DEVICE=cpu \
       CLM_CKPT=$HOME/repos/RINGUSB1/clm-stack/CLM-v0.1-8B/CLM_v0.1-8B.pt
cd ~/slopulant-agent && ~/clm-venv/bin/python
```

```python
from clm import Engine
from clm_dupe import decision as D

engine = Engine(emb_url="http://127.0.0.1:8090/v1/embeddings")   # create once, reuse

existing = {"folder": "search_experiments", "name": "Synonym Mining",
            "description": "Analyzes in-session query reformulations ..."}
new      = {"folder": "synonym_expansion", "name": "Search Query Synonym Mining",
            "description": "Mines high-confidence synonym pairs from search logs ..."}

D.decide(engine, existing, new)
# {'is_dupe': True, 'score': 0.6..., 'threshold': 0.19,
#  'model_version': 'clm-v0.1-8B/sys-same_problem/th0.19'}
```

- `existing` / `new` are **one topic each**: `folder`, `name`, `description`. Keywords are not
  used (they didn't help in calibration).
- Store `score`, `threshold` and `model_version` with every decision, not just `is_dupe`, so
  the threshold can be changed later without re-running CLM.
- Score pairs **one at a time** (as `decide` does). Batching several states into one encoder
  request shifts scores by up to ~0.006.
- **Pairs with a data-flow link** (`dataflow` = `upstream:...` / `downstream:...`) should not go
  to CLM: the shared tables already give the relation, and on the eval set they were right
  17/17.

### The rule

| | |
|---|---|
| Question | "Do these two systems solve the same problem, even if they use different techniques?" → *The same problem.* / *Different problems.* |
| State layout | `Existing system on main: <folder>` / `Topic:` / description, then `New system in a pull request: <folder>` / `Topic:` / description |
| Decision | `is_dupe = p(yes) >= 0.19` |
| Speed | ~0.1 s per pair |

## 3. How good it is

Calibrated on `billy-mosse/slopulant-agent` `eval/for_jesse.sqlite`: the 33 candidate pairs
without a data-flow link (`candidate = 1 AND is_related = 0 AND dataflow IS NULL`), 20 of them
duplicates.

| Approach | Recall | Precision |
|---|---|---|
| **CLM, this rule (th 0.19)** | **20/20** | **20/25 = 0.80** |
| CG score alone, th 0.425 | 20/20 | 20/26 = 0.77 |
| Dummy LLM classifier | 17/20 | 17/19 = 0.89 |
| Old CLM reuse prompt | 20/20 | 20/32 = 0.62 |

Read this with care:

- **Small set.** 33 pairs; the gain over the CG score is about two false positives. Treat it as a
  sanity check, not a benchmark.
- **The question was chosen from 54 variants on this same data.** The "same problem" wording was
  consistently strong across every layout, so the effect is real, but picking a single best
  variant overfits (nested leave-one-PR-out estimate: recall 17/20, precision 0.65). Don't
  tune the prompt further on this set; new wording needs new labelled PRs.
- **Why 0.19 and not 0.209.** 0.209 is the score of the weakest duplicate, only 0.004 above a
  non-duplicate (0.205). 0.19 sits in the middle of the next gap (0.175–0.205).
- **Remaining false positives are near neighbours**, e.g. `visual_similarity` vs
  `product_embeddings`, `sku_dedup` vs `competitor_price_matching`.
- **Recall bonus.** Over all pairs, CLM ranks 2 of the 3 duplicates the CG misses #1 for their
  PR. Scoring all pairs (not just candidates) could catch those; not built yet.

## 4. Re-running calibration

All in `clm_dupe/calibration/` (the eval set is `eval/for_jesse.sqlite`) (servers must be up, env vars as in section 2):

| File | What |
|---|---|
| `../decision.py` | **The rule in use** |
| `score_pairs.py` → `clm_scores.sqlite` | Scores all 1,186 pairs with the old reuse prompts |
| `calibrate.py` | Threshold + leave-one-PR-out for those (`python3 calibrate.py`) |
| `prompt_search.py` → `prompt_search.sqlite` | 6 layouts × 9 questions on the 33 pairs |
| `analyze_search.py` | Leaderboard + nested leave-one-PR-out (`python3 analyze_search.py`) |

If the encoder, checkpoint, prompt or topic-generation model changes, re-score and re-pick the
threshold, then bump `VERSION` in `clm_dupe/decision.py`.

## 5. Where everything is

| What | Path |
|---|---|
| CLM heads (78 MB) | `~/repos/RINGUSB1/clm-stack/CLM-v0.1-8B/` |
| Encoder weights, original safetensors | `~/repos/RINGUSB1/clm-stack/Qwen3-8B/` |
| Encoder weights, GGUF actually served | `~/models/qwen3-8b-bf16.gguf` |
| llama.cpp build (CUDA, sm_121) | `~/llama.cpp-src/llama.cpp-0.5.0/build/bin/` |
| Python env (`contrastive-lm`, CPU torch) | `~/clm-venv/` |
| Start script | `scripts/start_clm.sh` |
| Reuse-detector (earlier prompt work + its eval) | `~/repos/RINGUSB1/clm-stack/reuse-detector/` |

The GGUF encoder was checked against the Hugging Face reference: cosine ≥ 0.99997, and the
reuse-detector eval reproduces its published numbers (AUC 0.957, 7/7).

## Where it runs

`watcher/clm_worker.py` (`~/clm-venv/bin/python -m watcher.clm_worker`, systemd unit
`deploy/slopulant-clm.service`) polls the candidates in `data/watcher.db`, calls `decide()` for
each non-dataflow candidate, and writes `decisions` (`is_duplicate`, `confidence` = score,
`threshold`, `classifier` = `VERSION`). The unit runs `scripts/start_clm.sh` first, so the
servers come back after a reboot.
