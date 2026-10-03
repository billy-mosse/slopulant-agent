# For Jesse: eval data for the CLM classifier

You're replacing the dummy duplicate classifier with CLM and need to set its thresholds.
Everything you need is in one SQLite file. No topics need to be generated again.

**`eval/for_jesse.sqlite`**: generated with the **local model** we'll use at the
hackathon (Qwen3-Coder-Next Q6 on Ollama, GB10 box) and all-MiniLM-L6-v2 embeddings.
The `meta` table records exactly what produced it.

## Where the classifier sits

```
PR pushed → watcher splits each touched folder into topics (1..N per folder)
          → candidate generator: compares every PR topic with every topic on main
          → candidates = shared-table links (always) + top-5 by similarity score ≥ 0.38
          → CLASSIFIER (you): judges each candidate → relation + confidence
          → OpenClaw agent writes the [oc] comment on the PR
```

The candidate generator is tuned for **recall**: the right systems almost always show up
among the candidates, and some wrong ones come along too. Your classifier's job is
**precision**: keep the real ones and drop the rest.

## The data

The company is a mock home-goods e-commerce ML monorepo (`billy-mosse/slopulant-monorepo`):
41 systems ("folders") on `main`, plus 28 test PR branches that add or edit 29 folders.
The ground truth comes from `eval/company_manifest.json` via `eval/build_cases.py`:

| relation | meaning (PR folder vs folder on main) | how it's labelled |
|---|---|---|
| `duplicate` | does the same job (different code, names or even technique) | declared |
| `partial` | re-implements a component of the other (e.g. recomputes its features) | declared |
| `upstream` | the PR reads a table the main folder writes | computed from tables |
| `downstream` | the main folder reads a table the PR writes | computed from tables |
| `related` | plausible but debatable overlap | declared: **exclude from scoring** |
| `none` | everything else | |

A pair can carry several labels (e.g. `duplicate,related`).

### Tables

| table | rows | what |
|---|---|---|
| `meta` | | model, embedding model, base commit, K / floor, when it was built |
| `folders` | 70 | 41 on main (`branch='main'`) + 29 PR folders (`branch=<test branch>`), with owner |
| `topics` | ~80 | 1..N per folder: `name`, `description`, `keywords` (JSON), `inputs`/`outputs` (JSON tables), `embedding` (float32[384], L2-normalised) |
| `pairs` | ~1,190 | **every** (PR folder, main folder) pair: ground truth + candidate-generator signals |
| `dummy_classifier` | ~55 | the current dummy LLM classifier's verdicts on candidate pairs: the **baseline to beat** |
| `prs` | 28 | test branch, title, author, head commit |

Key `pairs` columns:

- `truth`, `is_positive` (duplicate/partial/upstream/downstream), `is_related` (exclude these)
- `score` = `(desc_score + kw_score) / 2` of the **best-matching topic pair**
  - `desc_score`: cosine of the two topics' description embeddings
  - `kw_score`: TF-IDF cosine of their keywords
- `pr_topic`, `repo_topic`: which topics matched. Join to `topics` on `(branch, pr_folder, name)` for the PR side and `('main', repo_id, name)` for the main side.
- `dataflow`: `upstream:<tables>` / `downstream:<tables>` when they share a table
- `sim_rank`: rank among non-dataflow pairs; `candidate`: 1 if the watcher would send it to you today

## Recipes

Load candidate pairs with both topics, ready to feed a classifier:

```python
import json, sqlite3
db = sqlite3.connect("eval/for_jesse.sqlite"); db.row_factory = sqlite3.Row

def topic(branch, folder, name):
    t = db.execute("SELECT * FROM topics WHERE branch=? AND folder=? AND name=?", (branch, folder, name)).fetchone()
    return {"folder": folder, "name": t["name"], "description": t["description"],
            **{k: json.loads(t[k]) for k in ("keywords", "inputs", "outputs")}}

pairs = db.execute("SELECT * FROM pairs WHERE candidate = 1 AND is_related = 0").fetchall()
examples = [(topic(p["branch"], p["pr_folder"], p["pr_topic"]),      # new system (PR)
             topic("main", p["repo_id"], p["repo_topic"]),            # existing system
             dict(p))                                                 # signals + truth
            for p in pairs]
```

Use `candidate = 1` for what the classifier sees in production. Use all `pairs` if you want
harder negatives or to study the candidate generator itself.

How good is the baseline?

```sql
SELECT p.truth, d.relation, count(*) FROM dummy_classifier d
JOIN pairs p USING (branch, pr_folder, repo_id) GROUP BY 1, 2 ORDER BY 1, 3 DESC;
```

In this file, the candidate generator sends 55 pairs to the classifier, and they include
**37 of the 40 positives**. The dummy classifier's verdicts on those pairs:

- **Duplicates (19):** 5 called `duplicate`, 11 `partial`, 2 `unrelated`, 1 `upstream`.
  Duplicate vs. partial is its weak spot.
- **Direction:** all 4 true `downstream` pairs come out as `upstream`, and 3 of 13 `upstream`
  pairs come out as `downstream`.
- **Negatives:** 11 of 13 `none` pairs are correctly `unrelated`; 2 are called `partial`.

Plenty of room to beat it.

## What to tune

1. **Decision thresholds** on CLM's score(s): per relation if CLM gives you one, or one
   "surface this to the author" threshold. Pick it on `candidate = 1 AND is_related = 0`,
   maximising precision at (near) full recall on `is_positive`.
2. For `upstream`/`downstream`, `dataflow` already gives the direction from the tables. You
   can trust it and only classify duplicate / partial / unrelated if that's easier.
3. **Optional:** if CLM is cheap enough, score *all* pairs, not just candidates. You may then be able to
   loosen the candidate generator (more K, a lower floor) and catch the few positives it misses.
   Check them with `SELECT * FROM pairs WHERE is_positive = 1 AND candidate = 0`.

Mind the size: 29 PR folders and about 40 positive pairs. Treat it as a sanity and calibration set,
not a benchmark, and don't over-fit thresholds to a handful of cases.

## Plugging CLM in

Replace `classify()` in `watcher/classifier.py`. Its interface:

```python
classify(pr: dict, candidate: dict, signals: dict) -> dict
#   pr / candidate: {"folder", "name", "description", "keywords", "inputs", "outputs"}  (one topic each)
#   signals:        {"score", "desc_score", "kw_score", "dataflow", ...}
#   returns {"relation": "duplicate|partial|upstream|downstream|unrelated",
#            "is_duplicate": bool, "confidence": 0..1, "reason": str}
```

The watcher stores the verdicts and the OpenClaw agent turns them into the `[oc]` comment. Nothing else changes.

## Regenerating (only if the code or the model changes)

```sh
python -m eval.export_for_jesse                          # active model; reuses cached topics
LLM_PROFILE=openrouter python -m eval.export_for_jesse   # same, with Qwen3-Coder-Next bf16 on OpenRouter
python -m eval.export_for_jesse --no-dummy-classifier    # skip the baseline verdicts
```

End-to-end candidate-generator metrics are in `python -m eval.run_eval`.
