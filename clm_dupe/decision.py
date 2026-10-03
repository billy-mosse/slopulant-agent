"""CLM duplicate decision: the calibrated prompt + threshold. Single source of truth.

Calibrated 2026-10-03 on billy-mosse/slopulant-agent eval/for_jesse.sqlite (candidate = 1,
is_related = 0, dataflow IS NULL: 33 pairs, 20 duplicates). See analyze_search.py.

  variant    "sys | same_problem" from prompt_search.py (54 variants tried; keywords not used)
  threshold  0.19 = midpoint of the 0.175..0.205 gap below the weakest duplicate. The full-recall
             edge (0.209) sits 0.004 above a negative (0.205), less than the ~0.006 score shift
             from encoder batching, so it is not used.
  in-sample  recall 20/20, precision 20/25 = 0.80, AUC 0.927
  LOPO       (at the 0.209 edge) recall 19/20, precision 0.83; nested-selection estimate 0.65

Encoder: Qwen3-8B bf16 GGUF on llama.cpp, --pooling last, -np 1 (scripts/start_clm.sh).
Pairs with a dataflow link are not sent to CLM: the tables give upstream/downstream.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the dashboard reads VERSION/THRESHOLD without the CLM venv
    from clm import Engine

VERSION = "clm-v0.1-8B/sys-same_problem/th0.19"
THRESHOLD = 0.19

QUESTION = {
    "same_problem": {
        "type": "noul",
        "instructions": "Do these two systems solve the same problem, even if they use different techniques?",
        "criteria": {"true": "The same problem.", "false": "Different problems."},
    }
}


def state(existing: dict, new: dict) -> str:
    """existing / new: {"folder", "name", "description"} (one topic each). Existing system first."""
    return (f"Existing system on main: {existing['folder']}\nTopic: {existing['name']}\n{existing['description']}"
            f"\n\n\nNew system in a pull request: {new['folder']}\nTopic: {new['name']}\n{new['description']}")


def score(engine: Engine, existing: dict, new: dict) -> float:
    """p(yes) that the two systems solve the same problem.

    One state per encoder request: batching several texts into one request shifts scores
    by up to ~0.006, so always score pairs one at a time to stay on the calibration."""
    return engine.answer(state(existing, new), QUESTION)["answers"]["same_problem"]["noul"]


def decide(engine: Engine, existing: dict, new: dict) -> dict:
    p = score(engine, existing, new)
    return {"is_dupe": p >= THRESHOLD, "score": p, "threshold": THRESHOLD, "model_version": VERSION}
