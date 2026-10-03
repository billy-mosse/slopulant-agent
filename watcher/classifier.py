"""Duplicate classifier: given a PR topic and a candidate topic on main, decide how
they relate.

This is a DUMMY classifier (one LLM yes/no-style question per candidate) that the
CLM classifier will replace. Keep the interface:

    classify(pr: dict, candidate: dict, signals: dict) -> dict
      pr / candidate: {"folder", "name", "description", "keywords", "inputs", "outputs"}
      signals: {"score", "desc_score", "kw_score", "dataflow"}
      returns {"relation": duplicate|partial|upstream|downstream|unrelated,
               "is_duplicate": bool, "confidence": 0..1, "reason": str}
"""
import json

from . import config, topics

NAME = "llm-dummy-v1"
RELATIONS = ("duplicate", "partial", "upstream", "downstream", "unrelated")

PROMPT = """Two systems at an e-commerce company. Decide how the NEW system (from a pull request)
relates to the EXISTING one.

- duplicate: the new system does the same job as the existing one (same purpose, even if the
  technique, names or code differ). The author should reuse or talk to its owner.
- partial: the new system re-implements a component of the existing one (e.g. rebuilds
  features or vectors the existing system already produces) but its overall job differs.
- upstream: the new system consumes what the existing one produces.
- downstream: the existing system consumes what the new one produces.
- unrelated: none of the above; similar wording or technique alone is not enough.

NEW ({new_folder}): {new_name}
{new_description}
keywords: {new_keywords}
reads: {new_inputs} | writes: {new_outputs}

EXISTING ({old_folder}): {old_name}
{old_description}
keywords: {old_keywords}
reads: {old_inputs} | writes: {old_outputs}

Shared tables: {dataflow}

Reply with JSON only:
{{"relation": "<one of duplicate|partial|upstream|downstream|unrelated>", "confidence": <0 to 1>, "reason": "<one sentence>"}}"""


def classify(pr, candidate, signals):
    fmt = lambda xs: ", ".join(xs) or "-"
    prompt = PROMPT.format(
        new_folder=pr["folder"], new_name=pr["name"], new_description=pr["description"],
        new_keywords=fmt(pr["keywords"][:15]), new_inputs=fmt(pr["inputs"]), new_outputs=fmt(pr["outputs"]),
        old_folder=candidate["folder"], old_name=candidate["name"], old_description=candidate["description"],
        old_keywords=fmt(candidate["keywords"][:15]), old_inputs=fmt(candidate["inputs"]), old_outputs=fmt(candidate["outputs"]),
        dataflow=signals.get("dataflow") or "none",
    )
    parsed = topics._json(topics._chat(prompt, max_tokens=200)) or {}
    relation = str(parsed.get("relation", "")).strip().lower()
    if relation not in RELATIONS:
        relation = "unrelated"
    try:
        confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))
    except (TypeError, ValueError):
        confidence = 0.5
    return {
        "relation": relation,
        "is_duplicate": relation in ("duplicate", "partial"),
        "confidence": confidence,
        "reason": str(parsed.get("reason") or "").strip()[:300] or "(no reason given)",
    }


def topic_by_name(entry, name):
    for t in entry["topics"]:
        if t["name"] == name:
            return t
    return entry["topics"][0]


def as_input(entry, topic):
    return {"folder": entry["folder"], **{k: topic[k] for k in ("name", "description", "keywords", "inputs", "outputs")}}


if __name__ == "__main__":  # quick manual check: python -m watcher.classifier
    a = {"folder": "a", "name": "churn", "description": "Predicts 90-day churn from RFM.", "keywords": ["churn"], "inputs": ["orders.lines"], "outputs": ["scores.p_churn"]}
    print(json.dumps(classify(a, a, {"dataflow": None}), indent=2), config.llm()["model"])
