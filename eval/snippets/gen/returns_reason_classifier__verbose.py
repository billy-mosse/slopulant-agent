"""Return reason inference pipeline.

This module supports the Returns Experience team by automatically assigning
high-fidelity reason codes to new return requests based on customer comments.
It combines deterministic pattern matching for clear-cut cases with a
statistical classifier trained on historical agent-verified decisions.

The system prioritizes precision: ambiguous or low-confidence cases are
flagged for human review rather than risk of misclassification.
"""
from __future__ import annotations

import argparse
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger("returns_inference")

VALID_REASONS = (
    "TOO_SMALL",
    "COLOR_DIFFERENT",
    "DAMAGED",
    "QUALITY",
    "CHANGED_MIND",
    "LATE",
)
CONFIDENCE_THRESHOLD = 0.55
SMOOTHING_FACTOR = 1.0

EXPLICIT_PATTERNS = {
    "DAMAGED": [
        r"\b(arrived|came) (broken|torn|ripped|cracked|damaged)\b",
        r"\bshattered\b",
        r"\bstain(ed|s)? (on|out of) the box\b",
    ],
    "LATE": [
        r"\b(arrived|came|delivered) (too )?late\b",
        r"\bmissed (the|my) (event|date)\b",
        r"\bnever arrived on time\b",
    ],
    "TOO_SMALL": [
        r"\btoo (small|short|narrow)\b",
        r"\bdoesn'?t fit (my|the) (bed|mattress)\b",
    ],
    "COLOR_DIFFERENT": [
        r"\b(colou?r|shade) (is |was )?(different|off|not the same)\b",
    ],
}
COMMON_WORDS = {
    "the", "a", "an", "and", "it", "is", "was", "i", "to", "of", "for", "my",
    "this", "that", "in", "on", "with", "but", "so", "me", "they", "be",
    "at", "as",
}


def sanitize_text(raw: str | None) -> str:
    if not raw:
        return ""
    normalized = raw.lower()
    normalized = re.sub(r"https?://\S+|\S+@\S+", " ", normalized)
    normalized = re.sub(r"\border\s*#?\s*\d+\b", " ", normalized)
    normalized = re.sub(r"[^a-z' ]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def extract_tokens(text: str) -> list[str]:
    base_tokens = [
        token.strip("'")
        for token in text.split()
        if token not in COMMON_WORDS and len(token) > 1
    ]
    bigrams = [f"{a}_{b}" for a, b in zip(base_tokens, base_tokens[1:])]
    return base_tokens + bigrams


def apply_explicit_rules(comment: str) -> str | None:
    for category, patterns in EXPLICIT_PATTERNS.items():
        if any(re.search(pattern, comment) for pattern in patterns):
            return category
    return None


@dataclass
class BayesianReasonInferencer:
    smoothing: float = SMOOTHING_FACTOR
    class_log_priors: dict[str, float] = field(default_factory=dict)
    feature_log_likelihoods: dict[str, dict[str, float]] = field(default_factory=dict)
    unseen_feature_log_prob: dict[str, float] = field(default_factory=dict)
    vocabulary: set[str] = field(default_factory=set)

    def train(self, documents: list[list[str]], labels: list[str]) -> "BayesianReasonInferencer":
        class_counts = Counter(labels)
        feature_counts: dict[str, Counter] = defaultdict(Counter)
        for tokens, label in zip(documents, labels):
            feature_counts[label].update(tokens)
            self.vocabulary.update(tokens)

        vocab_size = len(self.vocabulary)
        total_docs = len(labels)

        for category in VALID_REASONS:
            self.class_log_priors[category] = math.log(
                (class_counts[category] + 1) / (total_docs + len(VALID_REASONS))
            )
            total_feature_count = sum(feature_counts[category].values()) + self.smoothing * vocab_size
            self.unseen_feature_log_prob[category] = math.log(self.smoothing / total_feature_count)
            self.feature_log_likelihoods[category] = {
                feature: math.log((count + self.smoothing) / total_feature_count)
                for feature, count in feature_counts[category].items()
            }

        return self

    def compute_posteriors(self, tokens: list[str]) -> dict[str, float]:
        tokens = [t for t in tokens if t in self.vocabulary]
        scores = {}
        for category in VALID_REASONS:
            score = self.class_log_priors[category]
            for token in tokens:
                score += self.feature_log_likelihoods[category].get(
                    token, self.unseen_feature_log_prob[category]
                )
            scores[category] = score

        max_score = max(scores.values())
        exp_shifted = {c: math.exp(s - max_score) for c, s in scores.items()}
        total = sum(exp_shifted.values())
        return {c: p / total for c, p in exp_shifted.items()}


def process_returns(returns: pd.DataFrame, inferencer: BayesianReasonInferencer) -> pd.DataFrame:
    results = []
    for record in returns.itertuples():
        comment = sanitize_text(getattr(record, "comment", None))
        if not comment:
            results.append((getattr(record, "return_id"), "UNKNOWN", 0.0, "empty_comment"))
            continue

        explicit_match = apply_explicit_rules(comment)
        if explicit_match:
            results.append((getattr(record, "return_id"), explicit_match, 1.0, "explicit_rule"))
            continue

        posteriors = inferencer.compute_posteriors(extract_tokens(comment))
        top_category = max(posteriors, key=posteriors.get)
        confidence = posteriors[top_category]

        if confidence < CONFIDENCE_THRESHOLD:
            results.append((getattr(record, "return_id"), "UNKNOWN", confidence, "low_confidence"))
        else:
            results.append((getattr(record, "return_id"), top_category, confidence, "statistical"))

    return pd.DataFrame(
        results,
        columns=["return_id", "reason_code", "confidence", "inference_method"],
    )


def run_pipeline(dsn: str, cutoff_date: str, dry_run: bool) -> None:
    logger.info("Initializing return reason inference pipeline")
    engine = create_engine(dsn)

    query = text(
        "SELECT return_id, comment, verified_reason, created_at "
        "FROM returns.return_requests WHERE created_at >= :cutoff_date"
    )
    raw_data = pd.read_sql(query, engine, params={"cutoff_date": cutoff_date})
    training_set = raw_data[raw_data["verified_reason"].isin(VALID_REASONS)]

    inferencer = BayesianReasonInferencer().train(
        [extract_tokens(sanitize_text(c)) for c in training_set["comment"]],
        training_set["verified_reason"].tolist(),
    )
    logger.info(
        "Trained Bayesian model on %d verified cases; vocabulary size: %d",
        len(training_set),
        len(inferencer.vocabulary),
    )

    unlabelled = raw_data[raw_data["verified_reason"].isna()]
    predictions = process_returns(unlabelled, inferencer)
    logger.info("Prediction distribution:\n%s", predictions["reason_code"].value_counts().to_string())

    if not dry_run:
        predictions.to_sql(
            "reason_labels",
            engine,
            schema="returns",
            if_exists="append",
            index=False,
        )
        logger.info("Results persisted to returns.reason_labels")


def main() -> None:
    parser = argparse.ArgumentParser(description="Infer return reasons from customer comments")
    parser.add_argument("--dsn", required=True, help="Database connection string")
    parser.add_argument("--cutoff_date", default="2026-01-01", help="Earliest return date to process")
    parser.add_argument("--dry_run", action="store_true", help="Skip writing results to DB")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    run_pipeline(args.dsn, args.cutoff_date, args.dry_run)


if __name__ == "__main__":
    main()
