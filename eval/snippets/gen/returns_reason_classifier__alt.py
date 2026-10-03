import argparse
import logging
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("return_analyzer")

# Candidate return categories—must match those used in training labels
CATEGORIES = {"TOO_SMALL", "COLOR_DIFFERENT", "DAMAGED", "QUALITY", "CHANGED_MIND", "LATE"}
CONF_THRESHOLD = 0.55
SMOOTHING_PARAM = 1.0

# High-confidence lexical signals—only used if no model override
EXPLICIT_PATTERNS = {
    "DAMAGED": [r"\b(broken|fractured|shattered|cracked)\b", r"\brips|tears\b"],
    "LATE": [r"\bdelayed|postponed|missed\b", r"\barrived (late|after the date)\b"],
    "TOO_SMALL": [r"\b(size|fit|length) too (small|tight|short)\b", r"\bnot (fitting|can't fit)\b"],
    "COLOR_DIFFERENT": [r"\b(color|hue) isn’t|is not (the|my) (color|shade)\b", r"\bwrong color\b"],
}

EXTRA_STOP = {"the", "a", "an", "and", "it", "is", "was", "i", "to", "of", "for", "my", "this",
              "that", "in", "on", "with", "but", "so", "me", "they", "be", "at", "as", "have",
              "has", "had", "do", "does", "did"}


def preprocess(text: str | None) -> str:
    if not text:
        return ""
    txt = text.lower()
    txt = re.sub(r"http[s]?://\S+|mailto:\S+", " ", txt)
    txt = re.sub(r"\border\s*(?:ID)?\s*\d+", " ", txt)
    txt = re.sub(r"[^a-z\s]", " ", txt)
    return re.sub(r"\s+", " ", txt).strip()


def extract_tokens(doc: str) -> list[str]:
    parts = [w.strip("'") for w in doc.split() if w not in EXTRA_STOP and len(w) > 1]
    return parts + [f"{u}_{v}" for u, v in zip(parts, parts[1:])] if parts else []


def detect_explicit(text: str) -> str | None:
    for cat, regexes in EXPLICIT_PATTERNS.items():
        if any(re.search(p, text) for p in regexes):
            return cat
    return None


@dataclass
class DirichletClassifier:
    laplace: float = SMOOTHING_PARAM
    class_probs: dict[str, float] = field(default_factory=dict)
    word_dists: dict[str, dict[str, float]] = field(default_factory=dict)
    unseen_logp: dict[str, float] = field(default_factory=dict)
    feature_set: set[str] = field(default_factory=set)

    def train(self, docs: list[list[str]], targets: list[str]) -> "DirichletClassifier":
        cat_counts = Counter(targets)
        word_counts: dict[str, Counter] = defaultdict(Counter)
        for tokens, label in zip(docs, targets):
            word_counts[label].update(tokens)
            self.feature_set.update(tokens)
        vocab_size = len(self.feature_set)
        total_docs = len(targets)
        for cat in CATEGORIES:
            prior = (cat_counts[cat] + 1) / (total_docs + len(CATEGORIES))
            self.class_probs[cat] = math.log(prior)
            cat_total = sum(word_counts[cat].values()) + self.laplace * vocab_size
            self.unseen_logp[cat] = math.log(self.laplace / cat_total)
            self.word_dists[cat] = {
                w: math.log((cnt + self.laplace) / cat_total)
                for w, cnt in word_counts[cat].items()
            }
        return self

    def infer(self, tokens: list[str]) -> dict[str, float]:
        if not tokens:
            return {c: 1.0 / len(CATEGORIES) for c in CATEGORIES}
        scores = {}
        for cat in CATEGORIES:
            score = self.class_probs[cat]
            for token in tokens:
                if token in self.word_dists[cat]:
                    score += self.word_dists[cat][token]
                else:
                    score += self.unseen_logp[cat]
            scores[cat] = score
        max_score = max(scores.values())
        denom = sum(math.exp(s - max_score) for s in scores.values())
        return {c: math.exp(s - max_score) / denom for c, s in scores.items()}


def label_requests(data: pd.DataFrame, model: DirichletClassifier) -> pd.DataFrame:
    results = []
    for _, row in data.iterrows():
        txt = preprocess(row.comment)
        if not txt:
            results.append({"return_id": row.return_id, "category": "UNKNOWN", "score": 0.0, "source": "blank"})
            continue
        explicit = detect_explicit(txt)
        if explicit:
            results.append({"return_id": row.return_id, "category": explicit, "score": 1.0, "source": "pattern"})
            continue
        tokenized = extract_tokens(txt)
        posteriors = model.infer(tokenized)
        winner = max(posteriors, key=posteriors.get)
        if posteriors[winner] < CONF_THRESHOLD:
            results.append({"return_id": row.return_id, "category": "UNKNOWN", "score": posteriors[winner], "source": "low_conf"})
        else:
            results.append({"return_id": row.return_id, "category": winner, "score": posteriors[winner], "source": "model"})
    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Annotate return reasons using hybrid rule/model pipeline")
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--start_date", default="2026-01-01")
    parser.add_argument("--dry-run", action="store_true")
    opts = parser.parse_args()

    engine = create_engine(opts.dsn)
    query = text("SELECT return_id, comment, verified_category, created_at FROM returns.return_requests "
                 "WHERE created_at >= :start_date")
    raw = pd.read_sql(query, engine, params={"start_date": opts.start_date})
    labeled = raw[raw.verified_category.isin(CATEGORIES)]
    tokens = [extract_tokens(preprocess(c)) for c in labeled.comment.tolist()]
    classifier = DirichletClassifier().train(tokens, labeled.verified_category.tolist())
    logger.info("Trained model on %d entries; vocabulary size %d", len(labeled), len(classifier.feature_set))

    unlabeled = raw[raw.verified_category.isna()]
    annotations = label_requests(unlabeled, classifier)
    logger.info("Prediction distribution:\n%s", annotations.category.value_counts().to_string())
    if not opts.dry_run:
        annotations.to_sql("reason_annotations", engine, schema="returns", if_exists="append", index=False)


if __name__ == "__main__":
    main()
