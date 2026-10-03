"""
Provides a structured approach to extracting aspect-specific sentiment from
customer feedback. For each approved review, the pipeline identifies key
product characteristics (e.g., softness, durability) and computes a normalized
sentiment score by aggregating context-aware polarity judgments within a local
token window. Aggregates are published per product SKU and aspect for downstream
dashboarding and recommendation systems.

Designed for maintainability and clarity, this module follows an object-oriented
design that separates concerns: tokenization, contextual scoring, and aggregation
are all encapsulated within well-named helper classes.
"""
import argparse
import logging
import re
import unicodedata

import pandas as pd
from sqlalchemy import create_engine, text

from lexicon import (
    ASPECT_CATEGORIES,
    POLARITY_TERMS,
    NEGATION_EFFECT,
    NEGATION_WORDS,
    INTENSITY_ADJUSTMENTS,
    CLAUSE_BOUNDARIES,
)

logger = logging.getLogger("review_sentiment")

SENTIMENT_RADIUS = 5
NEGATION_CONTEXT = 3
TOKEN_PATTERN = re.compile(r"[a-z]+(?:'[a-z]+)?|[.,;!?]")
LONGEST_MULTIWORD = max(
    len(phrase.split()) for phrase in (
        list(POLARITY_TERMS)
        + list(INTENSITY_ADJUSTMENTS)
        + [term for terms in ASPECT_CATEGORIES.values() for term in terms]
    )
)


class TextProcessor:
    @staticmethod
    def clean(raw_text: str | None) -> str:
        if not raw_text:
            return ""
        normalized = unicodedata.normalize("NFKC", raw_text).lower()
        normalized = normalized.replace("’", "'").replace("‘", "'")
        normalized = re.sub(r"(.)\1{2,}", r"\1\1", normalized)
        return normalized

    @staticmethod
    def segment(input_text: str) -> list[str]:
        tokens = TOKEN_PATTERN.findall(input_text)
        result = []
        i = 0
        while i < len(tokens):
            for length in range(min(LONGEST_MULTIWORD, len(tokens) - i), 0, -1):
                candidate = " ".join(tokens[i : i + length])
                if candidate in POLARITY_TERMS or candidate in INTENSITY_ADJUSTMENTS or candidate in TOKEN_TO_ASPECT:
                    result.append(candidate)
                    i += length
                    break
            else:
                result.append(tokens[i])
                i += 1
        return result


class AspectScorer:
    def __init__(self, aspect_terms: dict[str, tuple[str, ...]]) -> None:
        self.aspect_terms = aspect_terms
        self.token_to_aspect = {
            term: aspect for aspect, terms in aspect_terms.items() for term in terms
        }

    def compute_sentiment(self, tokens: list[str]) -> list[tuple[str, float]]:
        scores = []
        for index, token in enumerate(tokens):
            aspect = self.token_to_aspect.get(token)
            if aspect is None:
                continue

            weight_sum, sentiment_sum = 0.0, 0.0
            start = max(0, index - SENTIMENT_RADIUS)
            end = min(len(tokens), index + SENTIMENT_RADIUS + 1)

            for j in range(start, end):
                if tokens[j] not in POLARITY_TERMS:
                    continue
                distance_weight = 1.0 / (1.0 + abs(index - j))
                sentiment_sum += distance_weight * self._local_polarity(tokens, j)
                weight_sum += distance_weight

            if weight_sum > 0:
                scores.append((aspect, sentiment_sum / weight_sum))

        return scores

    def _local_polarity(self, tokens: list[str], index: int) -> float:
        base = POLARITY_TERMS[tokens[index]]
        multiplier = 1.0
        for k in range(index - 1, max(-1, index - 1 - NEGATION_CONTEXT), -1):
            token = tokens[k]
            if token in CLAUSE_BOUNDARIES:
                break
            if token in INTENSITY_ADJUSTMENTS:
                multiplier *= INTENSITY_ADJUSTMENTS[token]
            elif token in NEGATION_WORDS:
                multiplier *= NEGATION_EFFECT
                break
        return max(-1.0, min(1.0, base * multiplier))


class SKUAspectAggregator:
    @staticmethod
    def build(reviews_frame: pd.DataFrame, scorer: AspectScorer) -> pd.DataFrame:
        records = []
        for _, row in reviews_frame.iterrows():
            content = TextProcessor.clean(f"{row.title or ''}. {row.body or ''}")
            aspects = scorer.compute_sentiment(TextProcessor.segment(content))
            for aspect, score in aspects:
                records.append({
                    "review_id": row.review_id,
                    "sku": row.sku,
                    "aspect": aspect,
                    "sentiment_score": score,
                })
        return pd.DataFrame(records)

    @staticmethod
    def summarize(mention_frame: pd.DataFrame, min_support: int = 3) -> pd.DataFrame:
        per_review = mention_frame.groupby(
            ["review_id", "sku", "aspect"], as_index=False
        )["sentiment_score"].mean()

        per_review["positive"] = per_review["sentiment_score"] > 0.15
        per_review["negative"] = per_review["sentiment_score"] < -0.15

        aggregated = (
            per_review.groupby(["sku", "aspect"], as_index=False)
            .agg(
                mention_count=("review_id", "nunique"),
                average_sentiment=("sentiment_score", "mean"),
                positive_ratio=("positive", "mean"),
                negative_ratio=("negative", "mean"),
            )
            .query("mention_count >= @min_support")
        )
        return aggregated


class SentimentPipeline:
    def __init__(self, engine_url: str, minimum_mentions: int) -> None:
        self.engine = create_engine(engine_url)
        self.minimum_mentions = minimum_mentions

    def run(self) -> None:
        reviews = pd.read_sql(
            text("SELECT review_id, sku, title, body FROM reviews.approved"),
            self.engine,
        )

        scorer = AspectScorer(ASPECT_CATEGORIES)
        mentions = SKUAspectAggregator.build(reviews, scorer)
        logger.info(
            "Processed %d reviews into %d aspect-sentiment observations",
            len(reviews),
            len(mentions),
        )

        summary = SKUAspectAggregator.summarize(mentions, self.minimum_mentions)
        summary.to_sql(
            "aspect_sentiment",
            self.engine,
            schema="reviews",
            if_exists="replace",
            index=False,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute aspect-level sentiment aggregates from approved reviews."
    )
    parser.add_argument("--dsn", required=True, help="Database connection string")
    parser.add_argument(
        "--min-mentions",
        type=int,
        default=3,
        help="Minimum number of mentions required per aspect",
    )
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()
    pipeline = SentimentPipeline(args.dsn, args.min_mentions)
    pipeline.run()


if __name__ == "__main__":
    main()
