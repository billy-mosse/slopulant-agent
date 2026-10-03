"""Module for generating and serving personalized 'complete your order' recommendations.

This component identifies high-value item pairings observed in historical order data,
then serves ranked suggestions when a customer adds items to their cart. It prioritizes
rules with strong statistical support (minimum co-occurrence frequency), predictive
reliability (confidence), and meaningful lift over baseline popularity. For items
with limited transaction history, it gracefully falls back to category-level patterns
and top-selling alternatives within the same category.

Rules are filtered to exclude substitutable items (same category) and capped at a
per-antecedent limit to ensure diversity and relevance.
"""
import argparse
import logging
from collections import defaultdict
from itertools import combinations

import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger("recommendation_engine")

# Thresholds for rule quality and coverage
CO_OCCURRENCE_MIN = 15
CONFIDENCE_MIN = 0.02
LIFT_MIN = 1.5
ITEM_ACTIVITY_THRESHOLD = 40
MAX_BASKET_SIZE = 30
RULES_PER_ANTICIPATED_ITEM = 20

ORDER_HISTORY_QUERY = """
SELECT order_id, sku, category_id
FROM orders.lines
WHERE order_date >= CURRENT_DATE - INTERVAL '365 days'
  AND status NOT IN ('cancelled', 'returned')
"""


def extract_baskets(dataframe, grouping_field):
    """Aggregate orders into item sets, deduplicated and size-constrained."""
    grouped = dataframe.groupby("order_id")[grouping_field].apply(lambda s: sorted(set(s)))
    return [basket for basket in grouped if 1 < len(basket) <= MAX_BASKET_SIZE]


def discover_associations(baskets_list):
    """Compute association metrics for item pairs meeting minimum support."""
    total_baskets = len(baskets_list)
    item_frequencies = Counter(item for basket in baskets_list for item in basket)
    pair_counts = Counter(pair for basket in baskets_list for pair in combinations(basket, 2))

    valid_rules = []
    for (first, second), co_occurrence in pair_counts.items():
        if co_occurrence < CO_OCCURRENCE_MIN:
            continue

        for antecedent, consequent in ((first, second), (second, first)):
            confidence = co_occurrence / item_frequencies[antecedent]
            baseline_consequent_rate = item_frequencies[consequent] / total_baskets
            lift = confidence / baseline_consequent_rate if baseline_consequent_rate > 0 else 0

            if confidence >= CONFIDENCE_MIN and lift >= LIFT_MIN:
                support = co_occurrence / total_baskets
                valid_rules.append((antecedent, consequent, support, confidence, lift))

    logger.info(
        "Processed %d baskets; identified %d candidate pairs, %d rules retained",
        total_baskets,
        sum(v >= CO_OCCURRENCE_MIN for v in pair_counts.values()),
        len(valid_rules),
    )
    return pd.DataFrame(valid_rules, columns=["antecedent", "consequent", "support", "confidence", "lift"])


def expand_sparse_coverage(raw_data, item_level_rules, category_level_rules, sku_to_category):
    """Supplement item-level rules with category-level fallbacks for low-activity SKUs."""
    item_activity = raw_data.groupby("sku").order_id.nunique()
    underrepresented_skus = item_activity[item_activity < ITEM_ACTIVITY_THRESHOLD].index

    top_sellers_by_category = (
        raw_data.groupby(["category_id", "sku"]).order_id.nunique()
        .reset_index()
        .sort_values("order_id", ascending=False)
        .groupby("category_id").head(3)
        .groupby("category_id").sku.apply(list).to_dict()
    )

    fallback_rules = []
    for sku in underrepresented_skus:
        category = sku_to_category.get(sku)
        if category not in category_level_rules.antecedent.values:
            continue

        for rule in category_level_rules[category_level_rules.antecedent == category].itertuples():
            for candidate in top_sellers_by_category.get(rule.consequent, []):
                fallback_rules.append((sku, candidate, rule.support, rule.confidence, rule.lift * 0.8))

    logger.info("Generated %d fallback rules for %d sparse items", len(fallback_rules), len(underrepresented_skus))
    fallback_df = pd.DataFrame(fallback_rules, columns=item_level_rules.columns).assign(level="category")
    return pd.concat([item_level_rules.assign(level="item"), fallback_df], ignore_index=True)


def eliminate_substitutes(rules_dataframe, sku_category_map):
    """Remove rules where antecedent and consequent belong to the same category."""
    same_category_mask = (
        rules_dataframe.antecedent.map(sku_category_map) == rules_dataframe.consequent.map(sku_category_map)
    )
    logger.info("Filtered out %d same-category substitute rules", same_category_mask.sum())
    return rules_dataframe[~same_category_mask]


class OrderCompletionEngine:
    """Serves personalized cart add-on suggestions using precomputed association rules."""

    def __init__(self, rule_set: pd.DataFrame, sku_category_mapping: dict):
        self.sku_category_map = sku_category_mapping
        self.indexed_rules = defaultdict(list)
        for record in rule_set.itertuples():
            self.indexed_rules[record.antecedent].append((record.consequent, record.lift, record.level))

    @classmethod
    def load_from_database(cls, connection, sku_category_mapping):
        """Initialize engine by fetching rules from persistent storage."""
        raw_rules = pd.read_sql(
            "SELECT antecedent, consequent, lift, level FROM recs.cart_addons", connection
        )
        return cls(raw_rules, sku_category_mapping)

    def suggest_add_ons(self, current_cart, limit=4):
        """Produce ranked suggestions for items to add to the given cart."""
        unique_cart_items = list(dict.fromkeys(current_cart))
        cart_categories = {self.sku_category_map.get(sku) for sku in unique_cart_items}
        scores = defaultdict(float)
        attribution = defaultdict(list)

        for sku in unique_cart_items:
            for candidate, lift_value, rule_level in self.indexed_rules.get(sku, []):
                if candidate in current_cart or self.sku_category_map.get(candidate) in cart_categories:
                    continue
                weight = lift_value if rule_level == "item" else 0.5 * lift_value
                scores[candidate] += weight
                attribution[candidate].append(sku)

        ranked_candidates = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
        final_selection = []
        selected_categories = set()

        for sku, score in ranked_candidates:
            category = self.sku_category_map.get(sku)
            if category in selected_categories:
                continue
            selected_categories.add(category)
            final_selection.append({
                "sku": sku,
                "score": round(score, 3),
                "because": attribution[sku],
            })
            if len(final_selection) == limit:
                break

        return final_selection


def main():
    parser = argparse.ArgumentParser(description="Generate and deploy cart completion rules")
    parser.add_argument("--dsn", required=True, help="Database connection string")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    engine = create_engine(args.dsn)

    historical_data = pd.read_sql(ORDER_HISTORY_QUERY, engine)
    sku_category_map = historical_data.drop_duplicates("sku").set_index("sku").category_id.to_dict()

    item_rules = discover_associations(extract_baskets(historical_data, "sku"))
    category_rules = discover_associations(extract_baskets(historical_data, "category_id"))
    category_rules = category_rules[category_rules.antecedent != category_rules.consequent]

    consolidated_rules = eliminate_substitutes(
        expand_sparse_coverage(historical_data, item_rules, category_rules, sku_category_map),
        sku_category_map,
    )

    final_rules = (
        consolidated_rules.sort_values("lift", ascending=False)
        .drop_duplicates(["antecedent", "consequent"])
        .groupby("antecedent").head(RULES_PER_ANTICIPATED_ITEM)
    )
    final_rules["generated_at"] = pd.Timestamp.utcnow()

    final_rules.to_sql("cart_addons", engine, schema="recs", if_exists="replace", index=False)
    logger.info("Persisted %d rules to recs.cart_addons", len(final_rules))


if __name__ == "__main__":
    main()
