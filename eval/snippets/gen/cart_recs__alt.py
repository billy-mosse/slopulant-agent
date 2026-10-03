import argparse
import logging
from collections import Counter
from itertools import combinations

import pandas as pd
from sqlalchemy import create_engine

logger = logging.getLogger("order_suggestions")

SUPPORT_THRESHOLD = 12
CONFIDENCE_MIN = 0.025
LIFT_MIN = 1.4
ITEM_COVERAGE_CUTOFF = 35
MAX_BASKET_SIZE = 28
RULES_PER_ANT = 18

QUERY_BASKETS = """
SELECT order_id, product_id, group_id
FROM transactions.line_items
WHERE created_at >= now() - interval '1 year'
  AND status NOT IN ('voided', 'refunded')
"""


def extract_sequences(df, field):
    grouped = df.groupby("order_id")[field].apply(lambda x: sorted(set(x)))
    return [seq for seq in grouped if 1 < len(seq) <= MAX_BASKET_SIZE]


def discover_patterns(seqs):
    total = len(seqs)
    freq_single = Counter(item for seq in seqs for item in seq)
    freq_pair = Counter(pair for seq in seqs for pair in combinations(seq, 2))
    results = []
    for (x, y), cnt in freq_pair.items():
        if cnt < SUPPORT_THRESHOLD:
            continue
        for a, b in ((x, y), (y, x)):
            conf = cnt / freq_single[a]
            lift = conf / (freq_single[b] / total)
            if conf >= CONFIDENCE_MIN and lift >= LIFT_MIN:
                results.append((a, b, cnt / total, conf, lift))
    logger.info("processed %d baskets, %d frequent pairs, %d rules generated", total, len(freq_pair), len(results))
    return pd.DataFrame(results, columns=["source", "target", "support", "confidence", "lift"])


def fallback_to_group(df, item_rules, group_rules, prod_group):
    item_freq = df.groupby("product_id").order_id.nunique()
    under_represented = item_freq[item_freq < ITEM_COVERAGE_CUTOFF].index
    top_by_group = (df.groupby(["group_id", "product_id"]).order_id.nunique()
                    .reset_index()
                    .sort_values("order_id", ascending=False)
                    .groupby("group_id").head(4)
                    .groupby("group_id").product_id.apply(list).to_dict())
    group_by_source = group_rules.groupby("source")
    rows = []
    for pid in under_represented:
        gid = prod_group.get(pid)
        if gid not in group_by_source.groups:
            continue
        for rule in group_by_source.get_group(gid).itertuples():
            for alt in top_by_group.get(rule.target, []):
                rows.append((pid, alt, rule.support, rule.confidence, rule.lift * 0.75))
    logger.info("expanded %d sparse items with %d group-based suggestions", len(under_represented), len(rows))
    fallback_df = pd.DataFrame(rows, columns=item_rules.columns).assign(source_type="group")
    return pd.concat([item_rules.assign(source_type="product"), fallback_df], ignore_index=True)


def filter_duplicates(rules, prod_group):
    same_group = rules.source.map(prod_group) == rules.target.map(prod_group)
    logger.info("removed %d rules where source and target share group", same_group.sum())
    return rules[~same_group]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-url", required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    engine = create_engine(args.db_url)

    df = pd.read_sql(QUERY_BASKETS, engine)
    prod_to_group = df.drop_duplicates("product_id").set_index("product_id").group_id.to_dict()
    item_patterns = discover_patterns(extract_sequences(df, "product_id"))
    group_patterns = discover_patterns(extract_sequences(df, "group_id"))
    group_patterns = group_patterns[group_patterns.source != group_patterns.target]

    merged = filter_duplicates(fallback_to_group(df, item_patterns, group_patterns, prod_to_group), prod_to_group)
    ranked = (merged.sort_values("lift", ascending=False)
              .drop_duplicates(["source", "target"])
              .groupby("source").head(RULES_PER_ANT))
    ranked["generated_at"] = pd.Timestamp.utcnow()
    ranked.to_sql("cart_suggestions", engine, schema="ml", if_exists="replace", index=False)
    logger.info("stored %d rules in ml.cart_suggestions", len(ranked))


if __name__ == "__main__":
    main()
