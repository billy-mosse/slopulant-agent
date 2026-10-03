import argparse
import logging
import pandas as pd
import numpy as np
import sqlalchemy as sqla
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("recommendations_engine")

CATALOG_QUERY = """
SELECT product_id, title, description, category_l1, is_active, in_stock
FROM catalog.products
"""

RECENT_ORDERS_QUERY = """
SELECT customer_id, product_id, order_ts
FROM orders.lines
WHERE order_ts >= %(cutoff)s AND status = 'fulfilled'
"""

TARGET_TABLE = "marketing.email_recs"
MAX_RECS = 6
RECENT_WINDOW_DAYS = 30
HISTORY_WINDOW_DAYS = 730
MIN_COSINE = 0.08
CHUNK_SIZE = 2048


def preprocess_text(series: pd.Series) -> pd.Series:
    cleaned = series.fillna("").str.lower()
    cleaned = cleaned.str.replace(r"<[^>]*>", " ", regex=True)
    cleaned = cleaned.str.replace(r"\b\d+\s*(?:tc|threadcount)\b", " threadcount ", regex=True)
    cleaned = cleaned.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    cleaned = cleaned.str.replace(r"\s+", " ", regex=True).str.strip()
    return cleaned


def execute_pipeline(engine, reference_date: pd.Timestamp, dry: bool) -> pd.DataFrame:
    items = pd.read_sql(CATALOG_QUERY, engine)
    logger.info(f"loaded {len(items):,} products ({items.is_active.sum():,} active)")

    items["content"] = preprocess_text(items["title"]) + " " + preprocess_text(items["title"]) + " " + preprocess_text(items["description"])
    items = items.reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.6,
        sublinear_tf=True,
        stop_words="english"
    )
    tfidf_matrix = vectorizer.fit_transform(items["content"])
    tfidf_matrix = tfidf_matrix.tocsr()
    logger.info(f"built TF-IDF matrix: {tfidf_matrix.shape[0]:,} items × {tfidf_matrix.shape[1]:,} features")

    pid2idx = pd.Series(np.arange(len(items)), index=items.product_id)
    active_mask = (items["is_active"] & items["in_stock"]).values

    orders = pd.read_sql(RECENT_ORDERS_QUERY, engine, params={"cutoff": reference_date - pd.Timedelta(days=HISTORY_WINDOW_DAYS)})
    user_purchases = orders.groupby("customer_id")["product_id"].apply(set)

    recent_orders = orders[orders.order_ts >= reference_date - pd.Timedelta(days=RECENT_WINDOW_DAYS)]
    last_purchase = recent_orders.sort_values("order_ts").groupby("customer_id").tail(1)
    last_purchase = last_purchase.set_index("customer_id")[["product_id", "order_ts"]]
    last_purchase = last_purchase[last_purchase.product_id.isin(pid2idx.index)]

    logger.info(f"{len(last_purchase):,} customers made a purchase in last {RECENT_WINDOW_DAYS} days")

    unique_anchors = last_purchase.product_id.unique()
    candidate_map = {}

    for i in range(0, len(unique_anchors), CHUNK_SIZE):
        batch = unique_anchors[i:i + CHUNK_SIZE]
        idx_batch = pid2idx[batch].to_numpy()
        batch_vecs = tfidf_matrix[idx_batch]
        scores = cosine_similarity(batch_vecs, tfidf_matrix).toarray()
        scores[:, ~active_mask] = -np.inf
        np.fill_diagonal(scores, -np.inf)

        top_indices = np.argsort(-scores, axis=1)[:, :MAX_RECS * 5]
        for j, pid in enumerate(batch):
            valid = [idx for idx in top_indices[j] if scores[j, idx] >= MIN_COSINE]
            candidate_map[pid] = [(items.product_id.iloc[k], float(scores[j, k])) for k in valid]
        logger.info(f"processed anchors {i:,}–{i + len(batch):,}")

    output_rows = []
    for cust, row in last_purchase.iterrows():
        owned = user_purchases.get(cust, set())
        candidates = candidate_map.get(row.product_id, [])
        filtered = [(p, s) for p, s in candidates if p not in owned][:MAX_RECS]
        for rank, (pid, score) in enumerate(filtered, 1):
            output_rows.append((cust, row.product_id, pid, rank, round(score, 4)))

    result = pd.DataFrame(output_rows, columns=["customer_id", "seed_product_id", "candidate_product_id", "position", "score"])
    result["created_at"] = reference_date

    stats = result.groupby("customer_id").size()
    logger.info(f"generated {len(result):,} recommendations for {result.customer_id.nunique():,} users; "
                f"{(stats < MAX_RECS).sum():,} users received fewer than {MAX_RECS} items")

    category_counts = result.merge(items[["product_id", "category_l1"]], left_on="candidate_product_id", right_on="product_id") \
        .groupby("category_l1").size().sort_values(ascending=False).head(8)
    logger.info(f"top categories: {category_counts.to_dict()}")

    if dry:
        logger.info("dry mode: no database write")
    else:
        schema, tbl = TARGET_TABLE.split(".")
        result.to_sql(tbl, engine, schema=schema, if_exists="replace", index=False, chunksize=50_000)
        logger.info(f"wrote {len(result):,} rows to {TARGET_TABLE}")

    return result


def cli():
    parser = argparse.ArgumentParser(description="Generate post-purchase product recommendations via email")
    parser.add_argument("--connection-string", required=True, help="Database connection URI")
    parser.add_argument("--as-of", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    engine = sqla.create_engine(args.connection_string)
    execute_pipeline(engine, pd.Timestamp(args.as_of), args.dry_run)


if __name__ == "__main__":
    cli()
