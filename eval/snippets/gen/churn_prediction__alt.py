import logging
import numpy as np
import pandas as pd
import sqlalchemy as sa
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score

log = logging.getLogger("churn")

ORDERS_QUERY = """
SELECT customer_id, order_id, order_ts, category_l1, quantity, net_revenue
FROM orders.lines
WHERE order_ts >= %(start)s AND order_ts < %(cutoff)s
  AND status NOT IN ('cancelled', 'fraud')
"""

SESSIONS_QUERY = """
SELECT customer_id, session_id, event_ts, event_type
FROM sessions.events
WHERE event_ts >= %(start)s AND event_ts < %(cutoff)s
  AND customer_id IS NOT NULL
"""

DECAY_HALFLIFE = 120.0
CORE_CATS = ["bedding", "bath", "decor", "kitchen", "rugs", "lighting", "furniture"]
LOOKBACK_WINDOW = 730


def construct_features(conn, anchor_date: pd.Timestamp) -> pd.DataFrame:
    """Generate customer-level behavioral features as of anchor_date."""
    start = anchor_date - pd.Timedelta(days=LOOKBACK_WINDOW)
    params = {"start": start, "cutoff": anchor_date}

    log.info(f"fetching order lines from {start.date()} to {anchor_date.date()}")
    orders = pd.read_sql(ORDERS_QUERY, conn, params=params, parse_dates=["order_ts"])
    log.info(f"acquired {len(orders):,} order lines for {orders.customer_id.nunique():,} customers")

    # aggregate per-order stats
    per_order = orders.groupby(["customer_id", "order_id"], as_index=False).agg(
        order_time=("order_ts", "min"),
        spend=("net_revenue", "sum"),
        item_count=("quantity", "sum")
    )

    # temporal RFM-style features
    rfm = per_order.groupby("customer_id").agg(
        latest_order=("order_time", "max"),
        earliest_order=("order_time", "min"),
        order_count=("order_id", "nunique"),
        total_spend=("spend", "sum"),
        avg_spend=("spend", "mean"),
        avg_items=("item_count", "mean")
    )
    rfm["days_since_last"] = (anchor_date - rfm["latest_order"]).dt.total_seconds() / 86400.0
    rfm["tenure_days"] = (anchor_date - rfm["earliest_order"]).dt.total_seconds() / 86400.0
    rfm["log_days_since_last"] = np.log1p(rfm["days_since_last"])
    rfm["log_order_count"] = np.log1p(rfm["order_count"])
    rfm["log_total_spend"] = np.log1p(rfm["total_spend"].clip(0))
    rfm["log_avg_spend"] = np.log1p(rfm["avg_spend"].clip(0))
    rfm["orders_per_month"] = rfm["order_count"] / np.maximum(rfm["tenure_days"] / 30.0, 1.0)

    # inter-order intervals
    per_order = per_order.sort_values(["customer_id", "order_time"])
    per_order["interval_days"] = per_order.groupby("customer_id")["order_time"].diff().dt.total_seconds() / 86400.0
    interval_stats = per_order.groupby("customer_id")["interval_days"].agg(
        avg_interval=("mean",),
        std_interval=("std",)
    ).rename(columns=lambda x: x[0])
    rfm = rfm.join(interval_stats)
    rfm["recency_ratio"] = rfm["days_since_last"] / np.maximum(rfm["avg_interval"].fillna(rfm["days_since_last"]), 1.0)

    # category decay-weighted mix
    age = (anchor_date - orders["order_ts"]).dt.total_seconds() / 86400.0
    decay = np.power(0.5, age / DECAY_HALFLIFE) * orders["net_revenue"].clip(0)
    orders["cat_group"] = orders["category_l1"].where(orders["category_l1"].isin(CORE_CATS), "misc")
    cat_weights = orders.groupby(["customer_id", "cat_group"])["decay"].sum().unstack(fill_value=0.0)
    cat_totals = cat_weights.sum(axis=1).replace(0, np.nan)
    cat_props = cat_weights.div(cat_totals, axis=0).fillna(0.0)
    probs = cat_props.to_numpy()
    cat_props["cat_entropy"] = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    cat_props["decay_spend"] = cat_totals.fillna(0.0)
    cat_props = cat_props.add_prefix("cat_prop_")

    # session activity
    log.info("loading session events")
    events = pd.read_sql(SESSIONS_QUERY, conn, params=params, parse_dates=["event_ts"])
    sessions = events.groupby(["customer_id", "session_id"], as_index=False).agg(
        session_start=("event_ts", "min"),
        event_count=("event_type", "size"),
        cart_adds=("event_type", lambda x: (x == "add_to_cart").sum())
    )
    sessions["days_ago"] = (anchor_date - sessions["session_start"]).dt.total_seconds() / 86400.0

    sess_agg = sessions.groupby("customer_id").agg(
        recency=("days_ago", "min"),
        total_sessions=("session_id", "nunique"),
        events_per_sess=("event_count", "mean"),
        total_cart_adds=("cart_adds", "sum")
    )
    for w in (7, 30, 90):
        sess_agg[f"sessions_last_{w}d"] = sessions[sessions["days_ago"] <= w].groupby("customer_id")["session_id"].nunique()
    sess_agg = sess_agg.fillna({f"sessions_last_{w}d": 0 for w in (7, 30, 90)})
    sess_agg["visit_stability"] = (sess_agg["sessions_last_30d"] + 1) / (sess_agg["sessions_last_90d"] / 3 + 1)
    sess_agg["log_recency"] = np.log1p(sess_agg["recency"])

    # combine all feature blocks
    features = rfm.join(cat_props, how="left").join(sess_agg, how="left")
    features["recency"] = features["recency"].fillna(LOOKBACK_WINDOW)
    features["log_recency"] = features["log_recency"].fillna(np.log1p(LOOKBACK_WINDOW))
    features = features.fillna(0.0).drop(columns=["latest_order", "earliest_order"])
    log.info(f"feature matrix: {features.shape[0]:,} customers × {features.shape[1]} features")
    return features


def generate_labels(conn, anchor_date: pd.Timestamp, horizon: int = 90) -> pd.Index:
    """Identify customers who placed orders within [anchor_date, anchor_date + horizon)."""
    end = anchor_date + pd.Timedelta(days=horizon)
    sql = "SELECT DISTINCT customer_id FROM orders.lines WHERE order_ts >= %(start)s AND order_ts < %(cutoff)s"
    active = pd.read_sql(sql, conn, params={"start": anchor_date, "cutoff": end})
    log.info(f"{len(active):,} customers active in {horizon}d post-anchor window")
    return pd.Index(active["customer_id"].unique(), name="customer_id")


def train_and_score(conn, anchor_date: pd.Timestamp, dry_run: bool = False):
    """Train churn model and output calibrated probabilities."""
    train_date = anchor_date - pd.Timedelta(days=2 * 90 + 30)
    calib_date = anchor_date - pd.Timedelta(days=90)

    train_df = construct_features(conn, train_date)
    calib_df = construct_features(conn, calib_date)
    score_df = construct_features(conn, anchor_date)

    train_labels = generate_labels(conn, train_date)
    calib_labels = generate_labels(conn, calib_date)

    train_df["churned"] = ~train_df.index.isin(train_labels)
    calib_df["churned"] = ~calib_df.index.isin(calib_labels)

    cols = [c for c in train_df.columns if c != "churned"]
    calib_df = calib_df.reindex(columns=cols + ["churned"], fill_value=0.0)
    score_df = score_df.reindex(columns=cols, fill_value=0.0)

    base = LGBMClassifier(
        n_estimators=500, learning_rate=0.04, max_depth=6, num_leaves=31,
        min_child_samples=200, reg_alpha=0.5, reg_lambda=1.0, random_state=42,
        n_jobs=-1, verbose=-1
    )
    calibrator = CalibratedClassifierCV(base, method="isotonic", cv="prefit")
    calibrator.fit(calib_df[cols], calib_df["churned"].astype(int))

    raw_calib = calibrator.predict_proba(calib_df[cols])[:, 1]
    log.info(
        f"calibration AUC: {roc_auc_score(calib_df['churned'], raw_calib):.4f}, "
        f"Brier: {brier_score_loss(calib_df['churned'], raw_calib):.4f}"
    )

    probs = calibrator.predict_proba(score_df[cols])[:, 1]
    output = pd.DataFrame({
        "customer_id": score_df.index,
        "churn_prob": np.round(probs, 5),
        "horizon_days": 90,
        "score_date": anchor_date.date(),
        "model_version": "churn_lgbm_v4"
    })
    decile_means = output.groupby(pd.qcut(output.churn_prob.rank(method="first"), 10, labels=False))["churn_prob"].mean()
    log.info(f"scored {len(output):,} customers; mean churn_prob {output.churn_prob.mean():.3f}; decile means {decile_means.round(3).tolist()}")

    if dry_run:
        log.info("dry run: skipping write")
        return

    schema, tbl = "scores", "p_churn"
    output.to_sql(tbl, conn, schema=schema, if_exists="append", index=False, chunksize=50_000)
    log.info(f"wrote {len(output):,} rows to {schema}.{tbl}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--as-of", required=True, help="YYYY-MM-DD")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    engine = sa.create_engine(args.dsn)
    anchor = pd.Timestamp(args.as_of)

    with engine.connect() as conn:
        train_and_score(conn, anchor, args.dry_run)


if __name__ == "__main__":
    main()
