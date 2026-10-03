import argparse
import logging
import math
from collections import defaultdict
from datetime import datetime, timedelta
from itertools import islice

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger("product_velocity")

ACTION_SCORES = {"view": 1.0, "cart": 3.5, "buy": 9.0}
DECAY_TAU_H = 8.0
WINDOW_H = 48
HIST_WINDOW_DAYS = 28
TOP_N = 50
HIST_MIN_COUNT = 25
RECENT_MIN_COUNT = 6
STOCK_MIN = 10

RAW_EVENTS_Q = """
SELECT product_id, cat_id, action, ts
FROM events.raw_events
WHERE action IN ('view', 'cart') AND ts >= %(start)s
"""

PURCHASES_Q = """
SELECT product_id, cat_id, 'buy' AS action, created_at AS ts
FROM orders.items
WHERE created_at >= %(start)s AND status != 'canceled'
"""


def ingest(db, cutoff):
    start = cutoff - timedelta(days=HIST_WINDOW_DAYS)
    with db.connect() as conn:
        raw = pd.read_sql(text(RAW_EVENTS_Q), conn, params={"start": start})
        buys = pd.read_sql(text(PURCHASES_Q), conn, params={"start": start})
    data = pd.concat([raw, buys], sort=False)
    data.ts = pd.to_datetime(data.ts)
    data["score"] = data.action.map(ACTION_SCORES)
    return data


def compute_scores(df, now):
    cutoff = now - timedelta(hours=WINDOW_H)
    hours_old = (now - df.ts).dt.total_seconds() / 3600.0
    df = df.assign(
        weight=df.score * np.exp(-hours_old / DECAY_TAU_H),
        active= df.ts >= cutoff
    )

    # historical baseline: 24h buckets over past 26 days
    hist = df[~df.active]
    bucket_size = timedelta(hours=WINDOW_H).total_seconds()
    hist["bucket"] = ((cutoff - hist.ts).dt.total_seconds() // bucket_size).astype(int)
    bucket_sums = hist.groupby(["product_id", "bucket"]).score.sum()
    stats = defaultdict(lambda: [0.0, 0.0, 0])
    for (pid, _), val in bucket_sums.items():
        s = stats[pid]
        s[0] += val
        s[1] += val * val
        s[2] += 1

    # recent activity
    recent = df[df.active].groupby(["product_id", "cat_id"]).agg(
        activity=("weight", "sum"),
        count=("score", "size")
    ).reset_index()

    # compute per-product baseline stats
    n_buckets = (HIST_WINDOW_DAYS * 24 - WINDOW_H) / WINDOW_H
    norm_factor = DECAY_TAU_H / math.log(2) * (1 - math.exp(-WINDOW_H / DECAY_TAU_H))

    rows = []
    for _, row in recent.iterrows():
        pid, cat = row["product_id"], row["cat_id"]
        act = row["activity"] / norm_factor
        mu, var, cnt = stats[pid]
        mean = mu / n_buckets if n_buckets > 0 else 0.0
        var = max(var / n_buckets - mean * mean, 0.0)
        std = math.sqrt(var + mean + 1.0)
        z = (act - mean) / std if std > 0 else 0.0
        rows.append([pid, cat, act, mean, z, int(row["count"]), int(hist.groupby("product_id").size().get(pid, 0))])

    return pd.DataFrame(rows, columns=["sku", "category_id", "score", "baseline", "z", "recent_count", "hist_count"])


def filter_candidates(df, inv):
    mask = (df.recent_count >= RECENT_MIN_COUNT) & (df.hist_count >= HIST_MIN_COUNT) & (df.z > 0)
    if inv is not None:
        mask &= df.sku.map(inv).fillna(0) >= STOCK_MIN
    logger.info("retained %d of %d candidates", mask.sum(), len(df))
    return df[mask]


def rank_by_cat(df, limit=TOP_N):
    results = []
    for cat, group in df.sort_values("z", ascending=False).groupby("category_id", sort=False):
        for i, (_, r) in enumerate(islice(group.iterrows(), limit), 1):
            results.append({
                "category_id": cat,
                "sku": r["sku"],
                "rank": i,
                "velocity_z": round(r["z"], 3),
                "score": round(r["score"], 2),
                "baseline": round(r["baseline"], 2)
            })
    return pd.DataFrame(results)


def run(dsn, inv_path=None, now_str=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--inventory", help="inventory snapshot parquet")
    parser.add_argument("--timestamp")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    now = datetime.fromisoformat(args.timestamp) if args.timestamp else datetime.utcnow().replace(minute=0, second=0, microsecond=0)

    engine = create_engine(args.dsn)
    inv = pd.read_parquet(args.inventory).set_index("sku").units if args.inventory else None

    df = compute_scores(ingest(engine, now), now)
    filtered = filter_candidates(df, inv)
    ranked = rank_by_cat(filtered)
    ranked["generated_at"] = now
    ranked.to_sql("velocity", engine, schema="recommendations", if_exists="append", index=False)
    logger.info("wrote %d records across %d categories", len(ranked), ranked.category_id.nunique())


if __name__ == "__main__":
    run()
