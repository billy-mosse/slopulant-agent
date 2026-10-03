import argparse
import logging
from datetime import date, timedelta
from math import floor

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("clearance")

INPUT_STOCK = "inventory.stock_levels"
INPUT_ELAST = "pricing.elasticities"
OUTPUT_TABLE = "pricing.promo_schedule"

# Discount tiers (cumulative % off), enforced to be non-decreasing over time
TIER_SCALE = np.array([0, 0.05, 0.12, 0.22, 0.35, 0.50])
BUCKET_COUNT = 32
WEEKLY_STORAGE_COST = 0.0038
LIQUIDATION_RECOVERY = 0.12
EXCESS_PENALTY = 0.30
DEFAULT_ELASTICITY = -2.1


def fetch_data(connection):
    inventory = pd.read_sql(
        """
        SELECT sku, category, units_on_hand AS stock,
               regular_price AS list_price, weekly_vol AS base_sales,
               season_close AS deadline
        FROM {table}
        WHERE season_close > CURRENT_DATE
        """.format(table=INPUT_STOCK),
        connection,
    )
    cat_elasticities = pd.read_sql(
        f"SELECT category, sensitivity AS elasticity FROM {INPUT_ELAST}", connection
    )
    return inventory, cat_elasticities


def qualify_candidates(frame: pd.DataFrame, reference_date: date) -> pd.DataFrame:
    days_remaining = (pd.to_datetime(frame["deadline"]) - pd.Timestamp(reference_date)).dt.days
    weeks = (days_remaining // 7).clip(lower=0)
    frame = frame.assign(weeks_left=weeks)
    expected = frame["base_sales"] * frame["weeks_left"]
    mask = (frame["weeks_left"] > 0) & (frame["stock"] > expected)
    return frame[mask].copy()


def forecast_sales(base_volume: float, discount_pct: float, curve: float) -> float:
    # Elasticity-based demand adjustment: sales scale with price ratio raised to sensitivity
    ratio = 1.0 - discount_pct
    if ratio <= 0:
        return 0.0
    return max(0.0, base_volume * (ratio ** curve))


def compute_policy(
    stock_qty: int, price: float, base_vol: float, sensitivity: float, weeks: int
):
    bucket_width = max(stock_qty / BUCKET_COUNT, 1)
    levels = len(TIER_SCALE)
    grid_size = min(floor(stock_qty / bucket_width) + 2, 100)
    value_mat = np.zeros((weeks + 1, grid_size, levels))
    decision_mat = np.zeros((weeks, grid_size, levels), dtype=np.int32)

    # terminal value: leftover inventory liquidated, penalized if unsold
    for b in range(grid_size):
        leftover = b * bucket_width
        value_mat[weeks, b, :] = leftover * price * (LIQUIDATION_RECOVERY - EXCESS_PENALTY)

    for week in reversed(range(weeks)):
        for s in range(grid_size):
            inventory = s * bucket_width
            for t in range(levels):
                best_val, best_next = -1e12, t
                for nxt in range(t, levels):
                    disc = TIER_SCALE[nxt]
                    pred_sold = forecast_sales(base_vol, disc, sensitivity)
                    actual_sold = min(pred_sold, inventory)
                    leftover_inv = inventory - actual_sold
                    gain = actual_sold * price * (1 - disc)
                    cost = leftover_inv * price * WEEKLY_STORAGE_COST
                    nxt_bucket = min(floor(leftover_inv / bucket_width), grid_size - 1)
                    fut_val = value_mat[week + 1, nxt_bucket, nxt]
                    total = gain - cost + fut_val
                    if total > best_val:
                        best_val, best_next = total, nxt
                value_mat[week, s, t] = best_val
                decision_mat[week, s, t] = best_next

    # reconstruct trajectory
    path, path_rows = [], []
    stock, tier = float(stock_qty), 0
    for w in range(weeks):
        bucket = min(floor(stock / bucket_width), grid_size - 1)
        chosen = decision_mat[w, bucket, tier]
        pred = forecast_sales(base_vol, TIER_SCALE[chosen], sensitivity)
        sold = min(pred, stock)
        stock -= sold
        path_rows.append(
            (w, TIER_SCALE[chosen], round(sold, 2), round(stock, 2))
        )
        tier = chosen
        path.append(TIER_SCALE[chosen])

    start_bucket = min(floor(stock_qty / bucket_width), grid_size - 1)
    return path, path_rows, value_mat[0, start_bucket, 0]


def assemble_schedule(candidates: pd.DataFrame, elasticities: pd.DataFrame, ref: date):
    map_el = dict(zip(elasticities["category"], elasticities["elasticity"]))
    records = []

    for idx, row in candidates.iterrows():
        cat = row["category"]
        cat_el = map_el.get(cat, DEFAULT_ELASTICITY)
        cat_el = min(cat_el, -0.15)  # cap at reasonable negativity
        if cat_el > -0.1:
            cat_el = DEFAULT_ELASTICITY
            log.warning("sku %s elasticity %.2f replaced with %.1f", row["sku"], cat_el, DEFAULT_ELASTICITY)

        ladder, timeline, val = compute_policy(
            int(row["stock"]),
            float(row["list_price"]),
            float(row["base_sales"]),
            cat_el,
            int(row["weeks_left"]),
        )

        for wk, pct, qty, rem in timeline:
            records.append({
                "item_id": row["sku"],
                "week_start": (ref + timedelta(weeks=wk)).isoformat(),
                "promotion_depth": int(pct * 100),
                "revised_price": round(float(row["list_price"]) * (1 - pct), 2),
                "forecast_units": round(qty, 1),
                "remainder_units": round(rem, 1),
                "elasticity_assumed": cat_el,
                "total_contrib": round(val, 2),
            })

        log.info(
            "SKU %s: path %s, final inventory %d",
            row["sku"],
            "-".join(str(int(x * 100)) for x in ladder),
            int(timeline[-1][3]) if timeline else 0,
        )

    return pd.DataFrame(records)


def run_pipeline(dsn: str, as_of: date, dry_run: bool):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    engine = create_engine(dsn)
    inv, elast = fetch_data(engine)
    pool = qualify_candidates(inv, as_of)
    log.info("identified %d candidates from %d SKUs", len(pool), len(inv))

    plan = assemble_schedule(pool, elast, as_of)
    if dry_run or plan.empty:
        print(plan.head(40).to_string())
        return
    plan.to_sql(
        OUTPUT_TABLE.split(".")[1],
        engine,
        schema=OUTPUT_TABLE.split(".")[0],
        if_exists="replace",
        index=False,
    )
    log.info("persisted %d decisions into %s", len(plan), OUTPUT_TABLE)


if __name__ == "__main__":
    p = argparse.ArgumentParser("inventory clearance planner")
    p.add_argument("--db", required=True)
    p.add_argument("--as-of", type=date.fromisoformat, default=date.today().isoformat())
    p.add_argument("--dry", action="store_true")
    run_pipeline(**vars(p.parse_args()))
