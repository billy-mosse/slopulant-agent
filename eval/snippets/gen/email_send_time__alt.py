import argparse
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
import sqlalchemy as sql

logger = logging.getLogger("optimal_send")

TABLE_EVENTS = "marketing.email_events"
TABLE_OUTPUT = "marketing.send_times"
HOURS_PER_BUCKET = 4
NUM_BUCKETS = 7 * 24 // HOURS_PER_BUCKET
MIN_INDIVIDUAL_DATA = 6
PRIOR_WEIGHT_MAX = 40.0
SAMPLES_PER_ITER = 1
DATA_WINDOW_DAYS = 365
DEFAULT_ZONE = "America/Los_Angeles"


def execute(engine, reference: pd.Timestamp, seed: int, dry: bool) -> pd.DataFrame:
    rng = np.random.Generator(np.random.PCG64(seed))
    query = f"""
        SELECT user_id, campaign_id, send_time_utc, open_time_utc, user_timezone, cohort
        FROM {TABLE_EVENTS}
        WHERE event_kind = 'delivered' AND send_time_utc >= %(from_ts) AND send_time_utc < %(to_ts)
    """
    raw = pd.read_sql(
        query, engine,
        params={"from_ts": reference - timedelta(days=DATA_WINDOW_DAYS), "to_ts": reference},
        parse_dates=["send_time_utc", "open_time_utc"]
    )
    logger.info(f"fetched {len(raw):,} deliveries from {len(raw.user_id.unique()):,} users")

    # normalize time zones and compute local send time
    raw["user_timezone"] = raw["user_timezone"].fillna(DEFAULT_ZONE)
    invalid_tz = ~raw["user_timezone"].str.contains(r"/", regex=True)
    if invalid_tz.any():
        logger.warning(f"{invalid_tz.sum():,} records had invalid tz; defaulted to {DEFAULT_ZONE}")
        raw.loc[invalid_tz, "user_timezone"] = DEFAULT_ZONE
    raw["send_time_utc"] = raw["send_time_utc"].dt.tz_localize("UTC", nonexistent=None, ambiguous="NaT")
    local_times = []
    for tz, subset in raw.groupby("user_timezone"):
        local = subset["send_time_utc"].dt.tz_convert(tz)
        local_times.append(pd.DataFrame({
            "weekday": local.dt.weekday,
            "hour": local.dt.hour
        }, index=subset.index))
    raw = raw.join(pd.concat(local_times))
    raw["slot"] = raw["weekday"] * (24 // HOURS_PER_BUCKET) + raw["hour"] // HOURS_PER_BUCKET

    # define success: open within 24h of delivery
    raw["success"] = (
        raw["open_time_utc"].notna() &
        ((raw["open_time_utc"].dt.tz_localize("UTC") - raw["send_time_utc"]) <= timedelta(hours=24))
    ).astype(int)

    # aggregate per user–slot
    user_slot = raw.groupby(["user_id", "slot"]).agg(
        trials=("success", "count"),
        successes=("success", "sum")
    ).reset_index()

    # build global slot-level prior via moment matching
    slot_agg = user_slot.groupby("slot").agg(
        trials=("trials", "sum"),
        successes=("successes", "sum")
    )
    slot_agg["base_rate"] = (slot_agg["successes"] + 0.5) / (slot_agg["trials"] + 1.0)
    filtered = user_slot[user_slot.trials >= 2].copy()
    filtered["empirical_rate"] = filtered["successes"] / filtered["trials"]
    slot_var = filtered.groupby("slot")["empirical_rate"].var()
    slot_mean_n = filtered.groupby("slot")["trials"].mean()
    binom_var = slot_agg["base_rate"] * (1 - slot_agg["base_rate"]) / slot_mean_n.fillna(3)
    between_var = (slot_var.fillna(0) - binom_var).clip(lower=1e-4)
    concentration = (slot_agg["base_rate"] * (1 - slot_agg["base_rate"]) / between_var - 1).clip(2, PRIOR_WEIGHT_MAX)
    prior_alpha = slot_agg["base_rate"] * concentration
    prior_beta = (1 - slot_agg["base_rate"]) * concentration
    prior = pd.Series(index=range(NUM_BUCKETS), dtype=float)
    prior["alpha"] = prior_alpha.reindex(range(NUM_BUCKETS)).fillna(prior_alpha.median())
    prior["beta"] = prior_beta.reindex(range(NUM_BUCKETS)).fillna(prior_beta.median())

    # compute posterior parameters per user–slot
    full_grid = pd.MultiIndex.from_product([raw.user_id.unique(), range(NUM_BUCKETS)], names=["user_id", "slot"])
    user_slot_full = user_slot.set_index(["user_id", "slot"]).reindex(full_grid, fill_value=0).reset_index()
    user_slot_full["alpha_post"] = prior["alpha"].iloc[user_slot_full["slot"]].values + user_slot_full["successes"]
    user_slot_full["beta_post"] = prior["beta"].iloc[user_slot_full["slot"]].values + user_slot_full["trials"] - user_slot_full["successes"]
    user_slot_full["mean"] = user_slot_full["alpha_post"] / (user_slot_full["alpha_post"] + user_slot_full["beta_post"])

    # draw from posteriors (Thompson)
    draws = rng.beta(user_slot_full["alpha_post"], user_slot_full["beta_post"], size=(SAMPLES_PER_ITER, len(user_slot_full)))
    user_slot_full["score"] = draws.mean(axis=0)
    best_idx = user_slot_full.groupby("user_id")["score"].idxmax()
    selections = user_slot_full.loc[best_idx, ["user_id", "slot", "mean", "score"]].set_index("user_id")
    selections.columns = ["slot", "posterior_mean", "thompson_score"]

    # enrich with user metadata
    user_meta = raw.groupby("user_id").agg(
        total_deliveries=("success", "count"),
        user_timezone=("user_timezone", "last"),
        cohort=("cohort", "last")
    )
    user_meta = user_meta.join(selections)
    user_meta["strategy"] = "thompson"

    # fallback to cohort-level best slot for sparse users
    cohort_slot = (
        user_slot_full
        .merge(user_meta[["cohort"]], left_on="user_id", right_index=True)
        .groupby(["cohort", "slot"])[["alpha_post", "beta_post"]].sum()
    )
    cohort_slot["rate"] = cohort_slot["alpha_post"] / (cohort_slot["alpha_post"] + cohort_slot["beta_post"])
    cohort_best = (
        cohort_slot.reset_index()
        .loc[lambda df: df.groupby("cohort")["rate"].idxmax()]
        .set_index("cohort")
    )
    sparse = user_meta["total_deliveries"] < MIN_INDIVIDUAL_DATA
    user_meta.loc[sparse, "slot"] = user_meta.loc[sparse, "cohort"].map(cohort_best["slot"]).fillna(slot_agg["base_rate"].idxmax())
    user_meta.loc[sparse, "posterior_mean"] = user_meta.loc[sparse, "cohort"].map(cohort_best["rate"])
    user_meta.loc[sparse, "strategy"] = "cohort_default"
    logger.info(f"{sparse.sum():,}/{len(user_meta):,} users ({sparse.mean():.1%}) fell back to cohort strategy")

    # format output columns
    user_meta["slot"] = user_meta["slot"].astype(int)
    user_meta["day_of_week"] = user_meta["slot"] // (24 // HOURS_PER_BUCKET)
    user_meta["send_hour"] = (user_meta["slot"] % (24 // HOURS_PER_BUCKET)) * HOURS_PER_BUCKET + HOURS_PER_BUCKET // 2

    result = user_meta.reset_index()[["user_id", "user_timezone", "day_of_week", "send_hour",
                                       "posterior_mean", "strategy", "total_deliveries"]]
    result["generated_at"] = reference

    if dry:
        logger.info(f"dry run: would emit {len(result):,} rows")
    else:
        schema, tbl = TABLE_OUTPUT.split(".")
        result.to_sql(tbl, engine, schema=schema, if_exists="replace", index=False, chunksize=50_000)
        logger.info(f"persisted {len(result):,} rows to {TABLE_OUTPUT}")

    return result


def cli():
    parser = argparse.ArgumentParser(description="Compute optimal send times per user")
    parser.add_argument("--connection", required=True, help="DB connection string")
    parser.add_argument("--asof", default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    execute(sql.create_engine(args.connection), pd.Timestamp(args.asof), args.seed, args.dry)


if __name__ == "__main__":
    cli()
