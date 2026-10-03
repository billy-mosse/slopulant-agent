import argparse
import logging
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("promo_scoring")

DB_URI = "postgresql://analytics@warehouse/slopulent"
QUERY = """
    SELECT customer_id,
           discount_pct,
           redeemed,
           sent_at,
           last_order_date
    FROM marketing.promo_history
    WHERE sent_at >= CURRENT_DATE - INTERVAL '18 months'
"""
TABLE_OUT = "marketing.coupon_propensity"
DEPTHS = [10, 15, 20, 25]
VALIDATION_WINDOW = 90
FEAT_COLS = ["prev_sends", "prev_redemptions", "redemption_ratio",
             "discount_pct", "days_since_order", "days_since_last_redeem",
             "avg_redeemed_discount"]

class TemporalAggregator:
    def fit(self, *args, **kwargs): return self
    def transform(self, df):
        df = df.sort_values(["customer_id", "sent_at"]).copy()
        g = df.groupby("customer_id")
        df["prev_sends"] = g.cumcount() - 1
        df["prev_redemptions"] = g["redeemed"].cumsum() - df["redeemed"]
        df["redemption_ratio"] = (df["prev_redemptions"] + 0.5) / (df["prev_sends"] + 2).clip(lower=1)
        df["days_since_order"] = (df["sent_at"] - df["last_order_date"]).dt.days.clip(lower=0)
        redeem_dates = df["sent_at"].where(df["redeemed"].astype(bool))
        df["prev_redeem_date"] = redeem_dates.groupby(df["customer_id"]).shift(1)
        df["prev_redeem_date"] = df.groupby("customer_id")["prev_redeem_date"].ffill()
        df["days_since_last_redeem"] = (df["sent_at"] - df["prev_redeem_date"]).dt.days
        df["avg_redeemed_discount"] = (df["discount_pct"].where(df["redeemed"].astype(bool))
                                       .groupby(df["customer_id"])
                                       .transform(lambda x: x.shift(1).expanding().mean()))
        return df

def fetch_data(conn):
    df = pd.read_sql(QUERY, conn, parse_dates=["sent_at", "last_order_date"])
    logger.info("Fetched %d promo events for %d unique customers", len(df), df["customer_id"].nunique())
    return df

def split_train_test(df, window_days=VALIDATION_WINDOW):
    cutoff = df["sent_at"].max() - timedelta(days=window_days)
    return df[df["sent_at"] < cutoff], df[df["sent_at"] >= cutoff]

def build_estimator():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler())
    ])
    preproc = ColumnTransformer([
        ("numeric", num_pipe, FEAT_COLS)
    ])
    return Pipeline([
        ("pre", preproc),
        ("gb", GradientBoostingRegressor(n_estimators=120, max_depth=4,
                                         learning_rate=0.05, subsample=0.8, random_state=42))
    ])

def evaluate(model, X, y):
    preds = model.predict(X)
    mae = np.mean(np.abs(preds - y))
    r2 = 1 - np.sum((y - preds) ** 2) / np.sum((y - y.mean()) ** 2)
    logger.info("Validation MAE=%.4f  R²=%.3f", mae, r2)

def latest_profile(df, now):
    last = df.groupby("customer_id").tail(1).copy()
    last["prev_sends"] = 0
    last["prev_redemptions"] = last["redeemed"]
    last["redemption_ratio"] = (last["prev_redemptions"] + 0.5) / 2
    last["days_since_order"] = (now - last["last_order_date"]).dt.days.clip(lower=0)
    last["days_since_last_redeem"] = np.nan
    return last

def generate_scores(model, profile, now):
    rows = []
    for d in DEPTHS:
        tmp = profile.assign(discount_pct=d)
        tmp["days_since_last_redeem"] = np.where(d == profile["discount_pct"], 0, np.nan)
        tmp["avg_redeemed_discount"] = np.where(d == profile["discount_pct"], d, np.nan)
        tmp["avg_redeemed_discount"] = tmp["avg_redeemed_discount"].fillna(method="ffill").fillna(0)
        preds = model.predict(tmp[FEAT_COLS])
        rows.append(pd.DataFrame({
            "customer_id": tmp["customer_id"],
            "discount_pct": d,
            "p_redeem": np.clip(preds, 0.0, 1.0)
        }))
    result = pd.concat(rows, ignore_index=True)
    result["scored_at"] = now
    return result

def main():
    parser = argparse.ArgumentParser(description="Estimate coupon redemption likelihood")
    parser.add_argument("--no-upload", action="store_true")
    args = parser.parse_args()

    conn = create_engine(DB_URI)
    raw = fetch_data(conn)
    enriched = TemporalAggregator().fit_transform(raw)
    train_df, val_df = split_train_test(enriched)

    est = build_estimator()
    est.fit(train_df[FEAT_COLS], train_df["redeemed"])
    evaluate(est, val_df[FEAT_COLS], val_df["redeemed"])

    est.fit(enriched[FEAT_COLS], enriched["redeemed"])
    now = pd.Timestamp.utcnow().replace(tzinfo=None)
    profile = latest_profile(enriched, now)
    scores = generate_scores(est, profile, now)

    logger.info("Generated %d scores (%d customers × %d depths)", len(scores),
                scores["customer_id"].nunique(), len(DEPTHS))

    if not args.no_upload:
        scores.to_sql(TABLE_OUT.split(".")[1], conn, schema="marketing", if_exists="replace", index=False)
        logger.info("Saved to %s", TABLE_OUT)

if __name__ == "__main__":
    main()
