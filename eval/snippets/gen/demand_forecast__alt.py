import argparse
import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from catboost import CatBoostRegressor, CatBoostClassifier

from features import KEYS, build_features, load_weekly_panel

log = logging.getLogger("demand_forecast.backtest")

HORIZON = 12


def mape(actual: np.ndarray, pred: np.ndarray) -> float:
    denom = np.abs(actual).sum()
    return float(np.abs(actual - pred).sum() / denom) if denom > 0 else np.nan


def relative_bias(actual: np.ndarray, pred: np.ndarray) -> float:
    total = actual.sum()
    return float((pred.sum() - total) / total) if total > 0 else np.nan


def generate_origins(week_series: pd.Series, n_folds: int, step_size: int) -> list[pd.Timestamp]:
    unique_weeks = np.sort(week_series.unique())
    cutoff = len(unique_weeks) - HORIZON - 1
    origins = [pd.Timestamp(unique_weeks[cutoff - i * step_size]) for i in range(n_folds)]
    return list(reversed(origins))


def execute_fold(data: pd.DataFrame, origin_date: pd.Timestamp) -> pd.DataFrame:
    train = data[data["week_start"] <= origin_date]
    test_window = data[
        (data["week_start"] > origin_date) &
        (data["week_start"] <= origin_date + timedelta(weeks=HORIZON))
    ]
    train_features = build_features(train.copy())
    point_model = CatBoostRegressor(iterations=300, learning_rate=0.04, depth=6, verbose=0)
    point_model.fit(train_features.drop(columns=["units", "week_start"]), train_features["units"])
    quantile_models = {}
    for q in [0.1, 0.5, 0.9]:
        q_model = CatBoostRegressor(iterations=200, learning_rate=0.03, depth=5, verbose=0)
        q_model.fit(train_features.drop(columns=["units", "week_start"]), train_features["units"],
                    sample_weight=np.where(train_features["units"] == 0, 0.5, 1.0))
        quantile_models[f"p{int(q * 100)}"] = q_model
    preds = []
    current = train.copy()
    last_date = current["week_start"].max()
    for h in range(1, HORIZON + 1):
        target_date = last_date + timedelta(weeks=h)
        future_row = test_window[test_window["week_start"] == target_date][KEYS + ["category"]].drop_duplicates()
        future_row = future_row.merge(
            test_window[["sku", "week_start", "markdown_pct"]].drop_duplicates(),
            on=["sku", "week_start"], how="left"
        )
        future_row["markdown_pct"] = future_row["markdown_pct"].fillna(0.0)
        combined = pd.concat([current, future_row], ignore_index=True)
        feats = build_features(combined)
        X = feats[feats["week_start"] == target_date].drop(columns=["units", "week_start"])
        p50 = np.maximum(point_model.predict(X), 0)
        p10 = np.minimum(quantile_models["p10"].predict(X), p50)
        p90 = np.maximum(quantile_models["p90"].predict(X), p50)
        res = future_row[KEYS].copy()
        res["week_start"] = target_date
        res["horizon"] = h
        res["p50"] = p50
        res["p10"] = p10
        res["p90"] = p90
        preds.append(res)
        future_row["units"] = p50
        current = pd.concat([current, future_row], ignore_index=True)
    merged = pd.concat(preds, ignore_index=True).merge(
        test_window[KEYS + ["week_start", "category", "units"]],
        on=KEYS + ["week_start"], how="left"
    )
    merged["origin"] = origin_date
    return merged


def aggregate_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df["h_group"] = pd.cut(df["horizon"], bins=[0, 4, 8, 12], labels=["1-4", "5-8", "9-12"])
    summary = []
    for (cat, hb), sub in df.groupby(["category", "h_group"], observed=True):
        y_true = sub["units"].to_numpy()
        y_hat = sub["p50"].to_numpy()
        coverage = ((sub["units"] >= sub["p10"]) & (sub["units"] <= sub["p90"])).mean()
        summary.append({
            "category": cat,
            "horizon": hb,
            "mape": mape(y_true, y_hat),
            "rel_bias": relative_bias(y_true, y_hat),
            "interval_coverage": coverage,
            "count": len(sub)
        })
    return pd.DataFrame(summary).sort_values(["category", "horizon"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-uri", required=True)
    parser.add_argument("--folds", type=int, default=6)
    parser.add_argument("--stride", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    engine = create_engine(args.db_uri)
    panel = load_weekly_panel(engine, "2022-01-03")
    all_results = []
    for origin in generate_origins(panel["week_start"], args.folds, args.stride):
        fold_df = execute_fold(panel, origin)
        y_true = fold_df["units"].to_numpy()
        y_hat = fold_df["p50"].to_numpy()
        log.info("origin %s: MAPE=%.3f rel_bias=%+.3f", origin.date(), mape(y_true, y_hat), relative_bias(y_true, y_hat))
        all_results.append(fold_df)
    print(aggregate_metrics(pd.concat(all_results, ignore_index=True)).to_string(index=False))


if __name__ == "__main__":
    main()
