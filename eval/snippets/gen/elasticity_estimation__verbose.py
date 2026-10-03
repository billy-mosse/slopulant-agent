"""Module for calculating own-price elasticity metrics at the category level.

This tool estimates price sensitivity for each product category using a
fixed-effects regression on weekly SKU-level sales data. The underlying
assumption is that consumers adjust purchase quantities in response to price
changes, and we quantify this relationship after controlling for SKU-level
heterogeneity and seasonal week-of-year effects.

The final outputs incorporate empirical Bayes shrinkage to stabilize estimates
for categories with limited variation or high uncertainty, borrowing strength
across the portfolio while preserving true heterogeneity. Cluster-robust
standard errors are computed at the SKU level to accommodate within-SKU
correlation over time.

Expected data origin: weekly aggregates generated from order lines and pricing
history (see *data pipeline documentation* for preprocessing).
"""
import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text


log = logging.getLogger("pricing_observability")

MINIMAL_SKUS_THRESHOLD = 3
MINIMAL_OBSERVATIONS_THRESHOLD = 60


@dataclass
class CategoryElasticityModel:
    category_label: str
    base_elasticity: float
    standard_error: float
    total_observations: int
    distinct_skus: int
    within_r_squared: float


def ingest_sales_data(database_url: str, observation_window_weeks: int) -> pd.DataFrame:
    sql_script = (Path(__file__).parent / "sql" / "weekly_sales.sql").read_text()
    engine = create_engine(database_url)
    with engine.connect() as conn:
        raw = pd.read_sql(text(sql_script), conn, params={"lookback_weeks": observation_window_weeks})
    cleaned = raw[(raw["units"] > 0) & (raw["avg_price"] > 0)].copy()
    cleaned["natural_log_quantity"] = np.log(cleaned["units"])
    cleaned["natural_log_price"] = np.log(cleaned["avg_price"])
    return cleaned


def remove_fixed_effects(
    data: pd.DataFrame,
    variables_to_center: List[str],
    grouping_key: str,
) -> pd.DataFrame:
    grouped_means = data.groupby(grouping_key)[variables_to_center].transform("mean")
    return data[variables_to_center] - grouped_means


def fit_single_category(data: pd.DataFrame, category_name: str) -> Optional[CategoryElasticityModel]:
    if data["sku"].nunique() < MINIMAL_SKUS_THRESHOLD or len(data) < MINIMAL_OBSERVATIONS_THRESHOLD:
        return None

    week_of_year_dummies = pd.get_dummies(
        data["week_of_year"].astype(int), prefix="w", drop_first=True, dtype=float
    )
    regressor_frame = pd.concat([data[["natural_log_price"]], week_of_year_dummies], axis=1)
    regressor_frame["sku_identifier"] = data["sku"].values
    regressor_frame["natural_log_quantity"] = data["natural_log_quantity"].values
    covariates = [col for col in regressor_frame.columns if col not in ("sku_identifier",)]

    demeaned = remove_fixed_effects(regressor_frame, covariates, "sku_identifier")
    dependent = demeaned["natural_log_quantity"].to_numpy()
    regressors = demeaned.drop(columns="natural_log_quantity").to_numpy()

    valid_regressors = np.abs(regressors).sum(axis=0) > 1e-12
    regressors = regressors[:, valid_regressors]
    if not valid_regressors[0]:
        log.warning("Category %s lacks within-SKU price variation", category_name)
        return None

    information_matrix_inv = np.linalg.pinv(regressors.T @ regressors)
    coefficient_vector = information_matrix_inv @ regressors.T @ dependent
    residuals = dependent - regressors @ coefficient_vector

    cluster_variance_component = np.zeros((regressors.shape[1], regressors.shape[1]))
    sku_ids = data["sku"].to_numpy()
    for sku_group in np.unique(sku_ids):
        mask = sku_ids == sku_group
        score_sum = regressors[mask].T @ residuals[mask]
        cluster_variance_component += np.outer(score_sum, score_sum)

    number_of_clusters = len(np.unique(sku_ids))
    degrees_of_freedom_correction = (
        (number_of_clusters / (number_of_clusters - 1))
        * ((len(dependent) - 1) / max(len(dependent) - regressors.shape[1] - number_of_clusters, 1))
    )
    robust_variance = degrees_of_freedom_correction * (
        information_matrix_inv @ cluster_variance_component @ information_matrix_inv
    )
    standard_error = float(np.sqrt(max(robust_variance[0, 0], 0.0)))
    r_squared = (
        1.0 - residuals.var() / dependent.var() if dependent.var() > 0 else 0.0
    )

    return CategoryElasticityModel(
        category_name,
        float(coefficient_vector[0]),
        standard_error,
        len(dependent),
        number_of_clusters,
        float(r_squared),
    )


def apply_bayesian_shrinkage(
    category_estimates: List[CategoryElasticityModel],
) -> pd.DataFrame:
    summary = pd.DataFrame([vars(c) for c in category_estimates])
    precision_weights = 1.0 / np.clip(summary["standard_error"], 1e-6, np.inf) ** 2
    pooled_elasticity = np.sum(precision_weights * summary["base_elasticity"]) / np.sum(precision_weights)

    between_variation_statistic = np.sum(precision_weights * (summary["base_elasticity"] - pooled_elasticity) ** 2)
    denominator = np.sum(precision_weights) - np.sum(precision_weights ** 2) / np.sum(precision_weights)
    between_variance_estimate = max(0.0, (between_variation_statistic - (len(summary) - 1)) / denominator) if denominator > 0 else 0.0
    log.info("Global elasticity estimate: %.3f, estimated between-category variance: %.4f", pooled_elasticity, between_variance_estimate)

    shrinkage_factors = between_variance_estimate / (between_variance_estimate + summary["standard_error"] ** 2) if between_variance_estimate > 0 else 0.0
    summary["final_elasticity"] = shrinkage_factors * summary["base_elasticity"] + (1 - shrinkage_factors) * pooled_elasticity

    summary["t_value"] = summary["base_elasticity"] / summary["standard_error"]
    summary["two_tailed_p"] = 2 * stats.t.sf(np.abs(summary["t_value"]), df=(summary["distinct_skus"] - 1).clip(lower=1))
    summary["ninety_five_ci_lower"] = summary["base_elasticity"] - 1.96 * summary["standard_error"]
    summary["ninety_five_ci_upper"] = summary["base_elasticity"] + 1.96 * summary["standard_error"]
    summary["pooled_elasticity_reference"] = pooled_elasticity

    return summary.rename(columns={"base_elasticity": "unshrunk_elasticity"})


def run_elasticity_analysis(
    dsn: str,
    lookback_period_weeks: int,
    is_dry_run: bool,
) -> None:
    panel = ingest_sales_data(dsn, lookback_period_weeks)
    log.info(
        "Loaded panel: %d rows, %d unique SKUs, %d categories",
        len(panel), panel["sku"].nunique(), panel["category"].nunique()
    )

    fitted_categories = []
    for label, subset in panel.groupby("category"):
        result = fit_single_category(subset.reset_index(drop=True), str(label))
        if result is None:
            log.info("Skipped category %s due to insufficient data", label)
            continue
        fitted_categories.append(result)

    if not fitted_categories:
        raise SystemExit("No eligible categories found for elasticity modeling")

    outcomes = apply_bayesian_shrinkage(fitted_categories)
    outcomes["executed_at"] = pd.Timestamp.utcnow().normalize()

    positive_count = (outcomes["final_elasticity"] > 0).sum()
    if positive_count:
        log.warning("%d categories exhibit positive elasticity after shrinkage", positive_count)

    if is_dry_run:
        print(outcomes.sort_values("final_elasticity").to_string(index=False))
        return

    outcomes.to_sql(
        "elasticities",
        create_engine(dsn),
        schema="pricing",
        if_exists="replace",
        index=False,
    )
    log.info("Successfully persisted %d elasticity records", len(outcomes))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute category-level own-price elasticity estimates")
    parser.add_argument("--dsn", required=True, help="Database connection string")
    parser.add_argument("--lookback-weeks", type=int, default=104, help="Number of weeks of historical data to include")
    parser.add_argument("--dry-run", action="store_true", help="Run analysis without persisting results")
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_elasticity_analysis(arguments.dsn, arguments.lookback_weeks, arguments.dry_run)
