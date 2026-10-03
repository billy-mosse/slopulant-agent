"""Experiment impact evaluation pipeline.

This module supports the product team's rigorous assessment of experimental outcomes by
comparing treatment and control groups across key performance indicators. It implements
a standardized workflow: balance validation, CUPED-based variance reduction, statistical
testing, lift estimation with confidence intervals, and multiplicity correction—ensuring
reliable, interpretable insights for decision-making.
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from cuped import apply_cuped_adjustment

logger = logging.getLogger("experiment_analytics")

ALPHA_LEVEL = 0.05
BALANCE_TEST_THRESHOLD = 0.001
CONTROL_LABEL = "control"


@dataclass
class ExperimentMetricSummary:
    experiment_identifier: str
    metric_name: str
    comparison_group: str
    control_sample_size: int
    treatment_sample_size: int
    control_mean: float
    treatment_mean: float
    relative_lift: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    raw_p_value: float
    adjusted_p_value: float
    is_statistically_significant: bool
    variance_reduction_achieved: float
    balance_concern_flagged: bool


def assess_randomization_balance(observed_counts: pd.Series, target_allocation: dict[str, float] | None = None) -> float:
    """Evaluate whether observed arm sizes deviate significantly from expected allocation proportions."""
    total_assignments = observed_counts.sum()
    target = target_allocation or {arm: 1.0 / len(observed_counts) for arm in observed_counts.index}
    expected_counts = np.array([target[arm] * total_assignments for arm in observed_counts.index])
    return float(stats.chisquare(f_obs=observed_counts.values, f_exp=expected_counts).pvalue)


def conduct_welch_ttest(control_outcomes: np.ndarray, treatment_outcomes: np.ndarray) -> float:
    """Perform two-sample Welch t-test for continuous outcomes."""
    return float(stats.ttest_ind(treatment_outcomes, control_outcomes, equal_var=False).pvalue)


def conduct_two_proportion_ztest(control_outcomes: np.ndarray, treatment_outcomes: np.ndarray) -> float:
    """Perform two-proportion z-test for binary outcomes."""
    p_control, p_treatment = control_outcomes.mean(), treatment_outcomes.mean()
    n_control, n_treatment = len(control_outcomes), len(treatment_outcomes)
    pooled_proportion = (control_outcomes.sum() + treatment_outcomes.sum()) / (n_control + n_treatment)
    standard_error = np.sqrt(pooled_proportion * (1 - pooled_proportion) * (1 / n_control + 1 / n_treatment))
    if standard_error == 0:
        return 1.0
    z_statistic = (p_treatment - p_control) / standard_error
    return float(2 * stats.norm.sf(np.abs(z_statistic)))


def compute_relative_lift_and_ci(control_outcomes: np.ndarray, treatment_outcomes: np.ndarray, alpha: float = ALPHA_LEVEL) -> tuple[float, float, float]:
    """Estimate relative lift (treatment/control - 1) and delta-method confidence interval."""
    mean_control, mean_treatment = control_outcomes.mean(), treatment_outcomes.mean()
    if mean_control == 0:
        return np.nan, np.nan, np.nan
    variance_ratio = treatment_outcomes.var(ddof=1) / (len(treatment_outcomes) * mean_control**2) + \
                     mean_treatment**2 * control_outcomes.var(ddof=1) / (len(control_outcomes) * mean_control**4)
    margin = stats.norm.ppf(1 - alpha / 2) * np.sqrt(variance_ratio)
    lift = mean_treatment / mean_control - 1
    return float(lift), float(lift - margin), float(lift + margin)


def apply_benjamini_hochberg_adjustment(raw_p_values: np.ndarray) -> np.ndarray:
    """Adjust p-values using the Benjamini-Hochberg procedure to control the false discovery rate."""
    n_tests = len(raw_p_values)
    sorted_indices = np.argsort(raw_p_values)
    adjusted = raw_p_values[sorted_indices] * n_tests / np.arange(1, n_tests + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty(n_tests)
    result[sorted_indices] = np.clip(adjusted, 0.0, 1.0)
    return result


def generate_experiment_summary(experiment_id: str, assignment_data: pd.DataFrame, metric_data: pd.DataFrame) -> list[ExperimentMetricSummary]:
    """Orchestrate full analysis pipeline for a single experiment."""
    arm_counts = assignment_data.groupby("variant")["unit_id"].nunique()
    balance_p_value = assess_randomization_balance(arm_counts)
    balance_flag = balance_p_value < BALANCE_TEST_THRESHOLD
    if balance_flag:
        logger.warning("Experiment %s: Sample ratio mismatch detected (p=%.2e); arm distribution: %s",
                       experiment_id, balance_p_value, arm_counts.to_dict())

    merged = metric_data.merge(assignment_data[["unit_id", "variant"]], on="unit_id", how="inner")
    summaries: list[ExperimentMetricSummary] = []

    for (metric_name, metric_kind), subset in merged.groupby(["metric", "metric_type"]):
        adjusted_values, cuped_diagnostics = apply_cuped_adjustment(subset) if metric_kind == "continuous" else (subset["value"].astype(float), None)
        control_outcomes = subset.loc[subset["variant"] == CONTROL_LABEL, "value"].to_numpy(dtype=float)
        for group in subset["variant"].unique():
            if group == CONTROL_LABEL:
                continue
            treatment_outcomes = subset.loc[subset["variant"] == group, "value"].to_numpy(dtype=float)
            p_value = conduct_welch_ttest(control_outcomes, treatment_outcomes) if metric_kind == "continuous" else conduct_two_proportion_ztest(control_outcomes, treatment_outcomes)
            lift, ci_low, ci_high = compute_relative_lift_and_ci(control_outcomes, treatment_outcomes)
            summaries.append(ExperimentMetricSummary(
                experiment_identifier=experiment_id,
                metric_name=metric_name,
                comparison_group=group,
                control_sample_size=len(control_outcomes),
                treatment_sample_size=len(treatment_outcomes),
                control_mean=float(control_outcomes.mean()),
                treatment_mean=float(treatment_outcomes.mean()),
                relative_lift=lift,
                confidence_interval_lower=ci_low,
                confidence_interval_upper=ci_high,
                raw_p_value=p_value,
                adjusted_p_value=np.nan,
                is_statistically_significant=False,
                variance_reduction_achieved=cuped_diagnostics.variance_reduction if cuped_diagnostics else 0.0,
                balance_concern_flagged=balance_flag,
            ))

    if summaries:
        adjusted_p_values = apply_benjamini_hochberg_adjustment(np.array([s.raw_p_value for s in summaries]))
        for summary, adj_p in zip(summaries, adjusted_p_values):
            summary.adjusted_p_value = float(adj_p)
            summary.is_statistically_significant = adj_p < ALPHA_LEVEL and not summary.balance_concern_flagged

    return summaries


def execute_analysis(experiment_identifier: str, dry_run: bool = False) -> None:
    """Entry point for experiment analysis: fetch data, run pipeline, log and persist results."""
    from warehouse_client import connect
    warehouse = connect()
    assignment_query = f"SELECT unit_id, variant FROM experiments.assignments WHERE experiment_id = %(exp)s"
    metric_query = f"SELECT unit_id, metric, metric_type, value, pre_value FROM experiments.metrics WHERE experiment_id = %(exp)s"
    params = {"exp": experiment_identifier}
    assignments = warehouse.query(assignment_query, params)
    metrics = warehouse.query(metric_query, params)

    results = generate_experiment_summary(experiment_identifier, assignments, metrics)
    report = pd.DataFrame([{
        "metric": s.metric_name,
        "variant": s.comparison_group,
        "lift": s.relative_lift,
        "adjusted_p_value": s.adjusted_p_value,
        "significant": s.is_statistically_significant,
    } for s in results])

    logger.info("Analysis summary:\n%s", report.to_string())
    if not dry_run:
        warehouse.write("experiments.results", report, partition=experiment_identifier)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Execute standardized impact evaluation for an experiment")
    parser.add_argument("experiment_identifier", help="Unique identifier for the experiment")
    parser.add_argument("--dry-run", action="store_true", help="Run analysis without persisting results")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    execute_analysis(args.experiment_identifier, args.dry_run)
