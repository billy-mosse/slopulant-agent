from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("experiment_analytics")

ASSIGNMENTS = "experiments.assignments"
METRICS = "experiments.metrics"
RESULTS = "experiments.results"
BASELINE = "control"
SRM_THRESHOLD = 1e-3
ALPHA = 0.05
MetricKind = Literal["continuous", "binary"]


@dataclass
class ExperimentOutcome:
    exp_id: str
    metric_name: str
    arm: str
    ctrl_n: int
    treat_n: int
    ctrl_mean: float
    treat_mean: float
    rel_lift: float
    ci_lower: float
    ci_upper: float
    raw_p: float
    fdr_p: float = np.nan
    is_signif: bool = False
    var_red: float = 0.0
    srm_issue: bool = False


def balance_test(observed: pd.Series, target: dict[str, float] | None = None) -> float:
    """Goodness-of-fit test for arm allocation balance."""
    n = observed.sum()
    target = target or {k: 1 / len(observed) for k in observed.index}
    expected = np.array([target[k] * n for k in observed.index])
    return float(stats.chisquare(observed.values, expected).pvalue)


def continuous_test(ctrl: np.ndarray, treat: np.ndarray) -> float:
    return float(stats.ttest_ind(ctrl, treat, equal_var=False).pvalue)


def binary_test(ctrl: np.ndarray, treat: np.ndarray) -> float:
    p0, p1 = ctrl.mean(), treat.mean()
    n0, n1 = len(ctrl), len(treat)
    p_pool = (ctrl.sum() + treat.sum()) / (n0 + n1)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n0 + 1 / n1))
    if se == 0:
        return 1.0
    return float(2 * stats.norm.sf(np.abs((p1 - p0) / se)))


def lift_interval(ctrl: np.ndarray, treat: np.ndarray, conf: float = 0.95) -> tuple[float, float, float]:
    """Relative lift = (mean_treat / mean_ctrl) - 1 with delta-method CI."""
    m0, m1 = ctrl.mean(), treat.mean()
    if m0 == 0:
        return np.nan, np.nan, np.nan
    var_ratio = treat.var(ddof=1) / (len(treat) * m0**2) + m1**2 * ctrl.var(ddof=1) / (len(ctrl) * m0**4)
    z = stats.norm.ppf(conf + (1 - conf) / 2)
    lift = m1 / m0 - 1
    margin = z * np.sqrt(var_ratio)
    return float(lift), float(lift - margin), float(lift + margin)


def fdr_correct(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR control."""
    k = len(pvals)
    idx = np.argsort(pvals)
    adj = pvals[idx] * k / (np.arange(1, k + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(k)
    out[idx] = np.clip(adj, 0, 1)
    return out


def variance_reduce(df: pd.DataFrame, metric: str, pre: str) -> tuple[pd.Series, float]:
    """Apply CUPED: Y_adj = Y - θ(X - X̄), θ = Cov(X,Y)/Var(X)."""
    y = df[metric].astype(float).values
    pre_cov = df[pre].astype(float)
    if pre_cov.notna().mean() < 0.5:
        return df[metric].astype(float), 0.0
    x = pre_cov.fillna(pre_cov.mean()).values
    var_x = np.var(x, ddof=1)
    if var_x == 0:
        return df[metric].astype(float), 0.0
    theta = float(np.cov(x, y, ddof=1)[0, 1] / var_x)
    y_adj = y - theta * (x - x.mean())
    var_y = np.var(y, ddof=1)
    red = 1 - np.var(y_adj, ddof=1) / var_y if var_y > 0 else 0.0
    return pd.Series(y_adj, index=df.index), red


def run_experiment(exp: str, assignments: pd.DataFrame, metrics: pd.DataFrame) -> list[ExperimentOutcome]:
    arm_counts = assignments.groupby("arm")["unit_id"].nunique()
    srm_p = balance_test(arm_counts)
    srm_issue = srm_p < SRM_THRESHOLD
    if srm_issue:
        log.warning("SRM violation in %s: p=%.2e, distribution=%s", exp, srm_p, arm_counts.to_dict())
    merged = metrics.merge(assignments[["unit_id", "arm"]], on="unit_id", how="inner")
    outcomes = []
    for (metric, kind), sub in merged.groupby(["metric", "metric_type"]):
        if kind == "continuous":
            adj_vals, red = variance_reduce(sub, "value", "pre_value")
            sub = sub.assign(value=adj_vals)
        else:
            red = 0.0
        ctrl_vals = sub.loc[sub.arm == BASELINE, "value"].to_numpy()
        for arm in sub.arm.unique():
            if arm == BASELINE:
                continue
            treat_vals = sub.loc[sub.arm == arm, "value"].to_numpy()
            p = continuous_test(ctrl_vals, treat_vals) if kind == "continuous" else binary_test(ctrl_vals, treat_vals)
            lift, lo, hi = lift_interval(ctrl_vals, treat_vals)
            outcomes.append(ExperimentOutcome(
                exp, metric, arm, len(ctrl_vals), len(treat_vals),
                float(ctrl_vals.mean()), float(treat_vals.mean()),
                lift, lo, hi, p, var_red=red, srm_issue=srm_issue,
            ))
    if outcomes:
        adj_ps = fdr_correct(np.array([o.raw_p for o in outcomes]))
        for o, padj in zip(outcomes, adj_ps):
            o.fdr_p = float(padj)
            o.is_signif = bool(padj < ALPHA and not o.srm_issue)
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute A/B test analytics pipeline")
    parser.add_argument("experiment_id", help="Experiment identifier")
    parser.add_argument("--commit", action="store_true", help="Write results to warehouse")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    from warehouse import get_connection
    conn = get_connection()
    params = {"eid": args.experiment_id}
    assignments = conn.fetch(f"SELECT unit_id, arm FROM {ASSIGNMENTS} WHERE experiment_id = %(eid)s", params)
    metrics = conn.fetch(f"SELECT unit_id, metric, metric_type, value, pre_value FROM {METRICS} WHERE experiment_id = %(eid)s", params)
    results = run_experiment(args.experiment_id, assignments, metrics)
    summary = pd.DataFrame([{
        "metric": r.metric_name,
        "arm": r.arm,
        "lift": r.rel_lift,
        "fdr_p": r.fdr_p,
        "significant": r.is_signif,
    } for r in results])
    log.info("Summary:\n%s", summary.to_string())
    if args.commit:
        full_df = pd.DataFrame([{
            "experiment_id": r.exp_id,
            "metric": r.metric_name,
            "arm": r.arm,
            "n_control": r.ctrl_n,
            "n_treatment": r.treat_n,
            "mean_control": r.ctrl_mean,
            "mean_treatment": r.treat_mean,
            "lift": r.rel_lift,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "p_value": r.raw_p,
            "fdr_p": r.fdr_p,
            "significant": r.is_signif,
            "variance_reduction": r.var_red,
            "srm_flag": r.srm_issue,
        } for r in results])
        conn.write(RESULTS, full_df, partition=args.experiment_id)


if __name__ == "__main__":
    main()
