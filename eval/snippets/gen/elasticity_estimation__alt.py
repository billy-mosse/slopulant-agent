from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize
from sqlalchemy import create_engine, text

log = logging.getLogger("demand_analytics")

SQL_FILE = Path(__file__).resolve().parent / "sql" / "weekly_sales.sql"
TARGET_TABLE = "pricing.elasticities"
SKU_MIN = 3
OBS_MIN = 60


@dataclass
class CategoryResult:
    cat: str
    point_est: float
    std_err: float
    obs_count: int
    sku_count: int
    adj_r2: float


def fetch_data(dsn: str, horizon: int) -> pd.DataFrame:
    sql = SQL_FILE.read_text()
    with create_engine(dsn).connect() as conn:
        raw = pd.read_sql(text(sql), conn, params={"lookback_weeks": horizon})
    mask = (raw.units > 0) & (raw.avg_price > 0)
    df = raw[mask].assign(
        ln_q=lambda x: np.log(x.units),
        ln_p=lambda x: np.log(x.avg_price),
        woy=lambda x: x.week_of_year.astype(int)
    )
    return df


def _center(df: pd.DataFrame, vars_: list[str], group_key: str) -> pd.DataFrame:
    return df[vars_].sub(df.groupby(group_key)[vars_].transform("mean"))


def estimate_category(df: pd.DataFrame, cat: str) -> CategoryResult | None:
    if df.sku.nunique() < SKU_MIN or len(df) < OBS_MIN:
        return None

    dummies = pd.get_dummies(df.woy, prefix="w", drop_first=True).astype(float)
    base = pd.concat([df[["ln_p"]], dummies], axis=1)
    base["sku"] = df.sku.values
    base["ln_q"] = df.ln_q.values
    vars_to_center = [c for c in base.columns if c not in ("sku",)]
    demeaned = _center(base, vars_to_center, "sku")

    y = demeaned.ln_q.to_numpy()
    X = demeaned.drop(columns="ln_q").to_numpy()
    valid_cols = np.where(X.sum(axis=0) > 1e-10)[0]
    if 0 not in valid_cols:
        log.warning("category %s lacks within-SKU price variation", cat)
        return None
    X = X[:, valid_cols]

    # OLS via normal equations
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta_hat = XtX_inv @ X.T @ y
    residuals = y - X @ beta_hat

    # Clustered SE by SKU
    meat = np.zeros((X.shape[1], X.shape[1]))
    for sku_id in np.unique(df.sku):
        idx = df.sku == sku_id
        xu = X[idx].T @ residuals[idx]
        meat += np.outer(xu, xu)
    n_skus = len(np.unique(df.sku))
    n_obs, n_params = len(y), X.shape[1]
    finite_df = max(n_obs - n_params - n_skus, 1)
    scale = (n_skus / (n_skus - 1)) * ((n_obs - 1) / finite_df)
    cov = scale * XtX_inv @ meat @ XtX_inv
    se = float(np.sqrt(max(cov[0, 0], 0.0)))

    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return CategoryResult(
        cat=cat,
        point_est=float(beta_hat[0]),
        std_err=se,
        obs_count=n_obs,
        sku_count=n_skus,
        adj_r2=r2
    )


def bayes_shrink(results: list[CategoryResult]) -> pd.DataFrame:
    df = pd.DataFrame([vars(r) for r in results])
    prec = 1.0 / np.maximum(df.std_err, 1e-8) ** 2
    mu_hat = np.average(df.point_est, weights=prec)

    # Empirical Bayes variance component via moment matching
    q = np.sum(prec * (df.point_est - mu_hat) ** 2)
    denom = np.sum(prec) - np.sum(prec**2) / np.sum(prec)
    tau_sq = max(0.0, (q - (len(df) - 1)) / max(denom, 1e-12))

    log.info("global mean %.4f, between-var %.4f", mu_hat, tau_sq)

    w = tau_sq / (tau_sq + df.std_err**2) if tau_sq > 0 else 0.0
    df["shrunk"] = w * df.point_est + (1 - w) * mu_hat
    df["t_val"] = df.point_est / df.std_err
    df["p_val"] = 2 * (1 - pd.Series(df.t_val).abs().map(lambda t: 0.5 * (1 + np.math.erf(t / np.sqrt(2)))))
    df["ci_low"] = df.point_est - 1.96 * df.std_err
    df["ci_high"] = df.point_est + 1.96 * df.std_err
    df["global_mean"] = mu_hat
    return df.rename(columns={"point_est": "base_elasticity"})


def run_pipeline(dsn: str, horizon: int, dry: bool) -> None:
    panel = fetch_data(dsn, horizon)
    log.info("loaded %d rows (%d SKUs, %d categories)",
             len(panel), panel.sku.nunique(), panel.category.nunique())

    fits = []
    for name, group in panel.groupby("category"):
        res = estimate_category(group.reset_index(drop=True), str(name))
        if res is None:
            log.info("skipping %s – too few data", name)
            continue
        fits.append(res)
    if not fits:
        raise SystemExit("no estimable categories found")

    out = bayes_shrink(fits)
    out["ingest_ts"] = pd.Timestamp.utcnow().floor("D")

    if dry:
        print(out.sort_values("shrunk").to_string(index=False))
        return

    schema, tbl = TARGET_TABLE.split(".")
    out.to_sql(tbl, create_engine(dsn), schema=schema, if_exists="replace", index=False)
    log.info("persisted %d rows to %s", len(out), TARGET_TABLE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate own-price elasticities with EB shrinkage")
    parser.add_argument("--dsn", required=True, help="Database connection string")
    parser.add_argument("--horizon", type=int, default=104, help="Lookback window in weeks")
    parser.add_argument("--dry-run", action="store_true")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parser.parse_args()
    run_pipeline(args.dsn, args.horizon, args.dry_run)


if __name__ == "__main__":
    main()
