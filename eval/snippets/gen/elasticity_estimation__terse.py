import argparse
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import create_engine, text

log = logging.getLogger("elasticity")

SQL_FILE = Path(__file__).parent / "sql" / "weekly_sales.sql"
OUT_TBL = "pricing.elasticities"
MIN_SKU = 3
MIN_ROW = 60


@dataclass
class CatRes:
    cat: str
    b: float
    s: float
    n: int
    k: int
    r2: float


def pull(engine, wks: int) -> pd.DataFrame:
    q = SQL_FILE.read_text()
    with engine.connect() as c:
        df = pd.read_sql(text(q), c, params={"lookback_weeks": wks})
    df = df[(df.units > 0) & (df.avg_price > 0)].copy()
    df.lq = np.log(df.units)
    df.lp = np.log(df.avg_price)
    return df


def _dm(df: pd.DataFrame, cols: list, g: str) -> pd.DataFrame:
    return df[cols] - df.groupby(g)[cols].transform("mean")


def est(df: pd.DataFrame, cat: str) -> Optional[CatRes]:
    if df.sku.nunique() < MIN_SKU or len(df) < MIN_ROW:
        return None
    woy = pd.get_dummies(df.week_of_year.astype(int), prefix="w", drop_first=True, dtype=float)
    X = pd.concat([df[["lp"]], woy], axis=1)
    X["sku"] = df.sku.values
    X["lq"] = df.lq.values
    cols = [c for c in X.columns if c not in ("sku",)]
    Z = _dm(X, cols, "sku")
    y = Z.lq.to_numpy()
    Xc = Z.drop(columns="lq").to_numpy()
    keep = np.abs(Xc).sum(0) > 1e-12
    Xc = Xc[:, keep]
    if not keep[0]:
        log.warning("no price var in %s", cat)
        return None
    inv = np.linalg.pinv(Xc.T @ Xc)
    b = inv @ Xc.T @ y
    r = y - Xc @ b
    meat = np.zeros((Xc.shape[1], Xc.shape[1]))
    for g in np.unique(df.sku):
        idx = df.sku == g
        s = Xc[idx].T @ r[idx]
        meat += np.outer(s, s)
    G = len(np.unique(df.sku))
    n = len(y)
    k = Xc.shape[1]
    c = (G / (G - 1)) * ((n - 1) / max(n - k - G, 1))
    vc = c * inv @ meat @ inv
    se = float(np.sqrt(max(vc[0, 0], 0.0)))
    r2 = 1 - r.var() / y.var() if y.var() > 0 else 0.0
    return CatRes(cat, float(b[0]), se, n, G, float(r2))


def adj(fits: List[CatRes]) -> pd.DataFrame:
    df = pd.DataFrame([f.__dict__ for f in fits])
    w = 1 / np.clip(df.s, 1e-6, None) ** 2
    bg = np.sum(w * df.b) / np.sum(w)
    q = np.sum(w * (df.b - bg) ** 2)
    d = w.sum() - (w ** 2).sum() / w.sum()
    t2 = max(0.0, (q - (len(df) - 1)) / d) if d > 0 else 0.0
    log.info("global %.3f, tau2 %.4f, Q %.1f", bg, t2, q)
    df.w = t2 / (t2 + df.s ** 2) if t2 > 0 else 0.0
    df.el = df.w * df.b + (1 - df.w) * bg
    df.t = df.b / df.s
    df.p = 2 * stats.t.sf(np.abs(df.t), df=(df.k - 1).clip(lower=1))
    df.lo = df.b - 1.96 * df.s
    df.hi = df.b + 1.96 * df.s
    df.glob = bg
    return df.rename(columns={"b": "raw"})


def run() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--lookback", type=int, default=104)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    eng = create_engine(a.dsn)
    df = pull(eng, a.lookback)
    log.info("panel: %d rows, %d skus, %d cats", len(df), df.sku.nunique(), df.cat.nunique())
    res = []
    for c, g in df.groupby("cat"):
        r = est(g.reset_index(drop=True), str(c))
        if r is None:
            log.info("skip %s", c)
            continue
        res.append(r)
    if not res:
        raise SystemExit("no estimates")
    out = adj(res)
    out.run_date = pd.Timestamp.utcnow().normalize()
    pos = (out.el > 0).sum()
    if pos:
        log.warning("%d positive elasticities", pos)
    if a.dry:
        print(out.sort_values("el").to_string(index=False))
        return
    sch, tbl = OUT_TBL.split(".")
    out.to_sql(tbl, eng, schema=sch, if_exists="replace", index=False)
    log.info("wrote %d rows to %s", len(out), OUT_TBL)


if __name__ == "__main__":
    run()
