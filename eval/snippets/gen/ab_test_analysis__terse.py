from __future__ import annotations

import argparse
import logging
from dataclasses import asdict
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats

from cuped import CupedResult, adjust

log = logging.getLogger("ab_test_analysis")

ALPHA = 0.05
SRM_THRESH = 0.001
CONT = "control"
MType = Literal["continuous", "binary"]


class Res:
    def __init__(self, eid, met, var, nc, nt, mc, mt, lift, lo, hi, p, adj=0.0, sig=False, vr=0.0, srm=False):
        self.expid = eid
        self.metric = met
        self.variant = var
        self.nc, self.nt = nc, nt
        self.mc, self.mt = mc, mt
        self.lift, self.lo, self.hi = lift, lo, hi
        self.pval, self.adjp = p, adj
        self.sig, self.vr, self.srm = sig, vr, srm


def _srm_test(counts: pd.Series) -> float:
    tot = counts.sum()
    exp = np.full(len(counts), 1 / len(counts)) * tot
    return float(stats.chisquare(counts.values, exp).pvalue)


def _t_welch(c: np.ndarray, t: np.ndarray) -> float:
    return float(stats.ttest_ind(t, c, equal_var=False).pvalue)


def _z_prop(c: np.ndarray, t: np.ndarray) -> float:
    pc, pt = c.mean(), t.mean()
    nc, nt = len(c), len(t)
    pooled = (c.sum() + t.sum()) / (nc + nt)
    se = np.sqrt(pooled * (1 - pooled) * (1 / nc + 1 / nt))
    if se == 0:
        return 1.0
    return float(2 * stats.norm.sf(abs((pt - pc) / se)))


def _lift_ci(c: np.ndarray, t: np.ndarray, a: float = ALPHA) -> tuple[float, float, float]:
    mc, mt = c.mean(), t.mean()
    if mc == 0:
        return np.nan, np.nan, np.nan
    vr = t.var(ddof=1) / (len(t) * mc**2) + mt**2 * c.var(ddof=1) / (len(c) * mc**4)
    z = stats.norm.ppf(1 - a / 2)
    lift = mt / mc - 1
    h = z * np.sqrt(vr)
    return float(lift), float(lift - h), float(lift + h)


def _bh_adj(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    ord_i = np.argsort(pvals)
    adj = pvals[ord_i] * n / (np.arange(1, n + 1))
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n)
    out[ord_i] = np.clip(adj, 0, 1)
    return out


def run(eid: str, df_a: pd.DataFrame, df_m: pd.DataFrame) -> list[Res]:
    arms = df_a.groupby("variant")["unit_id"].nunique()
    srm_p = _srm_test(arms)
    srm = srm_p < SRM_THRESH
    if srm:
        log.warning("%s: srm p=%.2e", eid, srm_p)
    df = df_m.merge(df_a[["unit_id", "variant"]], on="unit_id", how="inner")
    res = []
    for (met, mtyp), grp in df.groupby(["metric", "metric_type"]):
        y = adjust(grp).Y_adj if mtyp == "continuous" else grp["value"].astype(float)
        c_vals = grp.loc[grp.variant == CONT, "value"].to_numpy()
        y_c = grp.loc[grp.variant == CONT, "value"].astype(float).to_numpy() if mtyp == "continuous" else c_vals
        for var in grp.variant.unique():
            if var == CONT:
                continue
            t_vals = grp.loc[grp.variant == var, "value"].to_numpy()
            y_t = grp.loc[grp.variant == var, "value"].astype(float).to_numpy() if mtyp == "continuous" else t_vals
            p = _t_welch(y_c, y_t) if mtyp == "continuous" else _z_prop(y_c, y_t)
            lift, lo, hi = _lift_ci(y_c, y_t)
            res.append(Res(eid, met, var, len(y_c), len(y_t), float(np.mean(y_c)), float(np.mean(y_t)),
                           lift, lo, hi, p, vr=0.0, srm=srm))
    if res:
        padj = _bh_adj(np.array([r.pval for r in res]))
        for i, r in enumerate(res):
            r.adjp, r.sig = float(padj[i]), bool(padj[i] < ALPHA and not r.srm)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("expid")
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    from warehouse_client import connect
    wh = connect()
    df_a = wh.query("SELECT unit_id, variant FROM experiments.assignments WHERE experiment_id = %(e)s", {"e": args.expid})
    df_m = wh.query(
        "SELECT unit_id, metric, metric_type, value, pre_value FROM experiments.metrics WHERE experiment_id = %(e)s", {"e": args.expid})
    res = [asdict(r) for r in run(args.expid, df_a, df_m)]
    df_r = pd.DataFrame(res)
    log.info("\n%s", df_r[["metric", "variant", "lift", "adjp", "sig"]].to_string())
    if not args.dry:
        wh.write("experiments.results", df_r, partition=args.expid)


if __name__ == "__main__":
    main()
