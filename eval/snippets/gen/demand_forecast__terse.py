import argparse
import logging
from typing import List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from features import KEYS, build_features, load_weekly_panel
from forecast import HORIZON, recursive_forecast, train_models

log = logging.getLogger("backtest")


def _wape(a: np.ndarray, p: np.ndarray) -> float:
    s = np.abs(a).sum()
    return float(np.abs(a - p).sum() / s) if s else np.nan


def _bias(a: np.ndarray, p: np.ndarray) -> float:
    s = a.sum()
    return float((p.sum() - s) / s) if s else np.nan


def _origins(weeks: pd.Series, folds: int, step: int) -> List[pd.Timestamp]:
    uniq = np.sort(weeks.unique())
    last = len(uniq) - HORIZON - 1
    return [pd.Timestamp(uniq[last - i * step]) for i in range(folds)][::-1]


def _fold(panel: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    hist = panel[panel["week_start"] <= origin]
    fut = panel[panel["week_start"].between(origin + pd.Timedelta(weeks=1), origin + pd.Timedelta(weeks=HORIZON))]
    mod = train_models(build_features(hist.copy()), num_rounds=400)
    fc = recursive_forecast(mod, hist, fut[KEYS + ["week_start", "markdown_pct"]].drop_duplicates())
    out = fc.merge(fut[KEYS + ["category", "week_start", "units"]], on=KEYS + ["week_start"])
    out["origin"] = origin
    return out


def _summarize(res: pd.DataFrame) -> pd.DataFrame:
    res["h_grp"] = pd.cut(res["horizon"], [0, 4, 8, 12], labels=["1-4", "5-8", "9-12"])
    rows = []
    for (cat, hb), g in res.groupby(["category", "h_grp"], observed=True):
        a, p = g["units"].to_numpy(), g["p50"].to_numpy()
        cov = ((g["units"] >= g["p10"]) & (g["units"] <= g["p90"])).mean()
        rows.append({"category": cat, "horizon": hb, "wape": _wape(a, p), "bias": _bias(a, p),
                     "cov": cov, "n": len(g)})
    return pd.DataFrame(rows).sort_values(["category", "horizon"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--folds", type=int, default=6)
    ap.add_argument("--step", type=int, default=4)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    panel = load_weekly_panel(create_engine(args.dsn), "2022-01-03")
    res = []
    for o in _origins(panel["week_start"], args.folds, args.step):
        fold = _fold(panel, o)
        a, p = fold["units"].to_numpy(), fold["p50"].to_numpy()
        log.info("origin %s: WAPE=%.3f bias=%+.3f", o.date(), _wape(a, p), _bias(a, p))
        res.append(fold)
    print(_summarize(pd.concat(res, ignore_index=True)).to_string(index=False))


if __name__ == "__main__":
    main()
