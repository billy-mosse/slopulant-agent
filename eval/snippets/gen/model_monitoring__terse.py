from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats

log = logging.getLogger("model_monitoring.drift")

ALERTS_TBL = "monitoring.drift_alerts"
DEDUP_HRS = 24
SLACK_URL_VAR = "DRIFT_SLACK_WEBHOOK"

RULES = [
    ("psi", "gt", 0.2, "alert"),
    ("psi", "gt", 0.1, "warn"),
    ("ks_pvalue", "lt", 0.001, "alert"),
    ("null_rate_delta", "gt", 0.05, "alert"),
    ("mean_shift_sd", "abs_gt", 0.5, "warn"),
]

PRED_TBL = "monitoring.prediction_logs"
FEAT_TBL = "features.customer_daily"
N_BINS = 10
EPS = 1e-6
REF_DAYS = 28
CURR_DAYS = 1


@dataclass
class DriftHit:
    model: str
    feat: str
    metric: str
    val: float
    note: str = ""
    sev: str | None = None
    ts: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def k(self) -> tuple[str, str, str]:
        return self.model, self.feat, self.metric


def _sev_of(h: DriftHit) -> str | None:
    for m, op, th, sev in RULES:
        if m != h.metric:
            continue
        if op == "gt" and h.val > th:
            return sev
        if op == "lt" and h.val < th:
            return sev
        if op == "abs_gt" and abs(h.val) > th:
            return sev
    return None


def _slack_fmt(alerts: list[DriftHit], d: date) -> dict:
    lines = [f"*Model drift {d.isoformat()}* ({len(alerts)} alerts)"]
    for a in sorted(alerts, key=lambda x: (x.sev != "alert", x.model, x.feat)):
        icon = ":red_circle:" if a.sev == "alert" else ":large_yellow_circle:"
        lines.append(f"{icon} `{a.model}` / `{a.feat}` {a.metric}={a.val:.4g} {a.note}".rstrip())
    return {"text": "\n".join(lines)}


class DriftSink:
    def __init__(self, wh, slack: bool = True) -> None:
        self.wh = wh
        self.slack = slack
        self.buf: list[DriftHit] = []
        self.d: date | None = None

    def add(self, hits: list[DriftHit], d: date) -> None:
        self.d = d
        for h in hits:
            h.sev = _sev_of(h)
            if h.sev:
                self.buf.append(h)

    def _seen(self) -> set[tuple[str, str, str]]:
        since = datetime.now(timezone.utc) - timedelta(hours=DEDUP_HRS)
        df = self.wh.query(
            f"SELECT model, feature, metric FROM {ALERTS_TBL} WHERE created_at >= %(since)s", {"since": since})
        return set(map(tuple, df[["model", "feature", "metric"]].itertuples(index=False)))

    def commit(self) -> list[DriftHit]:
        seen = self._seen()
        fresh: dict[tuple[str, str, str], DriftHit] = {}
        for h in self.buf:
            if h.k not in seen and h.k not in fresh:
                fresh[h.k] = h
        alerts = list(fresh.values())
        log.info("%d alerts after dedup (%d suppressed)", len(alerts), len(self.buf) - len(alerts))
        self.buf.clear()
        if not alerts:
            return []
        self.wh.write(ALERTS_TBL, pd.DataFrame([asdict(a) for a in alerts]), partition=self.d.isoformat())
        if self.slack and (url := os.environ.get(SLACK_URL_VAR)):
            body = json.dumps(_slack_fmt(alerts, self.d)).encode()
            req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10)
        return alerts


def _qbins(ref: np.ndarray, n: int = N_BINS) -> np.ndarray:
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, n + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return edges


def _psi(ref: np.ndarray, cur: np.ndarray, n: int = N_BINS) -> float:
    edges = _qbins(ref, n)
    r = np.histogram(ref, edges)[0] / max(len(ref), 1)
    c = np.histogram(cur, edges)[0] / max(len(cur), 1)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def _is_cont(s: pd.Series) -> bool:
    return pd.api.types.is_float_dtype(s) or (pd.api.types.is_integer_dtype(s) and s.nunique() > 20)


def _feat_hits(model: str, ref: pd.DataFrame, cur: pd.DataFrame, feats: list[str]) -> list[DriftHit]:
    out: list[DriftHit] = []
    for f in feats:
        r, c = ref[f], cur[f]
        nj = float(c.isna().mean() - r.isna().mean())
        out.append(DriftHit(model, f, "null_rate_delta", nj))
        r, c = r.dropna().to_numpy(dtype=float), c.dropna().to_numpy(dtype=float)
        if len(r) == 0 or len(c) == 0:
            continue
        out.append(DriftHit(model, f, "psi", _psi(r, c)))
        if _is_cont(ref[f]):
            ks = stats.ks_2samp(r, c)
            out.append(DriftHit(model, f, "ks_pvalue", float(ks.pvalue), note=f"D={ks.statistic:.3f}"))
    return out


def _pred_hits(model: str, ref: pd.Series, cur: pd.Series) -> list[DriftHit]:
    r, c = ref.to_numpy(dtype=float), cur.to_numpy(dtype=float)
    shift = (c.mean() - r.mean()) / (r.std(ddof=1) + EPS)
    return [
        DriftHit(model, "__prediction__", "psi", _psi(r, c)),
        DriftHit(model, "__prediction__", "mean_shift_sd", float(shift)),
        DriftHit(model, "__prediction__", "ks_pvalue", float(stats.ks_2samp(r, c).pvalue)),
    ]


def _load(wh, model: str, start: date, end: date) -> pd.DataFrame:
    sql = f"""
        SELECT p.customer_id, p.score, f.*
        FROM {PRED_TBL} p
        LEFT JOIN {FEAT_TBL} f
          ON f.customer_id = p.customer_id AND f.as_of_date = CAST(p.predicted_at AS DATE)
        WHERE p.model_name = %(model)s AND p.predicted_at >= %(start)s AND p.predicted_at < %(end)s
    """
    return wh.query(sql, {"model": model, "start": start, "end": end})


def go(wh, models: list[str], d: date, sink: DriftSink) -> None:
    cur_w = (d - timedelta(days=CURR_DAYS), d)
    ref_w = (cur_w[0] - timedelta(days=REF_DAYS), cur_w[0])
    for m in models:
        ref, cur = _load(wh, m, *ref_w), _load(wh, m, *cur_w)
        if cur.empty:
            log.warning("%s: no data in %s", m, cur_w)
            continue
        feats = [c for c in ref.columns if c not in {"customer_id", "score", "as_of_date", "expires_at"}]
        hits = _feat_hits(m, ref, cur, feats) + _pred_hits(m, ref.score, cur.score)
        log.info("%s: %d findings over %d features", m, len(hits), len(feats))
        sink.add(hits, d)
    sink.commit()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--date", type=date.fromisoformat, default=date.today())
    ap.add_argument("--no-slack", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    from warehouse_client import connect
    wh = connect()
    go(wh, args.model, args.date, DriftSink(wh, slack=not args.no_slack))


if __name__ == "__main__":
    main()
