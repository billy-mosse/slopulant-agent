import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable

import numpy as np
import pandas as pd
from scipy.stats import anderson_ksamp, chi2_contingency

log = logging.getLogger("ml_ops.drift")

ALERTS_DB = "analytics.drift_notifications"
WINDOW_HOURS = 24
SLACK_URL_VAR = "DRIFT_ALERT_WEBHOOK"

THRESHOLDS = [
    ("psi", lambda v: v > 0.2, "critical"),
    ("psi", lambda v: v > 0.1, "warning"),
    ("anderson_p", lambda v: v < 0.01, "critical"),
    ("null_change", lambda v: abs(v) > 0.05, "critical"),
    ("z_score", lambda v: abs(v) > 0.5, "warning"),
]


@dataclass
class Signal:
    model_id: str
    field_name: str
    stat_name: str
    observed: float
    notes: str = ""
    level: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def fingerprint(self) -> str:
        key = f"{self.model_id}|{self.field_name}|{self.stat_name}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]


def grade(sig: Signal) -> str | None:
    for metric, cond, lvl in THRESHOLDS:
        if sig.stat_name == metric and cond(sig.observed):
            return lvl
    return None


def slack_payload(events: list[Signal], day: date) -> dict:
    header = f"*Data drift summary – {day}*\n{len(events)} flagged"
    rows = []
    for s in sorted(events, key=lambda x: (x.level != "critical", x.model_id, x.field_name)):
        icon = "🔴" if s.level == "critical" else "🟡"
        rows.append(f"{icon} `{s.model_id}` • `{s.field_name}` • {s.stat_name}={s.observed:.3g} {s.notes}".strip())
    return {"blocks": [{"type": "section", "text": {"type": "mrkdwn", "text": header}}] + [{"type": "section", "text": {"type": "mrkdwn", "text": r}} for r in rows]}


class NotificationHub:
    def __init__(self, conn, notify: bool = True) -> None:
        self.conn = conn
        self.notify = notify
        self.buffer: list[Signal] = []
        self.today: date | None = None

    def ingest(self, items: list[Signal], day: date) -> None:
        self.today = day
        for s in items:
            s.level = grade(s)
            if s.level:
                self.buffer.append(s)

    def _active_hashes(self) -> set[str]:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=WINDOW_HOURS)
        q = f"SELECT fingerprint FROM {ALERTS_DB} WHERE created_at >= %(t)s"
        df = self.conn.execute(q, {"t": cutoff}).fetch_pandas()
        return set(df["fingerprint"].tolist()) if not df.empty else set()

    def dispatch(self) -> list[Signal]:
        active = self._active_hashes()
        unique: dict[str, Signal] = {}
        for s in self.buffer:
            if s.fingerprint not in active and s.fingerprint not in unique:
                unique[s.fingerprint] = s
        kept = list(unique.values())
        log.info("Alerts after dedup: %d (%d dropped)", len(kept), len(self.buffer) - len(kept))
        self.buffer.clear()
        if not kept:
            return []
        self.conn.write(ALERTS_DB, pd.DataFrame([{"model_id": s.model_id, "field_name": s.field_name, "stat_name": s.stat_name,
                                                  "observed": s.observed, "level": s.level, "fingerprint": s.fingerprint,
                                                  "notes": s.notes, "created_at": s.timestamp} for s in kept]),
                        partition_key=self.today.isoformat())
        if self.notify and (url := os.getenv(SLACK_URL_VAR)):
            payload = json.dumps(slack_payload(kept, self.today)).encode()
            req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=8)
        return kept


def bin_edges(ref: np.ndarray, k: int = 10) -> np.ndarray:
    q = np.quantile(ref, np.linspace(0, 1, k + 1))
    return np.unique(np.clip(q, np.min(ref) - 1e-9, np.max(ref) + 1e-9))


def psi_score(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    edges = bin_edges(ref, bins)
    r_hist, _ = np.histogram(ref, edges)
    c_hist, _ = np.histogram(cur, edges)
    r = (r_hist + 1e-8) / r_hist.sum()
    c = (c_hist + 1e-8) / c_hist.sum()
    return float(np.sum((c - r) * np.log(c / r)))


def anderson_darling(ref: np.ndarray, cur: np.ndarray) -> tuple[float, float]:
    # Approximate Anderson–Darling via pooled empirical CDF
    combined = np.concatenate([ref, cur])
    ranks = np.argsort(np.argsort(combined))
    n1, n2 = len(ref), len(cur)
    e = np.arange(1, len(combined) + 1) / len(combined)
    F = np.concatenate([np.searchsorted(np.sort(ref), combined) / n1,
                        np.searchsorted(np.sort(cur), combined) / n2])
    A2 = -len(combined) - np.sum((2 * e - 1) * np.log(F * (1 - F) + 1e-12))
    p = max(1e-6, 1 - chi2.cdf(A2 * (len(combined) / (len(combined) - 1)), df=1))
    return A2, p


def extract_features(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [col for col in df.columns if col not in exclude and not col.startswith("__")]


def compute_signals(model: str, ref: pd.DataFrame, cur: pd.DataFrame, cols: list[str]) -> list[Signal]:
    signals = []
    for col in cols:
        r = ref[col].dropna().astype(float)
        c = cur[col].dropna().astype(float)
        if r.empty or c.empty:
            continue
        null_delta = float(c.isna().mean() - ref[col].isna().mean())
        signals.append(Signal(model, col, "null_change", null_delta))
        psi_val = psi_score(r.values, c.values)
        signals.append(Signal(model, col, "psi", psi_val))
        if pd.api.types.is_numeric_dtype(ref[col]) and ref[col].nunique() > 15:
            ad_stat, ad_p = anderson_darling(r.values, c.values)
            signals.append(Signal(model, col, "anderson_p", ad_p, notes=f"A²={ad_stat:.2f}"))
    return signals


def prediction_signals(model: str, ref_scores: pd.Series, cur_scores: pd.Series) -> list[Signal]:
    r = ref_scores.dropna().astype(float).values
    c = cur_scores.dropna().astype(float).values
    if r.size == 0 or c.size == 0:
        return []
    mean_diff = (c.mean() - r.mean()) / (r.std(ddof=1) + 1e-8)
    psi_val = psi_score(r, c)
    ad_stat, ad_p = anderson_darling(r, c)
    return [
        Signal(model, "__score__", "psi", psi_val),
        Signal(model, "__score__", "z_score", mean_diff),
        Signal(model, "__score__", "anderson_p", ad_p, notes=f"A²={ad_stat:.2f}"),
    ]


def fetch_window(conn, model: str, start: date, end: date) -> pd.DataFrame:
    sql = f"""
        SELECT p.customer_id, p.score, f.*
        FROM analytics.prediction_events p
        LEFT JOIN analytics.feature_snapshots f
          ON f.customer_id = p.customer_id AND f.snapshot_date = p.event_date
        WHERE p.model_name = %(m)s AND p.event_date >= %(s)s AND p.event_date < %(e)s
    """
    return conn.execute(sql, {"m": model, "s": start, "e": end}).fetch_pandas()


def execute(conn, models: list[str], day: date, hub: NotificationHub) -> None:
    cur_start = day - timedelta(days=1)
    ref_start = cur_start - timedelta(days=28)
    for m in models:
        ref_df = fetch_window(conn, m, ref_start, cur_start)
        cur_df = fetch_window(conn, m, cur_start, day)
        if cur_df.empty:
            log.warning("No data for %s in %s–%s", m, cur_start, day)
            continue
        cols = extract_features(ref_df, {"customer_id", "score", "event_date", "snapshot_date"})
        sigs = compute_signals(m, ref_df, cur_df, cols) + prediction_signals(m, ref_df.score, cur_df.score)
        log.info("%s: %d signals generated", m, len(sigs))
        hub.ingest(sigs, day)
    hub.dispatch()


def cli() -> None:
    import argparse
    p = argparse.ArgumentParser("Drift checker")
    p.add_argument("--model", action="append", required=True)
    p.add_argument("--date", type=date.fromisoformat, default=date.today())
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    logging.basicConfig(level=logging.WARNING if args.quiet else logging.INFO)
    from warehouse import Client
    conn = Client()
    execute(conn, args.model, args.date, NotificationHub(conn, notify=not args.quiet))


if __name__ == "__main__":
    cli()
