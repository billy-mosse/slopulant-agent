import argparse
import logging
import math
from typing import NamedTuple

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger("stock_planner")

Z_95 = 1.645
REVIEW_INTERVAL = 1  # weeks
IQR_TO_STD = 1.349  # approx (P90-P10)/2 under normality

class VendorSpec(NamedTuple):
    vendor: str
    lead_time_wk: float
    case_size: int
    min_order: int
    price: float

class ReplenishAction(NamedTuple):
    item: str
    loc: str
    supplier: str
    trigger: float
    target: float
    position: float
    amount: int
    cost: float
    gross_margin: float

    @property
    def line_value(self) -> float:
        return self.amount * self.cost


def pull_data(db_url: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with create_engine(db_url).connect() as conn:
        hist = pd.read_sql(
            text("SELECT sku, warehouse, week, p10, p50, p90 FROM demand.horizon "
                 "WHERE snapshot = (SELECT max(snapshot) FROM demand.horizon)"), conn)
        pos = pd.read_sql(
            text("SELECT sku, warehouse, physical, inbound, reserved, margin FROM inventory.balance"), conn)
        vends = pd.read_sql(
            text("SELECT sku, vendor, lead_time_days, case_size, min_order, price FROM supply.vendors "
                 "WHERE active = true AND primary_flag = true"), conn)
    return hist, pos, vends


def forecast_window(demand: pd.DataFrame, weeks: float) -> tuple[float, float]:
    demand = demand.sort_values("week")
    full, frac = divmod(weeks, 1.0)
    mu, var = 0.0, 0.0
    for idx, row in demand.iterrows():
        weight = 1.0 if idx < full else (frac if idx == full else 0.0)
        if weight <= 0:
            break
        sigma = max(row.p90 - row.p10, 0.0) / (2 * IQR_TO_STD)
        mu += weight * row.p50
        var += weight * sigma ** 2
    return mu, math.sqrt(var)


def pack_round(order: float, spec: VendorSpec) -> int:
    if order <= 0:
        return 0
    packs = math.ceil(order / spec.case_size)
    units = packs * spec.case_size
    if units < spec.min_order:
        packs = math.ceil(spec.min_order / spec.case_size)
        units = packs * spec.case_size
    return int(units)


def plan_restock(demand: pd.DataFrame, inventory: pd.DataFrame, vendors: pd.DataFrame) -> list[ReplenishAction]:
    specs = {row.sku: VendorSpec(row.vendor, row.lead_time_days / 7.0, row.case_size,
                                 row.min_order, row.price) for _, row in vendors.iterrows()}
    actions = []
    for (item, loc), group in demand.groupby(["sku", "warehouse"]):
        spec = specs.get(item)
        if not spec:
            logger.warning("Missing vendor spec for %s", item)
            continue
        stock = inventory[(inventory.sku == item) & (inventory.warehouse == loc)]
        if stock.empty:
            continue
        s = stock.iloc[0]
        lt_mu, lt_sig = forecast_window(group, spec.lead_time_wk)
        lt_rv_mu, lt_rv_sig = forecast_window(group, spec.lead_time_wk + REVIEW_INTERVAL)
        reorder = lt_mu + Z_95 * lt_sig
        cap = lt_rv_mu + Z_95 * lt_rv_sig
        ip = s.physical + s.inbound - s.reserved
        qty = pack_round(cap - ip, spec) if ip <= reorder else 0
        if qty:
            actions.append(ReplenishAction(item, loc, spec.vendor, reorder, cap, ip, qty, spec.price, s.margin))
    return actions


def enforce_cap(actions: list[ReplenishAction], limit: float) -> list[ReplenishAction]:
    total = sum(a.line_value for a in actions)
    if total <= limit:
        return actions
    candidates = sorted(actions, key=lambda a: a.gross_margin)
    while candidates and total > limit:
        cut = candidates.pop(0)
        total -= cut.line_value
        logger.info("Budget cap triggered: removed %s at %s (margin %.1f%%, $%d)",
                    cut.item, cut.loc, 100 * cut.gross_margin, cut.line_value)
    return candidates


def execute(args: argparse.Namespace) -> None:
    logging.basicConfig(level=logging.INFO)
    hist, pos, vends = pull_data(args.dsn)
    orders = enforce_cap(plan_restock(hist, pos, vends), args.budget)
    df = pd.DataFrame([a._asdict() | {"line_value": a.line_value} for a in orders])
    df["timestamp"] = pd.Timestamp.utcnow()
    logger.info("Generated %d PO entries, total value $%d", len(df), df.line_value.sum() if len(df) else 0)
    if not args.dry_run and len(df):
        df.to_sql("orders", create_engine(args.dsn), schema="supply", if_exists="append", index=False)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--budget", type=float, default=800_000)
    p.add_argument("--dry-run", action="store_true")
    execute(p.parse_args())
