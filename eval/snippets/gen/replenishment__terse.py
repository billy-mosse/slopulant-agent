import math
import argparse
import logging
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, text

log = logging.getLogger("repl")

Z95 = 1.645
RPT = 1.0
QSPREAD = 2.563

@dataclass(frozen=True)
class VSpec:
    v: str
    lt: float
    cp: int
    moq: int
    uc: float

@dataclass
class OrderRec:
    sku: str
    wh: str
    v: str
    s: float
    S: float
    ip: float
    qty: int
    uc: float
    m: float

    @property
    def ext(self) -> float:
        return self.qty * self.uc

def _load(e):
    fc = pd.read_sql(text("SELECT sku, warehouse_id, horizon, p10, p50, p90 FROM supply.demand_forecast WHERE run_date = (SELECT MAX(run_date) FROM supply.demand_forecast)"), e)
    st = pd.read_sql(text("SELECT sku, warehouse_id, on_hand, on_order, allocated, margin_pct FROM inventory.stock_levels"), e)
    vm = pd.read_sql(text("SELECT sku, vendor_id, lead_time_days, case_pack, moq_units, unit_cost FROM supply.vendor_terms WHERE is_primary"), e)
    return fc, st, vm

def _dmd(fc, w):
    fc = fc.sort_values("horizon")
    f = int(math.floor(w))
    r = w - f
    mu, var = 0.0, 0.0
    for i, row in enumerate(fc.itertuples()):
        wgt = 1.0 if i < f else (r if i == f else 0.0)
        if wgt == 0.0: break
        sig = max(row.p90 - row.p10, 0.0) / QSPREAD
        mu += wgt * row.p50
        var += wgt * sig ** 2
    return mu, math.sqrt(var)

def _pack(q, v):
    if q <= 0: return 0
    p = math.ceil(q / v.cp)
    u = p * v.cp
    if u < v.moq:
        u = math.ceil(v.moq / v.cp) * v.cp
    return int(u)

def _plan(fc, st, vm):
    vmap = {r.sku: VSpec(r.vendor_id, r.lead_time_days/7.0, int(r.case_pack), int(r.moq_units), float(r.unit_cost)) for r in vm.itertuples()}
    out = []
    for (sku, wh), g in fc.groupby(["sku", "warehouse_id"]):
        v = vmap.get(sku)
        if not v: continue
        srow = st[(st.sku == sku) & (st.warehouse_id == wh)]
        if srow.empty: continue
        s = srow.iloc[0]
        mu1, sd1 = _dmd(g, v.lt)
        mu2, sd2 = _dmd(g, v.lt + RPT)
        spt = mu1 + Z95 * sd1
        upt = mu2 + Z95 * sd2
        ip = s.on_hand + s.on_order - s.allocated
        qty = _pack(upt - ip, v) if ip <= spt else 0
        if qty:
            out.append(OrderRec(sku, wh, v.v, spt, upt, ip, qty, v.uc, s.margin_pct))
    return out

def _trim(recs, bud):
    tot = sum(r.ext for r in recs)
    if tot <= bud: return recs
    k = sorted(recs, key=lambda r: r.m, reverse=True)
    while k and tot > bud:
        d = k.pop()
        tot -= d.ext
        log.info("cut: %s @ %s (m %.1f%%, $%.0f)", d.sku, d.wh, 100*d.m, d.ext)
    return k

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dsn", required=True)
    p.add_argument("--bud", type=float, default=750_000.0)
    p.add_argument("--dry", action="store_true")
    a = p.parse_args()
    logging.basicConfig(level=logging.INFO)
    e = create_engine(a.dsn)
    recs = _trim(_plan(*_load(e)), a.bud)
    df = pd.DataFrame([{**r.__dict__, "ext": r.ext} for r in recs])
    df["ts"] = pd.Timestamp.utcnow()
    log.info("%d lines, $%.0f", len(df), df.ext.sum() if len(df) else 0)
    if not a.dry and len(df):
        df.to_sql("purchase_orders", e, schema="supply", if_exists="append", index=False)

if __name__ == "__main__":
    main()
