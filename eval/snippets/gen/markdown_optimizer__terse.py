import argparse
import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import create_engine

log = logging.getLogger("md")

DISC_STEPS = np.array([0.0, 0.1, 0.2, 0.3, 0.4])
BKT_CNT = 50
HOLD_COST = 0.004
SALVAGE = 0.15
PENALTY = 0.25
DEF_ELAST = -1.5

def _fetch(engine):
    s = pd.read_sql(
        "SELECT sku,category,on_hand_units,full_price,baseline_weekly_units,season_end_date "
        f"FROM inventory.stock_levels WHERE season_end_date > CURRENT_DATE", engine)
    e = pd.read_sql("SELECT category,elasticity FROM pricing.elasticities", engine)
    return s, e

def _excess(df, today):
    wks = ((pd.to_datetime(df.season_end_date) - pd.Timestamp(today)).dt.days // 7).clip(0)
    df = df.assign(wks_left=wks)
    proj = df.baseline_weekly_units * df.wks_left
    return df[(df.wks_left > 0) & (df.on_hand_units > proj)].copy()

def _demand(q0, d, e): 
    return q0 * (1 - d) ** e

def _solve(u, p, q0, e, w):
    bs = max(u / BKT_CNT, 1.0)
    nb = int(np.ceil(u / bs)) + 1
    nd = len(DISC_STEPS)
    V = np.zeros((w + 1, nb, nd))
    ch = np.zeros((w, nb, nd), dtype=int)

    for b in range(nb):
        V[w, b, :] = b * bs * p * (SALVAGE - PENALTY)

    for t in range(w - 1, -1, -1):
        for b in range(nb):
            st = b * bs
            for d in range(nd):
                mx, mk = -np.inf, d
                for k in range(d, nd):
                    q = _demand(q0, DISC_STEPS[k], e)
                    sl = min(q, st)
                    rm = st - sl
                    rev = sl * p * (1 - DISC_STEPS[k])
                    hc = rm * p * HOLD_COST
                    nb2 = min(int(round(rm / bs)), nb - 1)
                    val = rev - hc + V[t + 1, nb2, k]
                    if val > mx:
                        mx, mk = val, k
                V[t, b, d] = mx
                ch[t, b, d] = mk

    ladder, rows = [], []
    st, d = float(u), 0
    for t in range(w):
        b = min(int(round(st / bs)), nb - 1)
        k = ch[t, b, d]
        q = _demand(q0, DISC_STEPS[k], e)
        sl = min(q, st)
        rows.append((t, DISC_STEPS[k], sl, st - sl))
        st -= sl
        d = k
        ladder.append(DISC_STEPS[k])
    b0 = min(int(round(u / bs)), nb - 1)
    return ladder, rows, float(V[0, b0, 0])

def _mkplan(df, el, today):
    em = dict(zip(el.category, el.elasticity))
    out = []
    for r in df.itertuples(index=False):
        e = em.get(r.category, DEF_ELAST)
        if e > -0.2:
            log.warning("sku %s: bad elast %.2f, fallback", r.sku, e)
            e = DEF_ELAST
        ld, rows, val = _solve(int(r.on_hand_units), float(r.full_price),
                               float(r.baseline_weekly_units), e, int(r.wks_left))
        for wk, d, sl, rm in rows:
            out.append({
                "sku": r.sku,
                "week_start": today + timedelta(weeks=wk),
                "disc_pct": round(d * 100),
                "price": round(r.full_price * (1 - d), 2),
                "sold": round(sl, 1),
                "left": round(rm, 1),
                "e_used": e,
                "val": round(val, 2),
            })
        log.info("sku %s: ladder %s, leftover %.0f", r.sku,
                 "/".join(f"{int(x*100)}" for x in ld), rows[-1][3] if rows else 0)
    return pd.DataFrame(out)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    eng = create_engine(args.dsn)
    st, el = _fetch(eng)
    cand = _excess(st, args.as_of)
    log.info("%d/%d SKUs excess", len(cand), len(st))

    pl = _mkplan(cand, el, args.as_of)
    if args.dry or pl.empty:
        print(pl.head(50).to_string(index=False))
        return
    pl.to_sql("markdown_plan", eng, schema="pricing", if_exists="replace", index=False)
    log.info("wrote %d rows to pricing.markdown_plan", len(pl))

if __name__ == "__main__":
    main()
