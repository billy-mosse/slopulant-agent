import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln, hyp2f1

SRC = "orders.lines"
DST = "scores.clv"
WEEKS_YR = 52.0
DISC_YR = 0.10


def _rfm(df, cutoff):
    d = df.assign(d= df.order_ts.dt.normalize())
    d = d.groupby(["customer_id", "d"], as_index=False).net_revenue.sum()
    g = d.groupby("customer_id")
    f, l = g.d.min(), g.d.max()
    r = pd.DataFrame({
        "freq": g.d.count() - 1,
        "rec": (l - f).dt.days / 7.0,
        "T": (cutoff - f).dt.days / 7.0,
    })
    rep = d[d.d > d.customer_id.map(f)]
    r["mon"] = rep.groupby("customer_id").net_revenue.mean().reindex(r.index).fillna(0.0)
    return r


def _ll_bg(p, x, tx, T):
    r, α, a, b = np.exp(p)
    A1 = gammaln(r + x) - gammaln(r) + r * np.log(α)
    A2 = betaln(a, b + x) - betaln(a, b)
    A3 = -(r + x) * np.log(α + T)
    A4 = np.where(x > 0, np.log(a) - np.log(np.maximum(b + x - 1, 1e-12)) - (r + x) * np.log(α + tx), -np.inf)
    return -np.sum(A1 + A2 + np.logaddexp(A3, A4))


def _fit_bg(df):
    x, tx, T = (df[c].to_numpy(float) for c in ("freq", "rec", "T"))
    res = minimize(lambda p: _ll_bg(p, x, tx, T) / len(x) + 1e-3 * np.sum(np.exp(p) ** 2),
                   np.zeros(4), method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-7, "fatol": 1e-9})
    return np.exp(res.x)


def _exp_bg(p, t, x, tx, T):
    r, α, a, b = p
    z = t / (α + T + t)
    h = hyp2f1(r + x, b + x, a + b + x - 1, z)
    num = (a + b + x - 1) / (a - 1) * (1 - ((α + T) / (α + T + t)) ** (r + x) * h)
    den = 1 + (x > 0) * a / (b + x - 1) * ((α + T) / (α + tx)) ** (r + x)
    return num / den


def _alive(p, x, tx, T):
    r, α, a, b = p
    odds = (x > 0) * a / (b + x - 1) * ((α + T) / (α + tx)) ** (r + x)
    return 1.0 / (1.0 + odds)


def _ll_gg(lp, x, m):
    p, q, g = np.exp(lp)
    return -np.sum(gammaln(p * x + q) - gammaln(p * x) - gammaln(q) + q * np.log(g)
                   + (p * x - 1) * np.log(m) + p * x * np.log(x) - (p * x + q) * np.log(g + m * x))


def _fit_gg(df):
    r = df[(df.freq > 0) & (df.mon > 0)]
    x, m = r.freq.to_numpy(float), r.mon.to_numpy(float)
    res = minimize(lambda lp: _ll_gg(lp, x, m) / len(x), np.log([1.0, 1.0, m.mean()]), method="L-BFGS-B")
    return np.exp(res.x)


def _aov_gg(p, x, m):
    p_, q_, g_ = p
    pop = p_ * g_ / (q_ - 1)
    w = (q_ - 1) / (p_ * x + q_ - 1)
    return w * pop + (1 - w) * np.where(x > 0, m, pop)


def _score(df, bg, gg, horizon=WEEKS_YR):
    x, tx, T, m = (df[c].to_numpy(float) for c in ("freq", "rec", "T", "mon"))
    months = np.arange(1, int(round(horizon / 4.345)) + 1)
    cum = np.stack([_exp_bg(bg, k * 4.345, x, tx, T) for k in np.r_[0, months]])
    disc = (1 + DISC_YR) ** (-months / 12.0)
    aov = _aov_gg(gg, x, m)
    out = df.copy()
    out["alive"] = _alive(bg, x, tx, T)
    out["purch_12"] = cum[-1]
    out["aov"] = aov
    out["clv"] = aov * (np.diff(cum, axis=0) * disc[:, None]).sum(0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--asof", required=True)
    a = ap.parse_args()
    import sqlalchemy as sa
    eng = sa.create_engine(a.dsn)
    asof = pd.Timestamp(a.asof)
    df = pd.read_sql(f"SELECT customer_id, order_ts, net_revenue FROM {SRC} WHERE order_ts < %(asof)s AND status = 'completed'",
                     eng, params={"asof": asof}, parse_dates=["order_ts"])
    r = _rfm(df, asof)
    bg = _fit_bg(r)
    gg = _fit_gg(r)
    print(f"BG r,α,a,b = {np.round(bg,4)}; GG p,q,γ = {np.round(gg,4)}")
    out = _score(r, bg, gg).reset_index().assign(asof_date=asof.date())
    sch, tbl = DST.split(".")
    out.to_sql(tbl, eng, schema=sch, if_exists="append", index=False)
    print(f"{len(out)} rows → {DST}; CLV total {out.clv.sum():,.0f}")


if __name__ == "__main__":
    main()
