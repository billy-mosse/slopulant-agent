import argparse
import logging
import numpy as np
import pandas as pd
import sqlalchemy as sa

log = logging.getLogger("opt_send")

TBL_EVENTS = "marketing.email_events"
TBL_OUT = "marketing.send_times"
H_BKT = 3
N_BKT = 7 * 24 // H_BKT
MIN_SND = 8
KAP_CAP = 50.0
N_DRAWS = 1
LOOKBACK = 365
TZ_DEF = "America/New_York"

def exec(engine, asof, seed, dry):
    rng = np.random.default_rng(seed)
    sql = f"""
        SELECT customer_id, campaign_id, sent_ts_utc, opened_ts_utc, customer_tz, segment_name
        FROM {TBL_EVENTS}
        WHERE event_type = 'send' AND sent_ts_utc >= %(start)s AND sent_ts_utc < %(asof)s
    """
    df = pd.read_sql(sql, engine, params={"start": asof - pd.Timedelta(days=LOOKBACK), "asof": asof},
                     parse_dates=["sent_ts_utc", "opened_ts_utc"])
    log.info(f"loaded {len(df):,} sends for {df.customer_id.nunique():,} customers")

    df["customer_tz"] = df["customer_tz"].fillna(TZ_DEF)
    bad = ~df["customer_tz"].str.contains("/", regex=False)
    if bad.any():
        log.warning(f"{bad.sum():,} malformed tz, fallback to {TZ_DEF}")
        df.loc[bad, "customer_tz"] = TZ_DEF
    df["sent_ts_utc"] = df["sent_ts_utc"].dt.tz_localize("UTC").dt.tz_convert(df["customer_tz"])
    df["dow"] = df["sent_ts_utc"].dt.dayofweek
    df["hr"] = df["sent_ts_utc"].dt.hour
    df["bkt"] = df["dow"] * (24 // H_BKT) + df["hr"] // H_BKT
    df["open"] = ((df["opened_ts_utc"].notna()) &
                  ((df["opened_ts_utc"].dt.tz_localize("UTC") - df["sent_ts_utc"].dt.tz_localize("UTC"))
                   <= pd.Timedelta(hours=48))).astype(int)

    cb = df.groupby(["customer_id", "bkt"]).agg(s=("open", "size"), o=("open", "sum")).reset_index()
    gl = cb.groupby("bkt").agg(s=("s", "sum"), o=("o", "sum"))
    gl["mu"] = (gl["o"] + 1) / (gl["s"] + 2)
    cr = cb[cb.s >= 3].assign(r=lambda x: x.o / x.s)
    vb = cr.groupby("bkt")["r"].var()
    nb = cr.groupby("bkt")["s"].mean()
    pr = gl.join(vb.rename("v")).join(nb.rename("n"))
    bv = (pr["v"].fillna(0) - pr["mu"] * (1 - pr["mu"]) / pr["n"].fillna(3)).clip(lower=1e-5)
    kp = (pr["mu"] * (1 - pr["mu"]) / bv - 1).clip(lower=2.0, upper=KAP_CAP)
    pr["a"] = pr["mu"] * kp
    pr["b"] = (1 - pr["mu"]) * kp
    pr = pr.reindex(range(N_BKT))
    pr["a"] = pr["a"].fillna(pr["a"].median())
    pr["b"] = pr["b"].fillna(pr["b"].median())

    cust = df.groupby("customer_id").agg(
        ts=("open", "size"), to=("open", "sum"),
        tz=("customer_tz", "last"), seg=("segment_name", "last")
    )
    grid = pd.MultiIndex.from_product([cust.index, range(N_BKT)], names=["cid", "bkt"])
    pt = cb.set_index(["customer_id", "bkt"]).reindex(grid, fill_value=0).reset_index()
    pt["a"] = pr["a"].to_numpy()[pt["bkt"]] + pt["o"]
    pt["b"] = pr["b"].to_numpy()[pt["bkt"]] + pt["s"] - pt["o"]
    pt["d"] = rng.beta(pt["a"], pt["b"], size=(N_DRAWS, len(pt))).mean(axis=0)
    mx = pt.groupby("cid")["d"].idxmax()
    pk = pt.loc[mx, ["cid", "bkt", "d"]].set_index("cid")
    cust = cust.join(pk.rename(columns={"bkt": "b", "d": "d"}))
    cust["m"] = "thompson"

    sg = (pt.merge(cust[["seg"]], left_on="cid", right_index=True)
          .groupby(["seg", "b"])[["a", "b"]].sum())
    sg["mu"] = sg["a"] / (sg["a"] + sg["b"])
    sb = sg.reset_index().loc[lambda x: x.groupby("seg")["mu"].idxmax()].set_index("seg")
    th = cust["ts"] < MIN_SND
    cust.loc[th, "b"] = cust.loc[th, "seg"].map(sb["b"]).fillna(gl["mu"].idxmax())
    cust.loc[th, "d"] = cust.loc[th, "seg"].map(sb["mu"])
    cust.loc[th, "m"] = "segfb"
    log.info(f"{th.sum():,}/{len(cust):,} ({th.mean():.1%}) use seg fallback")

    cust["dow"] = cust["b"] // (24 // H_BKT)
    cust["hr"] = (cust["b"] % (24 // H_BKT)) * H_BKT + H_BKT // 2
    out = cust.reset_index()[["cid", "tz", "dow", "hr", "d", "m", "ts"]]
    out["at"] = asof

    if dry:
        log.info(f"dry run: {len(out):,} rows skipped")
    else:
        sch, tbl = TBL_OUT.split(".")
        out.to_sql(tbl, engine, schema=sch, if_exists="replace", index=False, chunksize=50_000)
        log.info(f"wrote {len(out):,} rows to {TBL_OUT}")
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsn", required=True)
    ap.add_argument("--asof", default=pd.Timestamp.utcnow().strftime("%Y-%m-%d"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    exec(sa.create_engine(args.dsn), pd.Timestamp(args.asof), args.seed, args.dry)

if __name__ == "__main__":
    main()
