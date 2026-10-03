import argparse
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

MGN = 22.0
CST = 1.80

def _qini(df, sc="uplift"):
    d = df.sort_values(sc, ascending=False).reset_index(drop=True)
    tr = (d["treated"] == 1).to_numpy()
    cv = d["converted"].to_numpy()
    nt = np.cumsum(tr)
    nc = np.cumsum(~tr)
    yt = np.cumsum(cv * tr)
    yc = np.cumsum(cv * ~tr)
    q = yt - np.where(nc > 0, yc * nt / nc, 0.0)
    f = np.arange(1, len(d) + 1) / len(d)
    return pd.DataFrame({"frac": f, "q": q, "score": d[sc].to_numpy()})


def _auuc(cv):
    tot = cv["q"].iloc[-1]
    rnd = cv["frac"] * tot
    return float(np.trapz(cv["q"] - rnd, cv["frac"]) / max(len(cv), 1))


def _best_cut(cv, pop):
    sc = pop / len(cv)
    inc = cv["q"] * sc
    tgt = cv["frac"] * pop
    prf = inc * MGN - tgt * CST
    i = int(np.argmax(prf.to_numpy()))
    return {
        "frac": float(cv["frac"].iloc[i]),
        "thr": float(cv["score"].iloc[i]),
        "inc_conv": float(inc.iloc[i]),
        "profit": float(prf.iloc[i]),
        "n_tgt": int(tgt.iloc[i]),
    }


def _deciles(df):
    d = df.copy()
    d["dec"] = pd.qcut(d["uplift"].rank(method="first", ascending=False), 10, labels=range(1, 11))
    g = d.groupby(["dec", "treated"], observed=True)["converted"].mean().unstack()
    g.columns = ["ctrl_cr", "trt_cr"]
    g["obs_uplift"] = g["trt_cr"] - g["ctrl_cr"]
    g["pred_uplift"] = d.groupby("dec", observed=True)["uplift"].mean()
    return g


def _feat_cols(df):
    skip = {"customer_id", "campaign_id", "treated", "converted", "sent_at", "updated_at"}
    return [c for c in df.columns if c not in skip and pd.api.types.is_numeric_dtype(df[c])]


class _Tlearner:
    def __init__(self, p=None):
        p = p or dict(max_iter=300, lr=0.05, max_leaf_nodes=31,
                      min_samples_leaf=100, l2_reg=1.0, early_stopping=True,
                      validation_fraction=0.1, random_state=17)
        self.m_t = HistGradientBoostingClassifier(**p)
        self.m_c = HistGradientBoostingClassifier(**p)
        self.feat = None

    def fit(self, X, tr, y):
        self.feat = list(X.columns)
        t = tr.astype(bool).to_numpy()
        self.m_t.fit(X[t], y[t])
        self.m_c.fit(X[~t], y[~t])
        return self

    def predict(self, X):
        X = X[self.feat]
        pt = self.m_t.predict_proba(X)[:, 1]
        pc = self.m_c.predict_proba(X)[:, 1]
        return pd.DataFrame({"p_treat": pt, "p_control": pc, "uplift": pt - pc}, index=X.index)


def _load_data(engine):
    h = pd.read_sql("SELECT customer_id,campaign_id,treated,converted,sent_at FROM marketing.promo_history WHERE is_randomized=TRUE", engine, parse_dates=["sent_at"])
    f = pd.read_sql("SELECT * FROM features.customer_embeddings", engine)
    h = h.sort_values("sent_at").drop_duplicates("customer_id", keep="last")
    return h.merge(f, on="customer_id", how="inner"), f


def _split(df, frac=0.25):
    st = df["treated"].astype(str) + "_" + df["converted"].astype(str)
    return train_test_split(df, test_size=frac, stratify=st, random_state=17)


def run_train(dsn, mdl_p, hold_p, nowrite):
    eng = create_engine(dsn)
    df, f = _load_data(eng)
    cols = _feat_cols(df)
    tr, ho = _split(df)
    m = _Tlearner().fit(tr[cols], tr["treated"], tr["converted"])
    import joblib
    joblib.dump(m, mdl_p)
    sc = ho[["customer_id", "treated", "converted"]].join(m.predict(ho[cols]))
    sc.to_parquet(hold_p, index=False)
    log.info("holdout mean uplift %.4f", sc["uplift"].mean())
    all_s = f[["customer_id"]].join(m.predict(f[cols].fillna(0)))
    all_s["scored_at"] = pd.Timestamp.utcnow()
    log.info("scored %d customers; %.1f%% positive uplift", len(all_s), 100 * (all_s["uplift"] > 0).mean())
    if not nowrite:
        all_s.to_sql("promo_uplift", eng, schema="marketing", if_exists="replace", index=False)


def run_eval(hold_p, pop):
    df = pd.read_parquet(hold_p)
    cv = _qini(df)
    log.info("AUUC: %.4f", _auuc(cv))
    log.info("Qini area: %.2f", np.trapz(cv["q"] - cv["frac"] * cv["q"].iloc[-1], cv["frac"]))
    print(_deciles(df).round(4).to_string())
    cut = _best_cut(cv, pop)
    log.info("target top %.1f%% (thr %.4f): %d cust, +%0.f conv, profit $%0.f",
             100 * cut["frac"], cut["thr"], cut["n_tgt"], cut["inc_conv"], cut["profit"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "eval"], required=True)
    p.add_argument("--dsn")
    p.add_argument("--model-path", default="uplift_tlearner.joblib")
    p.add_argument("--holdout-path", default="holdout_scored.parquet")
    p.add_argument("--population", type=int, default=1_200_000)
    p.add_argument("--no-write", action="store_true")
    a = p.parse_args()
    if a.mode == "train":
        run_train(a.dsn, a.model_path, a.holdout_path, a.no_write)
    else:
        run_eval(a.holdout_path, a.population)


if __name__ == "__main__":
    main()
