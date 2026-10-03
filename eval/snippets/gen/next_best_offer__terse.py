import numpy as np
import pandas as pd
import sqlalchemy as sa

ARMS = ["free_shipping", "pct10_bedding", "bundle_discount", "none"]
DISCOUNT = {"pct10_bedding", "bundle_discount"}
MAX_DISC = 2
EPS = 0.05
UPLIFT_MAP = {"free_shipping": "uplift_free_shipping", "pct10_bedding": "uplift_pct10_bedding", "bundle_discount": "uplift_bundle_discount"}

def _inv_batch(A):
    return np.linalg.inv(A)

def _ucb_scores(X, theta, Ainv, alpha):
    m = X @ theta.T
    w = np.sqrt(np.einsum("ni,nij,nj->n", X, Ainv, X).clip(0))
    return m, m + alpha * w[:, None]

def _mask_allowed(ctx, recent, arms):
    n = len(ctx)
    msk = np.ones((n, len(arms)), dtype=bool)
    for i, a in enumerate(arms):
        col = UPLIFT_MAP.get(a)
        if col and col in ctx:
            msk[:, i] &= ctx[col].to_numpy() >= 0
        if a in DISCOUNT:
            msk[:, i] &= recent.reindex(ctx.index).fillna(0).to_numpy() < MAX_DISC
    msk[:, arms.index("none")] = True
    return msk

def _select(msk, ucb, rng):
    ucb = np.where(msk, ucb, -np.inf)
    g = ucb.argmax(axis=1)
    c = msk.sum(axis=1)
    prb = msk * (EPS / c[:, None])
    prb[np.arange(len(msk)), g] += 1 - EPS
    u = rng.random(len(msk))[:, None]
    return (prb.cumsum(axis=1) < u).sum(axis=1), prb[np.arange(len(msk)), g], ucb[np.arange(len(msk)), g]

class Bandit:
    def __init__(self, d, arms=ARMS, alpha=0.5, lam=1.0):
        self.arms = list(arms)
        self.d = d
        self.alpha = alpha
        self.A = np.stack([np.eye(d) * lam for _ in self.arms])
        self.b = np.zeros((len(self.arms), d))
        self.Ainv = _inv_batch(self.A)

    def fit(self, X, idx, r):
        for k in range(len(self.arms)):
            sel = idx == k
            if not sel.any():
                continue
            Xk = X[sel]
            self.A[k] += Xk.T @ Xk
            self.b[k] += Xk.T @ r[sel]
        self.Ainv = _inv_batch(self.A)

    @property
    def theta(self):
        return np.einsum("kij,kj->ki", self.Ainv, self.b)

    def predict(self, X):
        _, ucb = _ucb_scores(X, self.theta, self.Ainv, self.alpha)
        return ucb.argmax(axis=1)

    def dump(self, pth):
        np.savez(pth, A=self.A, b=self.b, alpha=self.alpha, arms=np.array(self.arms))

    @classmethod
    def load(cls, pth):
        z = np.load(pth, allow_pickle=False)
        m = cls(z["A"].shape[1], list(z["arms"]), float(z["alpha"]))
        m.A, m.b = z["A"], z["b"]
        m.Ainv = _inv_batch(m.A)
        return m

def _build_ctx(engine):
    emb = pd.read_sql("SELECT * FROM features.customer_embeddings WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM features.customer_embeddings)", engine).set_index("customer_id")
    emb = emb[[c for c in emb.columns if c.startswith("emb_")]]
    seg = pd.read_sql("SELECT customer_id, segment_name FROM marketing.segments", engine).set_index("customer_id")
    seg = pd.get_dummies(seg.segment_name, prefix="seg", dtype=float)
    upl = pd.read_sql("SELECT customer_id, promo_type, predicted_uplift FROM marketing.promo_uplift", engine)
    upl = upl.pivot_table(index="customer_id", columns="promo_type", values="predicted_uplift", aggfunc="mean").add_prefix("uplift_")
    ctx = emb.join(seg, how="left").join(upl, how="left").fillna(0.0)
    nrm = np.linalg.norm(ctx.to_numpy(), axis=1).clip(1e-9)
    ctx = ctx.div(nrm, axis=0)
    ctx["bias"] = 1.0
    return ctx

def run(dsn, alpha=0.5, seed=0, dry=False):
    eng = sa.create_engine(dsn)
    rng = np.random.default_rng(seed)
    ctx = _build_ctx(eng)
    X = ctx.to_numpy()
    try:
        mdl = Bandit.load("state.npz")
        mdl.alpha = alpha
    except FileNotFoundError:
        mdl = Bandit(X.shape[1], alpha=alpha)
    hist = pd.read_sql("SELECT customer_id, arm, reward FROM marketing.offers WHERE decided_at >= CURRENT_DATE - INTERVAL '7 days'", eng)
    seen = hist[hist.reward.notna() & hist.customer_id.isin(ctx.index)]
    if len(seen):
        idx = seen.arm.map(dict(zip(ARMS, range(len(ARMS))))).to_numpy()
        mdl.fit(ctx.loc[seen.customer_id].to_numpy(), idx, seen.reward.to_numpy())
    disc_cnt = hist[hist.arm.isin(DISCOUNT)].groupby("customer_id").size()
    upl = pd.read_sql("SELECT customer_id, promo_type, predicted_uplift FROM marketing.promo_uplift", eng)
    upl = upl.pivot_table(index="customer_id", columns="promo_type", values="predicted_uplift").add_prefix("uplift_").reindex(ctx.index).fillna(0.0)
    msk = _mask_allowed(upl, disc_cnt, ARMS)
    idx, prop, sc = _select(msk, mdl.predict(ctx), rng)
    out = pd.DataFrame({
        "customer_id": ctx.index,
        "arm": [ARMS[i] for i in idx],
        "propensity": np.round(prop, 5),
        "ucb_score": np.round(sc, 5),
        "n_allowed_arms": msk.sum(axis=1),
        "reward": np.nan,
        "decided_at": pd.Timestamp.utcnow(),
        "policy_version": f"bandit_a{alpha}",
    })
    if not dry:
        out.to_sql("offers", eng, schema="marketing", if_exists="append", index=False)
        mdl.dump("state.npz")
