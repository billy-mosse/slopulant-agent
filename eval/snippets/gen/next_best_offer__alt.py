import numpy as np
import pandas as pd
import sqlalchemy as sa
from typing import List, Tuple, Optional
from scipy.linalg import solve

ARMS = ["free_shipping", "pct10_bedding", "bundle_discount", "none"]

class DisjointLinear:
    def __init__(self, dim: int, arms: List[str], exploration: float = 0.5, reg: float = 1.0):
        self.labels = list(arms)
        self.d = dim
        self.k = len(arms)
        self.beta = exploration
        self.reg = reg
        self._A = np.stack([np.eye(dim) * reg for _ in range(self.k)])
        self._c = np.zeros((self.k, dim))
        self._invA = np.linalg.inv(self._A)

    def fit_batch(self, contexts: np.ndarray, selections: np.ndarray, outcomes: np.ndarray) -> None:
        for idx, label in enumerate(self.labels):
            sel = selections == idx
            if not np.any(sel):
                continue
            X = contexts[sel]
            y = outcomes[sel]
            self._A[idx] += X.T @ X
            self._c[idx] += X.T @ y
        self._invA = np.linalg.inv(self._A)

    def _coefficients(self) -> np.ndarray:
        return np.array([solve(self._A[i], self._c[i], assume_a='pos') for i in range(self.k)])

    def evaluate(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        theta = self._coefficients()
        pred = X @ theta.T
        var = np.array([X @ self._invA[i] @ X.T for i in range(self.k)]).diagonal(axis1=1, axis2=2).T
        ucb = pred + self.beta * np.sqrt(np.maximum(var, 0))
        return pred, ucb

    def recommend(self, X: np.ndarray) -> np.ndarray:
        _, ucb = self.evaluate(X)
        return np.argmax(ucb, axis=1)

    def persist(self, dest: str) -> None:
        np.savez(dest, A=self._A, c=self._c, beta=self.beta, arms=np.array(self.labels))

    @classmethod
    def restore(cls, src: str) -> "DisjointLinear":
        data = np.load(src, allow_pickle=False)
        m = cls(data["A"].shape[1], list(data["arms"]), float(data["beta"]))
        m._A, m._c = data["A"], data["c"]
        m._invA = np.linalg.inv(m._A)
        return m


def prepare_features(conn) -> pd.DataFrame:
    """Construct feature matrix from embeddings, segment dummies, and uplift predictions."""
    emb = pd.read_sql(
        "SELECT * FROM features.customer_embeddings "
        "WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM features.customer_embeddings)",
        conn
    ).set_index("customer_id")
    emb = emb.filter(like="emb_")

    seg = pd.read_sql("SELECT customer_id, segment_name FROM marketing.segments", conn).set_index("customer_id")
    seg_oh = pd.get_dummies(seg.segment_name, prefix="seg", dtype=float)

    upl = pd.read_sql("SELECT customer_id, promo_type, predicted_uplift FROM marketing.promo_uplift", conn)
    upl = upl.pivot(index="customer_id", columns="promo_type", values="predicted_uplift")
    upl.columns = [f"uplift_{c}" for c in upl.columns]

    feat = emb.join(seg_oh, how="left").join(upl, how="left").fillna(0.0)
    norm = np.linalg.norm(feat.values, axis=1, keepdims=True)
    feat = feat.div(np.where(norm < 1e-9, 1.0, norm), axis=0)
    feat["bias"] = 1.0
    return feat


DISCOUNT_ARMS = {"pct10_bedding", "bundle_discount"}
MAX_DISCOUNT_PER_WEEK = 2
UPLIFT_COLS = {
    "free_shipping": "uplift_free_shipping",
    "pct10_bedding": "uplift_pct10_bedding",
    "bundle_discount": "uplift_bundle_discount"
}


def eligibility(ctx: pd.DataFrame, weekly_discounts: pd.Series) -> np.ndarray:
    n = len(ctx)
    ok = np.ones((n, len(ARMS)), dtype=bool)
    for i, arm in enumerate(ARMS):
        col = UPLIFT_COLS.get(arm)
        if col and col in ctx.columns:
            ok[:, i] &= ctx[col].values >= 0
        if arm in DISCOUNT_ARMS:
            cnt = weekly_discounts.reindex(ctx.index).fillna(0).values
            ok[:, i] &= cnt < MAX_DISCOUNT_PER_WEEK
    ok[:, ARMS.index("none")] = True
    return ok


def select_arm(model: DisjointLinear, X: np.ndarray, mask: np.ndarray, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    _, ucb = model.evaluate(X)
    ucb = np.where(mask, ucb, -np.inf)
    greedy = ucb.argmax(axis=1)
    allowed = mask.sum(axis=1, keepdims=True)
    base = mask * (0.05 / allowed)
    base[np.arange(len(X)), greedy] += 0.95
    cum = np.cumsum(base, axis=1)
    r = rng.random(len(X))[:, None]
    chosen = (cum > r).argmax(axis=1)
    return chosen, base[np.arange(len(X)), chosen]


def run_policy(dsn: str, alpha: float = 0.5, seed: int = 0, dry: bool = False) -> None:
    engine = sa.create_engine(dsn)
    rng = np.random.default_rng(seed)

    ctx = prepare_features(engine)
    X = ctx.values
    try:
        policy = DisjointLinear.restore("linucb_state.npz")
        policy.beta = alpha
    except FileNotFoundError:
        policy = DisjointLinear(X.shape[1], ARMS, alpha)
        logger = __import__("logging").getLogger(__name__)
        logger.warning("Fresh model initialized with d=%d", X.shape[1])

    hist = pd.read_sql(
        f"SELECT customer_id, arm, reward FROM marketing.offers "
        "WHERE decided_at >= CURRENT_DATE - INTERVAL '7 days'", engine)
    pos = hist[hist.reward.notna() & hist.customer_id.isin(ctx.index)]
    if len(pos):
        policy.fit_batch(
            ctx.loc[pos.customer_id].values,
            pos.arm.map(dict(zip(ARMS, range(len(ARMS))))).values,
            pos.reward.values
        )

    weekly = hist[hist.arm.isin(DISCOUNT_ARMS)].groupby("customer_id").size()
    elig = eligibility(ctx, weekly)
    idx, prop = select_arm(policy, X, elig, rng)

    result = pd.DataFrame({
        "customer_id": ctx.index,
        "arm": [ARMS[i] for i in idx],
        "propensity": np.round(prop, 5),
        "ucb_score": np.round(policy.evaluate(X)[1][np.arange(len(X)), idx], 5),
        "n_allowed_arms": elig.sum(axis=1),
        "reward": np.nan,
        "decided_at": pd.Timestamp.utcnow(),
        "policy_version": f"linucb_a{alpha}"
    })

    if not dry:
        result.to_sql("offers", engine, schema="marketing", if_exists="append", index=False)
        policy.persist("linucb_state.npz")
