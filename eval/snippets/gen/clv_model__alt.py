import argparse
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import betaln, gammaln
from scipy.stats import gamma

SRC_TABLE = "orders.lines"
OUT_TABLE = "scores.clv"
FORECAST_WEEKS = 52.0
DISCOUNT_RATE = 0.10


def build_rfm(df: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Aggregate order history into RFM features (in weeks)."""
    daily = df.assign(date=df.order_ts.dt.normalize())
    agg = daily.groupby(["customer_id", "date"]).net_revenue.sum().reset_index()
    cust = agg.groupby("customer_id")
    first_order = cust.date.min()
    last_order = cust.date.max()
    freq = cust.date.count() - 1
    rec = ((last_order - first_order).dt.total_seconds() / 604800).clip(lower=0)
    age = ((cutoff - first_order).dt.total_seconds() / 604800).clip(lower=0)
    rep = agg[agg.date > agg.customer_id.map(first_order)]
    avg_spend = rep.groupby("customer_id").net_revenue.mean().reindex(freq.index).fillna(0.0)
    return pd.DataFrame({
        "customer_id": freq.index,
        "frequency": freq.values,
        "recency": rec.values,
        "T": age.values,
        "monetary": avg_spend.values
    }).set_index("customer_id")


def estimate_bgnbd(data: pd.DataFrame) -> np.ndarray:
    """Fit BG/NBD via MLE using log-parameterization."""
    x = data.frequency.values.astype(float)
    tx = data.recency.values.astype(float)
    T = data.T.values.astype(float)
    n = len(x)

    def neg_loglik(theta):
        r, alpha, a, b = np.exp(theta)
        term1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
        term2 = betaln(a, b + x) - betaln(a, b)
        term3 = -(r + x) * np.log(alpha + T)
        term4 = np.where(x > 0,
                         np.log(a) - np.log(b + x - 1) - (r + x) * np.log(alpha + tx),
                         -np.inf)
        ll = np.sum(term1 + term2 + np.logaddexp(term3, term4))
        reg = np.sum(theta ** 2) * 1e-4
        return -(ll / n) + reg

    res = minimize(neg_loglik, np.zeros(4), method="BFGS", options={"maxiter": 5000})
    return np.exp(res.x)


def expected_purchases(params, t, x, tx, T):
    """Expected number of transactions in next t weeks under BG/NBD."""
    r, alpha, a, b = params
    delta = (x > 0).astype(float)
    num = (a + b + x - 1) / (a - 1)
    base = (alpha + T) / (alpha + T + t)
    power = r + x
    # Hypergeometric approximation via series expansion for stability
    h = 1.0
    for k in range(1, 10):
        h += np.prod([(power + i - 1) * (b + x + i - 1) / ((a + b + x - 1 + i - 1) * i) for i in range(1, k + 1)]) * (1 - base) ** k
    term_a = num * (1 - base ** power * h)
    term_b = 1 + delta * a / (b + x - 1) * ((alpha + T) / (alpha + tx)) ** power
    return term_a / term_b


def survival_prob(params, x, tx, T):
    """Probability customer is still active."""
    r, alpha, a, b = params
    delta = (x > 0).astype(float)
    odds = delta * a / (b + x - 1) * ((alpha + T) / (alpha + tx)) ** (r + x)
    return 1.0 / (1.0 + odds)


def fit_gamma_gamma(data: pd.DataFrame) -> np.ndarray:
    """Estimate Gamma-Gamma parameters (p, q, γ) for monetary value."""
    sub = data[(data.frequency > 0) & (data.monetary > 0)]
    x = sub.frequency.values.astype(float)
    m = sub.monetary.values.astype(float)

    def neg_loglik(theta):
        p, q, gamma_param = np.exp(theta)
        ll = (gammaln(p * x + q) - gammaln(p * x) - gammaln(q) +
              q * np.log(gamma_param) +
              (p * x - 1) * np.log(m) +
              p * x * np.log(x) -
              (p * x + q) * np.log(gamma_param + m * x))
        return -np.mean(ll)

    res = minimize(neg_loglik, np.log([1.0, 1.0, m.mean()]), method="L-BFGS-B")
    return np.exp(res.x)


def avg_order_value(params, x, m):
    """Expected average order value given past frequency and monetary."""
    p, q, gamma_param = params
    baseline = p * gamma_param / (q - 1)
    weight = (q - 1) / (p * x + q - 1)
    return weight * baseline + (1 - weight) * np.where(x > 0, m, baseline)


def compute_clv(rfm: pd.DataFrame, bg_params, gg_params, horizon=FORECAST_WEEKS) -> pd.DataFrame:
    """Compute 12-month CLV with monthly discounting."""
    x = rfm.frequency.values.astype(float)
    tx = rfm.recency.values.astype(float)
    T = rfm.T.values.astype(float)
    m = rfm.monetary.values.astype(float)

    # Monthly forecast steps
    months = np.arange(1, int(horizon / 4.345) + 1)
    weeks = months * 4.345
    # Cumulative expected purchases up to each month
    cum = np.column_stack([expected_purchases(bg_params, w, x, tx, T) for w in weeks])
    # Discount factors
    disc = (1 + DISCOUNT_RATE) ** (-months / 12.0)

    alive = survival_prob(bg_params, x, tx, T)
    aov = avg_order_value(gg_params, x, m)
    clv_val = aov * np.sum(np.diff(cum, prepend=0, axis=1) * disc, axis=1)

    result = rfm.copy()
    result["p_alive"] = alive
    result["exp_purchases_12m"] = cum[-1]
    result["exp_aov"] = aov
    result["clv_12m"] = clv_val
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--asof", required=True)
    args = parser.parse_args()

    import sqlalchemy as sqla
    engine = sqla.create_engine(args.db)
    cutoff = pd.Timestamp(args.asof)

    df = pd.read_sql(
        f"SELECT customer_id, order_ts, net_revenue FROM {SRC_TABLE} WHERE order_ts < :asof AND status = 'completed'",
        engine, params={"asof": cutoff}, parse_dates=["order_ts"])

    rfm = build_rfm(df, cutoff)
    bg = estimate_bgnbd(rfm)
    gg = fit_gamma_gamma(rfm)

    print(f"BG/NBD: r={bg[0]:.4f}, α={bg[1]:.4f}, a={bg[2]:.4f}, b={bg[3]:.4f}")
    print(f"Gamma-Gamma: p={gg[0]:.4f}, q={gg[1]:.4f}, γ={gg[2]:.4f}")

    clv_df = compute_clv(rfm, bg, gg).reset_index()
    schema, table = OUT_TABLE.split(".")
    clv_df.to_sql(table, engine, schema=schema, if_exists="append", index=False)
    print(f"Wrote {len(clv_df)} records to {OUT_TABLE}; total CLV: ${clv_df.clv_12m.sum():,.0f}")


if __name__ == "__main__":
    main()
