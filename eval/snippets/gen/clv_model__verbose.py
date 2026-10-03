class CustomerLifetimeValueEstimator:
    """
    Computes 12-month customer lifetime value using a probabilistic transaction model
    (BG/NBD for repeat purchase behavior) combined with a conditional spend model
    (Gamma-Gamma for average order value). Designed for product teams to understand
    long-term customer value potential without exposing underlying database schema.
    """

    def __init__(self, analysis_date: pd.Timestamp, discount_rate_annual: float = 0.10):
        self.analysis_date = analysis_date
        self.discount_rate = discount_rate_annual
        self.bg_params = None
        self.gg_params = None

    def _compute_behavioral_features(self, transaction_records: pd.DataFrame) -> pd.DataFrame:
        """Derives RFM-style metrics from raw order history: frequency, recency, age, and monetary."""
        daily_revenue = transaction_records.groupby(["customer_id", transaction_records.order_ts.dt.normalize()]).net_revenue.sum()
        customer_groups = daily_revenue.groupby(level=0)
        first_purchase = customer_groups.apply(lambda x: x.index.get_level_values(1).min())
        last_purchase = customer_groups.apply(lambda x: x.index.get_level_values(1).max())
        
        features = pd.DataFrame({
            "frequency": customer_groups.size() - 1,
            "recency": (last_purchase - first_purchase).dt.total_seconds() / (7 * 86400),
            "T": (self.analysis_date - first_purchase).dt.total_seconds() / (7 * 86400),
        })
        
        repeat_transactions = daily_revenue[daily_revenue.index.get_level_values(1) > first_purchase.reindex(daily_revenue.index.get_level_values(0), method="ffill")]
        features["monetary"] = repeat_transactions.groupby(level=0).mean().reindex(features.index, fill_value=0.0)
        return features

    def _fit_transaction_model(self, behavioral_data: pd.DataFrame) -> np.ndarray:
        """Estimates BG/NBD parameters (r, α, a, b) via maximum likelihood."""
        freq, rec, age = (behavioral_data[col].values for col in ("frequency", "recency", "T"))
        def objective(log_params):
            r, alpha, a, b = np.exp(log_params)
            term1 = gammaln(r + freq) - gammaln(r) + r * np.log(alpha)
            term2 = betaln(a, b + freq) - betaln(a, b)
            term3 = -(r + freq) * np.log(alpha + age)
            term4 = np.where(freq > 0, np.log(a) - np.log(np.maximum(b + freq - 1, 1e-12)) - (r + freq) * np.log(alpha + rec), -np.inf)
            return -(np.sum(term1 + term2 + np.logaddexp(term3, term4)) / len(freq)) + 1e-3 * np.sum(np.exp(2 * log_params))
        
        result = minimize(objective, np.zeros(4), method="Nelder-Mead", options={"maxiter": 4000, "xatol": 1e-7})
        return np.exp(result.x)

    def _fit_spend_model(self, behavioral_data: pd.DataFrame) -> np.ndarray:
        """Estimates Gamma-Gamma parameters (p, q, γ) for average order value."""
        valid = behavioral_data[(behavioral_data.frequency > 0) & (behavioral_data.monetary > 0)]
        freq, monetary = valid.frequency.values, valid.monetary.values
        
        def objective(log_params):
            p, q, gamma = np.exp(log_params)
            return -np.sum(gammaln(p * freq + q) - gammaln(p * freq) - gammaln(q) + q * np.log(gamma) +
                          (p * freq - 1) * np.log(monetary) + p * freq * np.log(freq) -
                          (p * freq + q) * np.log(gamma + monetary * freq)) / len(freq)
        
        result = minimize(objective, np.log([1.0, 1.0, monetary.mean()]), method="L-BFGS-B")
        return np.exp(result.x)

    def _expected_purchases(self, horizon_weeks: float, freq: np.ndarray, recency: np.ndarray, age: np.ndarray) -> np.ndarray:
        """Predicts expected purchases over the next horizon using fitted BG/NBD model."""
        r, alpha, a, b = self.bg_params
        t = horizon_weeks
        z = t / (alpha + age + t)
        h = hyp2f1(r + freq, b + freq, a + b + freq - 1, z)
        numerator = (a + b + freq - 1) / (a - 1) * (1 - ((alpha + age) / (alpha + age + t)) ** (r + freq) * h)
        denominator = 1 + (freq > 0) * a / (b + freq - 1) * ((alpha + age) / (alpha + recency)) ** (r + freq)
        return numerator / denominator

    def _survival_probability(self, freq: np.ndarray, recency: np.ndarray, age: np.ndarray) -> np.ndarray:
        """Computes probability customer is still active given their transaction history."""
        r, alpha, a, b = self.bg_params
        odds = (freq > 0) * a / (b + freq - 1) * ((alpha + age) / (alpha + recency)) ** (r + freq)
        return 1.0 / (1.0 + odds)

    def _expected_aov(self, freq: np.ndarray, monetary: np.ndarray) -> np.ndarray:
        """Estimates expected average order value using Gamma-Gamma model."""
        p, q, gamma = self.gg_params
        population_mean = p * gamma / (q - 1)
        weight = (q - 1) / (p * freq + q - 1)
        return weight * population_mean + (1 - weight) * np.where(freq > 0, monetary, population_mean)

    def compute_clv(self, transaction_data: pd.DataFrame, horizon_weeks: float = 52.0) -> pd.DataFrame:
        """Produces customer-level CLV metrics including survival, expected purchases, AOV, and discounted CLV."""
        features = self._compute_behavioral_features(transaction_data)
        self.bg_params = self._fit_transaction_model(features)
        self.gg_params = self._fit_spend_model(features)
        
        freq, rec, age, monetary = (features[col].values for col in ("frequency", "recency", "T", "monetary"))
        months = np.arange(1, int(round(horizon_weeks / 4.345)) + 1)
        cumulative_purchases = np.stack([self._expected_purchases(k * 4.345, freq, rec, age) for k in np.r_[0, months]])
        discount_factors = (1 + self.discount_rate) ** (-months / 12.0)
        
        result = features.copy()
        result["p_alive"] = self._survival_probability(freq, rec, age)
        result["exp_purchases_12m"] = cumulative_purchases[-1]
        result["exp_aov"] = self._expected_aov(freq, monetary)
        result["clv_12m"] = result["exp_aov"] * np.sum(np.diff(cumulative_purchases, axis=0) * discount_factors[:, None], axis=0)
        return result.reset_index()
