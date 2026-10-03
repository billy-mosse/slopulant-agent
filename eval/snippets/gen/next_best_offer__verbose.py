class ContextualBanditPolicy:
    """
    Implements a contextual bandit decision engine that recommends personalized promotions
    based on customer features, historical behavior, and business constraints.
    
    The system uses a disjoint linear UCB algorithm (LinUCB) to balance exploration and
    exploitation across offer options while respecting merchandising policies: customers
    receive at most two discount offers per week, offers with negative projected uplift
    are suppressed, and fallback to a default "no promotion" option is guaranteed.
    
    Each interaction is logged with decision propensity scores to support offline
    evaluation methods including inverse propensity weighting and doubly robust estimation.
    """

    ARMS = ["free_shipping", "pct10_bedding", "bundle_discount", "no_offer"]
    DISCOUNT_TYPES = {"pct10_bedding", "bundle_discount"}
    MAX_DISCOUNT_CAPACITY = 2

    def __init__(self, feature_dim: int, exploration_factor: float = 0.5, regularization: float = 1.0) -> None:
        self.feature_dimension = feature_dim
        self.exploration_weight = exploration_factor
        self.covariance_matrix = np.stack(
            [regularization * np.eye(feature_dim) for _ in self.ARMS]
        )
        self.response_vector = np.zeros((len(self.ARMS), feature_dim))
        self._inverse_covariance = np.linalg.inv(self.covariance_matrix)

    def train(self, contexts: np.ndarray, selections: np.ndarray, outcomes: np.ndarray) -> None:
        """Incrementally update model parameters using observed triples (context, selection, outcome)."""
        for arm_index, arm in enumerate(self.ARMS):
            relevant_indices = selections == arm_index
            if not np.any(relevant_indices):
                continue
            feature_batch = contexts[relevant_indices]
            reward_batch = outcomes[relevant_indices]
            self.covariance_matrix[arm_index] += feature_batch.T @ feature_batch
            self.response_vector[arm_index] += feature_batch.T @ reward_batch
        self._inverse_covariance = np.linalg.inv(self.covariance_matrix)

    @property
    def coefficient_estimates(self) -> np.ndarray:
        """Compute ridge regression coefficients for each arm: theta = A^{-1} b."""
        return np.einsum("kij,kj->ki", self._inverse_covariance, self.response_vector)

    def evaluate_proposals(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate predicted rewards and uncertainty-adjusted scores for all arms.
        Returns (point_estimate_matrix, ucb_score_matrix) each of shape (n_customers, n_arms).
        """
        expected_returns = features @ self.coefficient_estimates.T
        uncertainty = np.sqrt(
            np.einsum("ni,kij,nj->nk", features, self._inverse_covariance, features).clip(min=0.0)
        )
        return expected_returns, expected_returns + self.exploration_weight * uncertainty

    def select_arms(self, features: np.ndarray, eligibility: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Choose offers using epsilon-greedy strategy over UCB scores, respecting eligibility constraints.
        Returns (arm_indices, propensities).
        """
        _, scores = self.evaluate_proposals(features)
        adjusted_scores = np.where(eligibility, scores, -np.inf)
        greedy_selections = adjusted_scores.argmax(axis=1)
        eligible_counts = eligibility.sum(axis=1)
        base_prob = 0.05 / eligible_counts[:, None]
        action_probabilities = eligibility.astype(float) * base_prob
        action_probabilities[np.arange(len(features)), greedy_selections] += 0.95
        cumulative_probs = np.cumsum(action_probabilities, axis=1)
        random_draws = np.random.random((len(features), 1))
        selected_arms = (cumulative_probs > random_draws).argmax(axis=1)
        return selected_arms, action_probabilities[np.arange(len(features)), selected_arms]

    def enforce_business_rules(
        self,
        customer_features: pd.DataFrame,
        weekly_discount_count: pd.Series,
    ) -> np.ndarray:
        """
        Construct eligibility matrix using uplift projections and frequency constraints.
        Row i, column j is True iff customer i may receive arm j under policy rules.
        """
        n_customers = len(customer_features)
        eligibility = np.ones((n_customers, len(self.ARMS)), dtype=bool)
        
        for idx, arm in enumerate(self.ARMS):
            uplift_key = f"uplift_{arm}" if arm != "no_offer" else None
            if uplift_key and uplift_key in customer_features:
                eligibility[:, idx] &= customer_features[uplift_key].to_numpy() >= 0.0
            if arm in self.DISCOUNT_TYPES:
                weekly_count = weekly_discount_count.reindex(customer_features.index, fill_value=0)
                eligibility[:, idx] &= weekly_count.to_numpy() < self.MAX_DISCOUNT_CAPACITY
        
        # Always allow the fallback option
        eligibility[:, self.ARMS.index("no_offer")] = True
        return eligibility

    def persist_state(self, filepath: str) -> None:
        """Serialize model state to disk for persistence across deployments."""
        np.savez(
            filepath,
            covariance=self.covariance_matrix,
            response=self.response_vector,
            exploration=self.exploration_weight,
            arms=np.array(self.ARMS),
        )

    @classmethod
    def restore(cls, filepath: str) -> "ContextualBanditPolicy":
        """Reconstruct policy instance from saved state."""
        data = np.load(filepath, allow_pickle=False)
        policy = cls(
            feature_dim=data["covariance"].shape[1],
            exploration_factor=float(data["exploration"]),
        )
        policy.covariance_matrix = data["covariance"]
        policy.response_vector = data["response"]
        policy._inverse_covariance = np.linalg.inv(policy.covariance_matrix)
        policy.ARMS = list(data["arms"])
        return policy
