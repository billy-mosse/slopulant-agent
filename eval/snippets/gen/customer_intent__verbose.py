class ConversionProbabilityEstimator:
    """
    Converts behavioral session data into conversion likelihood scores using a
    lightweight linear model. This estimator synthesizes session-level signals
   —including view count, cart interaction frequency, total engagement duration,
    and aggregate product preferences (derived from mean-pooled embeddings)—into
    a single probability estimate for whether the session will culminate in a
    purchase.

    The model is designed for rapid inference and relies on shallow features
    without requiring complex feature engineering during deployment.
    """

    def __init__(self, weight_vector: np.ndarray, bias: float = -3.0):
        self._weights = np.asarray(weight_vector, dtype=np.float32)
        self._bias = bias

    def compute_session_signature(self, behavioral_trace: list[dict], item_representation_store: dict[str, np.ndarray]) -> np.ndarray:
        """
        Builds a unified feature vector representing the session's narrative.

        Interprets viewing events to extract engagement metrics and aggregates
        semantic product preferences using pretrained item embeddings.
        """
        view_events = [step for step in behavioral_trace if step.get("event") == "view"]
        cart_interactions = sum(1 for step in behavioral_trace if step.get("event") == "add_to_cart")
        total_dwell = sum(step.get("dwell_s", 0.0) for step in view_events)

        viewing_embeddings = [
            item_representation_store[step["sku"]]
            for step in view_events
            if step.get("sku") in item_representation_store
        ]
        embedding_summary = (
            np.mean(viewing_embeddings, axis=0)
            if viewing_embeddings
            else np.zeros(128)
        )

        base_metrics = [
            float(len(view_events)),
            float(cart_interactions),
            np.log1p(total_dwell)
        ]
        return np.concatenate([base_metrics, embedding_summary], dtype=np.float32)

    def infer(self, session_trace: list[dict], product_embeddings: dict[str, np.ndarray]) -> float:
        """
        Returns the predicted probability that the session terminates in conversion.
        """
        features = self.compute_session_signature(session_trace, product_embeddings)
        logit = np.dot(features, self._weights) + self._bias
        return 1.0 / (1.0 + np.exp(-logit))
