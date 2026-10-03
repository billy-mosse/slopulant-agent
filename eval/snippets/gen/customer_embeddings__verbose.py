class CustomerTasteProfileBuilder:
    """Constructs a numerical representation of each customer's purchasing behavior.

    This module generates a fixed-length feature vector per customer that captures
    their engagement intensity (recency, frequency, monetary value) and category
    preference profile. The category profile is computed as a time-decayed tally
    of purchases across predefined product groups, normalized to a probability
    distribution. The engagement metrics are transformed to stabilize variance
    and improve model compatibility.

    Input: sequence of order line items, each with customer identifier, category,
    spend amount, and recency in days.

    Output: mapping from customer ID to a 10-dimensional float vector.
    """

    PRODUCT_CATEGORIES = ["bedding", "bath", "decor", "kitchen", "furniture"]
    DECAY_TIMESCALE_DAYS = 90

    def __init__(self):
        self._category_index = {cat: idx for idx, cat in enumerate(self.PRODUCT_CATEGORIES)}

    def _assemble_profile(self, transaction_history):
        category_weights = [0.0] * len(self.PRODUCT_CATEGORIES)
        for record in transaction_history:
            if record["category"] in self._category_index:
                decay_factor = 0.5 ** (record["days_ago"] / self.DECAY_TIMESCALE_DAYS)
                category_weights[self._category_index[record["category"]]] += decay_factor

        total_weight = sum(category_weights)
        if total_weight > 0:
            category_weights = [w / total_weight for w in category_weights]

        recency_score = min(r["days_ago"] for r in transaction_history)
        frequency_score = len(transaction_history)
        monetary_score = sum(r["amount"] for r in transaction_history)

        engagement_metrics = [
            np.exp(-recency_score / 30.0),
            np.log1p(frequency_score),
            np.log1p(monetary_score),
        ]
        return np.concatenate([engagement_metrics, category_weights])

    def generate_profiles(self, order_records):
        grouped = {}
        for item in order_records:
            customer_id = item["customer_id"]
            grouped.setdefault(customer_id, []).append(item)

        return {
            cid: self._assemble_profile(history)
            for cid, history in grouped.items()
        }


if __name__ == "__main__":
    sample_data = [
        {"customer_id": "C1", "sku": "BED-001", "category": "bedding", "amount": 189.0, "days_ago": 10},
        {"customer_id": "C1", "sku": "BTH-010", "category": "bath", "amount": 34.0, "days_ago": 200},
    ]
    builder = CustomerTasteProfileBuilder()
    profiles = builder.generate_profiles(sample_data)
    print({k: np.round(v, 3).tolist() for k, v in profiles.items()})
