"""
Price Estimation Service

This module implements a statistical pricing engine that infers baseline price
points for new items based on observed historical sell-through behavior. By
leveraging product attributes such as category and material composition, the
system constructs a linear model to estimate competitive list prices.

The service accepts tagged product records and returns calibrated price
recommendations, supporting consistent pricing across the catalog.
"""
import numpy as np


ATTRIBUTE_VOCABULARY = [
    "bedding",
    "bath",
    "linen",
    "cotton",
    "towel",
    "duvet",
    "pillow",
]


def normalize_attribute(raw_attribute: str) -> str:
    """Standardize attribute strings to canonical form."""
    normalized = raw_attribute.strip().lower()
    mapping = {
        "linens": "linen",
        "towels": "towel",
        "bed": "bedding",
        "pillows": "pillow",
    }
    return mapping.get(normalized, normalized)


def build_feature_vector(attributes: list[str]) -> np.ndarray:
    """Construct a feature vector from a product's attribute set."""
    canonical_attributes = {normalize_attribute(attr) for attr in attributes}
    base = 1.0
    indicator_features = [
        1.0 if attr in canonical_attributes else 0.0
        for attr in ATTRIBUTE_VOCABULARY
    ]
    return np.array([base, *indicator_features])


class PricingEngine:
    """Linear regression-based estimator for suggested list prices."""

    def __init__(self) -> None:
        self._coefficients: np.ndarray | None = None

    def train(self, historical_records: list[dict]) -> None:
        """Fit the pricing model using historical transaction data."""
        feature_matrix = np.vstack(
            [build_feature_vector(record["attributes"]) for record in historical_records]
        )
        target_prices = np.array([record["price"] for record in historical_records])
        self._coefficients, *_ = np.linalg.lstsq(feature_matrix, target_prices, rcond=None)

    def estimate(self, attributes: list[str]) -> float:
        """Generate a suggested price for a product given its attributes."""
        if self._coefficients is None:
            raise RuntimeError("Model must be trained before making predictions")
        features = build_feature_vector(attributes)
        return float(np.dot(features, self._coefficients))
