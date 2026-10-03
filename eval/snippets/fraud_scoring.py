"""Score orders for payment fraud before fulfillment."""
import math


def features(order):
    return [
        float(order["billing_country"] != order["shipping_country"]),
        math.log1p(order["total"]),
        float(order["account_age_days"] < 2),
        order["failed_payment_attempts"],
    ]


def p_fraud(order, w=(1.5, 0.4, 1.2, 0.8), b=-5.0):
    z = b + sum(wi * xi for wi, xi in zip(w, features(order)))
    return 1 / (1 + math.exp(-z))
