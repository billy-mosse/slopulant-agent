import numpy as np

SECTIONS = ["bedding", "bath", "decor", "kitchen", "furniture"]
DECAY_RATE = 0.0077  # corresponds to ~90-day half-life

def summarize_profile(transactions):
    weights = np.zeros(len(SECTIONS))
    earliest = float("inf")
    total_spend = 0.0
    
    for t in transactions:
        if t["category"] in SECTIONS:
            idx = SECTIONS.index(t["category"])
            decay = np.exp(-DECAY_RATE * t["days_ago"])
            weights[idx] += decay
        earliest = min(earliest, t["days_ago"])
        total_spend += t["amount"]
    
    if weights.sum() > 0:
        weights /= weights.sum()
    
    recency_score = np.exp(-earliest / 30.0)
    freq_score = np.log1p(len(transactions))
    spend_score = np.log1p(total_spend)
    
    return np.hstack([[recency_score, freq_score, spend_score], weights])


def generate_features(raw_orders):
    grouped = {}
    for record in raw_orders:
        cid = record["customer_id"]
        grouped.setdefault(cid, []).append(record)
    
    return {cust_id: summarize_profile(txns) for cust_id, txns in grouped.items()}


if __name__ == "__main__":
    sample = [
        {"customer_id": "C1", "sku": "BED-001", "category": "bedding", "amount": 189.0, "days_ago": 10},
        {"customer_id": "C1", "sku": "BTH-010", "category": "bath", "amount": 34.0, "days_ago": 200},
    ]
    result = generate_features(sample)
    print({k: np.round(v, 3).tolist() for k, v in result.items()})
