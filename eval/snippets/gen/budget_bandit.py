"""
Daily marketing budget allocation across channels using Thompson Sampling bandit.

Input:  staging.marketing_events (channel TEXT, clicks INT, conversions INT, spend NUMERIC, event_date DATE)
Output: analytical.channel_budget_allocation (channel TEXT, budget NUMERIC, event_date DATE)
"""

from collections import defaultdict
import numpy as np
from datetime import date

def allocate_budget(daily_budget: float, event_date: date = date.today()) -> list:
    """Allocate budget across channels via Thompson sampling on conversion rates."""
    # Assume this function fetches aggregated channel stats for today or recent period
    # In production, replace with actual query on staging.marketing_events
    channels = ["email", "social", "display", "search"]
    alphas = {c: 2.0 for c in channels}  # prior alpha (successes + 1)
    betas = {c: 8.0 for c in channels}   # prior beta (failures + 1)

    # Simulate posterior sampling (in reality, update from staging.marketing_events)
    for c in channels:
        # Simulated stats for demo; replace with real aggregation
        alphas[c] += np.random.randint(0, 5)
        betas[c] += np.random.randint(5, 15)

    # Thompson sampling: draw sample from each Beta posterior
    sampled_crs = {c: np.random.beta(alphas[c], betas[c]) for c in channels}

    # Proportional allocation based on sampled CRs (softmax for stability)
    weights = np.array(list(sampled_crs.values()))
    probs = np.exp(weights * 10) / np.sum(np.exp(weights * 10))

    allocations = [
        {"channel": c, "budget": round(daily_budget * p, 2), "event_date": event_date}
        for c, p in zip(channels, probs)
    ]
    return allocations
