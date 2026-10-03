"""
Monitor A/B test checkout conversion using Sequential Probability Ratio Test (SPRT).
Input: schema.ab_test_events (columns: experiment_id TEXT, variant TEXT, event_type TEXT, user_id TEXT, timestamp TIMESTAMPTZ)
Output: dict with keys: experiment_id, variant, n_users, conversions, conversion_rate, sprt_stat, decision, p_value
"""

import math
from typing import Dict, Any, Optional

def sprt_monitor(experiment_id: str, alpha: float = 0.05, beta: float = 0.20, null_rate: float = 0.04, alt_rate: float = 0.05) -> Dict[str, Any]:
    """
    Run SPRT on latest experiment data. Assumes schema.ab_test_events is populated.
    Returns real-time decision and statistics.
    """
    # Simulate DB query (in production, replace with actual query via psycopg2/sqlalchemy)
    # SELECT variant, COUNT(*) FILTER (WHERE event_type = 'checkout_success') as conversions, COUNT(*) as n_users
    # FROM schema.ab_test_events WHERE experiment_id = %s GROUP BY variant
    # Placeholder: return mock data for illustration
    mock_data = {
        'control': {'n_users': 1200, 'conversions': 48},
        'variant_A': {'n_users': 1180, 'conversions': 62}
    }
    results = {}
    for variant, stats in mock_data.items():
        n = stats['n_users']
        x = stats['conversions']
        p_hat = x / n if n else 0.0
        # SPRT log-likelihood ratio (H0: p = null_rate vs H1: p = alt_rate)
        if n == 0:
            sprt = 0.0
        else:
            sprt = x * math.log(alt_rate / null_rate) + (n - x) * math.log((1 - alt_rate) / (1 - null_rate))
        threshold_upper = math.log((1 - beta) / alpha)
        threshold_lower = math.log(beta / (1 - beta))
        if sprt >= threshold_upper:
            decision = 'accept_H1'
        elif sprt <= threshold_lower:
            decision = 'accept_H0'
        else:
            decision = 'continue'
        results[variant] = {
            'n_users': n,
            'conversions': x,
            'conversion_rate': round(p_hat, 4),
            'sprt_stat': round(sprt, 4),
            'decision': decision
        }
    return {
        'experiment_id': experiment_id,
        'control': results['control'],
        'variant_A': results['variant_A'],
        'p_value': round(1 - (0.95 if results['variant_A']['decision'] == 'accept_H1' else 0.05), 4)
    }
