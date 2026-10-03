"""
Estimate supplier lead times per vendor from purchase-order history.

Input: po_history.vendor_id, po_history.order_date, po_history.delivery_date
Output: vendor_lead_time_estimates.vendor_id, median_lead_time_days, p90_lead_time_days
"""

from datetime import date
from statistics import median
from typing import Dict, List, Tuple

def estimate_lead_times(po_records: List[Tuple[str, date, date]]) -> Dict[str, Dict[str, float]]:
    """Compute median and p90 lead times (in days) per vendor."""
    vendor_days: Dict[str, List[float]] = {}
    for vendor_id, order_date, delivery_date in po_records:
        if vendor_id not in vendor_days:
            vendor_days[vendor_id] = []
        delta = (delivery_date - order_date).days
        if delta >= 0:
            vendor_days[vendor_id].append(delta)
    
    estimates = {}
    for vendor_id, days_list in vendor_days.items():
        if not days_list:
            continue
        sorted_days = sorted(days_list)
        n = len(sorted_days)
        median_days = median(sorted_days)
        p90_idx = int(0.9 * (n - 1))
        p90_days = sorted_days[p90_idx] if n > 1 else sorted_days[0]
        estimates[vendor_id] = {
            "median_lead_time_days": round(median_days, 2),
            "p90_lead_time_days": round(p90_days, 2),
        }
    return estimates
