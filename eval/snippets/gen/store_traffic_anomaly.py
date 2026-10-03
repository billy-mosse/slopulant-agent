"""
Detect anomalous hourly foot traffic in physical stores using rolling z-score.
Input: raw_table (schema.raw_foot_traffic) with columns: store_id, timestamp_hour, foot_traffic
Output: schema.anomalies_foot_traffic with columns: store_id, timestamp_hour, foot_traffic, z_score, is_anomaly
"""

import pandas as pd
from typing import Tuple

def detect_anomalies(raw_df: pd.DataFrame, window_size: int = 24, threshold: float = 2.5) -> pd.DataFrame:
    """Process foot traffic and flag anomalies per store using rolling z-score."""
    raw_df = raw_df.copy()
    raw_df.sort_values(['store_id', 'timestamp_hour'], inplace=True)
    
    # Compute rolling mean/std per store with 1-row lag (current hour not included in stats)
    raw_df['rolling_mean'] = raw_df.groupby('store_id')['foot_traffic'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=1).mean()
    )
    raw_df['rolling_std'] = raw_df.groupby('store_id')['foot_traffic'].transform(
        lambda x: x.shift(1).rolling(window=window_size, min_periods=2).std()
    )
    
    # Handle division by zero; use 1e-6 to avoid NaN/inf
    raw_df['z_score'] = (raw_df['foot_traffic'] - raw_df['rolling_mean']) / raw_df['rolling_std'].replace(0, 1e-6)
    
    raw_df['is_anomaly'] = raw_df['z_score'].abs() > threshold
    return raw_df[['store_id', 'timestamp_hour', 'foot_traffic', 'z_score', 'is_anomaly']].copy()
