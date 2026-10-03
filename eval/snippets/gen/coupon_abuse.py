"""
Flag accounts that abuse sign-up coupons by sharing devices, addresses, or payment cards.

Input: raw_data.signups, raw_data.users, raw_data.payments
Output: flagged_accounts.abuse_signals
"""

import pandas as pd

def detect_coupon_abuse(signups: pd.DataFrame, users: pd.DataFrame, payments: pd.DataFrame) -> pd.DataFrame:
    """
    Identify accounts likely sharing devices, addresses, or payment cards to abuse sign-up coupons.
    Returns a DataFrame with account_id, abuse_type, and confidence_score.
    """
    # Merge relevant tables
    merged = signups.merge(users[['user_id', 'device_id', 'address_id']], on='user_id', how='left')
    merged = merged.merge(payments[['payment_id', 'card_last4', 'billing_address_id']], 
                          left_on='payment_id', right_index=True, how='left')
    
    # Group by shared attributes
    device_groups = merged.groupby('device_id').filter(lambda x: len(x) > 1)
    address_groups = merged.groupby('address_id').filter(lambda x: len(x) > 1)
    card_groups = merged.groupby('card_last4').filter(lambda x: len(x) > 1)
    
    # Build abuse signals
    signals = []
    for _, group in device_groups.groupby('device_id'):
        for idx, row in group.iterrows():
            signals.append({'account_id': row['account_id'], 'abuse_type': 'shared_device', 'confidence_score': 0.85})
    for _, group in address_groups.groupby('address_id'):
        for idx, row in group.iterrows():
            signals.append({'account_id': row['account_id'], 'abuse_type': 'shared_address', 'confidence_score': 0.90})
    for _, group in card_groups.groupby('card_last4'):
        for idx, row in group.iterrows():
            signals.append({'account_id': row['account_id'], 'abuse_type': 'shared_card', 'confidence_score': 0.95})
    
    return pd.DataFrame(signals) if signals else pd.DataFrame(columns=['account_id', 'abuse_type', 'confidence_score'])
