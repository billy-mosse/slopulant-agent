"""
Reconcile gift-card balances between the ledger and payment processor.

Input: ledger.gift_card_transactions, payment_processor.gift_card_balances
Output: reconciliation.gift_card_balance_diffs (with columns: gift_card_id, ledger_balance, processor_balance, difference, status)
"""

import pandas as pd
from typing import Tuple

def reconcile_gift_card_balances(ledger_df: pd.DataFrame, processor_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reconcile gift-card balances between ledger and payment processor.
    
    Args:
        ledger_df: DataFrame from ledger.gift_card_transactions
        processor_df: DataFrame from payment_processor.gift_card_balances
    
    Returns:
        DataFrame with balance differences and status for reconciliation
    """
    # Aggregate ledger balances by gift_card_id (sum of net transactions)
    ledger_balances = ledger_df.groupby('gift_card_id')['amount'].sum().reset_index()
    ledger_balances.columns = ['gift_card_id', 'ledger_balance']
    
    # Align processor balances (ensure numeric)
    processor_balances = processor_df[['gift_card_id', 'balance']].copy()
    processor_balances.columns = ['gift_card_id', 'processor_balance']
    processor_balances['processor_balance'] = pd.to_numeric(processor_balances['processor_balance'], errors='coerce').fillna(0)
    
    # Merge and compute differences
    merged = ledger_balances.merge(processor_balances, on='gift_card_id', how='outer')
    merged['difference'] = merged['ledger_balance'].fillna(0) - merged['processor_balance'].fillna(0)
    merged['status'] = merged['difference'].apply(
        lambda x: 'MATCH' if abs(x) < 0.01 else ('LEDGER_EXCESS' if x > 0 else 'PROCESSOR_EXCESS')
    )
    
    return merged[['gift_card_id', 'ledger_balance', 'processor_balance', 'difference', 'status']]
