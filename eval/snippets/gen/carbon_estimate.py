"""
Estimate CO2 emissions per shipment for home goods e-commerce.

Input: raw.shipments (columns: distance_km, weight_kg, carrier_mode)
Output: analytics.shipment_co2_estimates (columns: shipment_id, co2_kg)
"""

import pandas as pd

def estimate_co2_per_shipment(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate CO2 emissions (kg) per shipment based on distance, weight, and carrier mode.
    Uses industry-average emission factors (g CO2 per tonne-km) adjusted for home goods.
    """
    # Emission factors (g CO2 / tonne-km) for common carriers, scaled for home goods (lighter avg load)
    factors = {
        "ground": 120,   # regional trucking
        "air": 500,      # cargo flights
        "rail": 30,      # freight rail
        "ocean": 15      # container ships
    }
    
    df = df.copy()
    df["co2_kg"] = (
        df["distance_km"]
        * df["weight_kg"]
        * df["carrier_mode"].str.lower().map(factors).fillna(120)  # default to ground
        / 1_000_000  # convert g to kg
    )
    
    return df[["shipment_id", "co2_kg"]]
