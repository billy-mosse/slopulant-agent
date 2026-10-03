"""Inventory replenishment engine implementing a periodic review (s, S) policy.

This module calculates optimal reorder quantities for each SKU–warehouse combination
to maintain a 95% cycle service level, respecting vendor constraints (case packs,
minimum order quantities) and corporate budget limits. It consumes demand forecasts,
current inventory positions, and vendor agreements to produce a prioritized list of
purchase order recommendations.

Key behaviors:
  • Computes reorder points (s) and order-up-to levels (S) based on lead time plus
    review period demand, incorporating safety stock derived from forecast uncertainty.
  • Enforces vendor-specific packaging and minimum order constraints.
  • Applies budgetary discipline by retaining only the highest-margin recommendations
    when total projected spend exceeds the open-to-buy ceiling.
"""
from __future__ import annotations

import argparse
import logging
import math
from dataclasses import dataclass

import pandas as pd
from sqlalchemy import create_engine, text

logger = logging.getLogger("inventory_planning")

Z_SCORE_95_PERCENTILE = 1.645
REVIEW_INTERVAL_WEEKS = 1.0
QUANTILE_SPREAD_TO_STDDEV = 2.563  # approximates normal distribution spread


@dataclass(frozen=True)
class VendorAgreement:
    identifier: str
    supplier_code: str
    procurement_lead_time_weeks: float
    packaging_size: int
    minimum_order_quantity: int
    unit_purchase_price: float


@dataclass
class ReplenishmentRecommendation:
    stock_keeping_unit: str
    facility_code: str
    supplier_code: str
    reorder_threshold: float
    target_inventory_level: float
    current_inventory_position: float
    recommended_order_units: int
    unit_cost: float
    gross_margin_percentage: float

    @property
    def projected_expenditure(self) -> float:
        return self.recommended_order_units * self.unit_cost


def fetch_replenishment_inputs(database_uri: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load demand forecasts, inventory snapshots, and vendor agreements."""
    with create_engine(database_uri).connect() as conn:
        forecast = pd.read_sql(
            text("SELECT sku, warehouse_id, horizon, p10, p50, p90 "
                 "FROM supply.demand_forecast "
                 "WHERE run_date = (SELECT MAX(run_date) FROM supply.demand_forecast)"),
            conn
        )
        inventory = pd.read_sql(
            text("SELECT sku, warehouse_id, on_hand, on_order, allocated, margin_pct "
                 "FROM inventory.stock_levels"),
            conn
        )
        vendors = pd.read_sql(
            text("SELECT sku, vendor_id, lead_time_days, case_pack, moq_units, unit_cost "
                 "FROM supply.vendor_terms WHERE is_primary = TRUE"),
            conn
        )
    return forecast, inventory, vendors


def _aggregate_demand_over_horizon(
    forecast_series: pd.DataFrame,
    planning_horizon_weeks: float
) -> tuple[float, float]:
    """Estimate expected demand and standard deviation over a specified time window."""
    forecast_series = forecast_series.sort_values("horizon").reset_index(drop=True)
    full_weeks = int(math.floor(planning_horizon_weeks))
    partial_week = planning_horizon_weeks - full_weeks
    cumulative_mean = 0.0
    cumulative_variance = 0.0

    for idx, record in enumerate(forecast_series.itertuples()):
        weight = 1.0 if idx < full_weeks else (partial_week if idx == full_weeks else 0.0)
        if weight <= 0.0:
            break
        weekly_stddev = max(record.p90 - record.p10, 0.0) / QUANTILE_SPREAD_TO_STDDEV
        cumulative_mean += weight * record.p50
        cumulative_variance += weight * (weekly_stddev ** 2)

    return cumulative_mean, math.sqrt(cumulative_variance)


def _enforce_packaging_constraints(
    suggested_quantity: float,
    agreement: VendorAgreement
) -> int:
    """Adjust suggested quantity to comply with case pack and MOQ requirements."""
    if suggested_quantity <= 0:
        return 0
    case_packs_needed = math.ceil(suggested_quantity / agreement.packaging_size)
    units_after_packaging = case_packs_needed * agreement.packaging_size
    if units_after_packaging < agreement.minimum_order_quantity:
        case_packs_needed = math.ceil(agreement.minimum_order_quantity / agreement.packaging_size)
        units_after_packaging = case_packs_needed * agreement.packaging_size
    return int(units_after_packaging)


def generate_replenishment_plan(
    demand_forecast: pd.DataFrame,
    inventory_snapshot: pd.DataFrame,
    vendor_agreements: pd.DataFrame
) -> list[ReplenishmentRecommendation]:
    """Compute reorder recommendations for all SKU–warehouse combinations."""
    vendor_map = {
        row.sku: VendorAgreement(
            row.sku,
            row.vendor_id,
            row.lead_time_days / 7.0,
            int(row.case_pack),
            int(row.moq_units),
            float(row.unit_cost)
        )
        for row in vendor_agreements.itertuples()
    }
    recommendations: list[ReplenishmentRecommendation] = []

    for (sku, facility), group in demand_forecast.groupby(["sku", "warehouse_id"]):
        agreement = vendor_map.get(sku)
        if not agreement:
            logger.warning("Missing primary vendor agreement for SKU %s; skipping", sku)
            continue

        inventory_row = inventory_snapshot[
            (inventory_snapshot.sku == sku) &
            (inventory_snapshot.warehouse_id == facility)
        ]
        if inventory_row.empty:
            continue
        inventory_row = inventory_row.iloc[0]

        lead_time_demand_mean, lead_time_demand_std = _aggregate_demand_over_horizon(
            group, agreement.procurement_lead_time_weeks
        )
        review_plus_lead_demand_mean, review_plus_lead_demand_std = _aggregate_demand_over_horizon(
            group, agreement.procurement_lead_time_weeks + REVIEW_INTERVAL_WEEKS
        )

        reorder_point = lead_time_demand_mean + Z_SCORE_95_PERCENTILE * lead_time_demand_std
        target_level = review_plus_lead_demand_mean + Z_SCORE_95_PERCENTILE * review_plus_lead_demand_std

        inventory_position = (
            inventory_row.on_hand
            + inventory_row.on_order
            - inventory_row.allocated
        )

        order_quantity = 0
        if inventory_position <= reorder_point:
            order_quantity = _enforce_packaging_constraints(
                target_level - inventory_position, agreement
            )

        if order_quantity > 0:
            recommendations.append(ReplenishmentRecommendation(
                sku=sku,
                facility_code=facility,
                supplier_code=agreement.supplier_code,
                reorder_threshold=reorder_point,
                target_inventory_level=target_level,
                current_inventory_position=inventory_position,
                recommended_order_units=order_quantity,
                unit_cost=agreement.unit_purchase_price,
                gross_margin_percentage=inventory_row.margin_pct
            ))

    return recommendations


def enforce_budget_cap(
    recommendations: list[ReplenishmentRecommendation],
    available_funds: float
) -> list[ReplenishmentRecommendation]:
    """Reduce recommendations to fit within budget, prioritizing higher-margin items."""
    total_cost = sum(r.projected_expenditure for r in recommendations)
    if total_cost <= available_funds:
        return recommendations

    sorted_recommendations = sorted(
        recommendations,
        key=lambda r: r.gross_margin_percentage,
        reverse=True
    )
    retained = []
    current_spend = 0.0

    for rec in sorted_recommendations:
        if current_spend + rec.projected_expenditure <= available_funds:
            retained.append(rec)
            current_spend += rec.projected_expenditure
        else:
            logger.info(
                "Budget constraint triggered: excluded %s at %s "
                "(margin %.1f%%, $%.0f projected)",
                rec.stock_keeping_unit,
                rec.facility_code,
                100 * rec.gross_margin_percentage,
                rec.projected_expenditure
            )

    return retained


def execute_replenishment_workflow(
    database_uri: str,
    budget_limit: float,
    simulate_only: bool = False
) -> None:
    """Orchestrate the full replenishment calculation and output pipeline."""
    logger.info("Starting replenishment cycle with budget cap of $%.0f", budget_limit)
    forecasts, inventories, agreements = fetch_replenishment_inputs(database_uri)
    candidate_orders = generate_replenishment_plan(forecasts, inventories, agreements)
    approved_orders = enforce_budget_cap(candidate_orders, budget_limit)

    output_records = [
        {**vars(r), "projected_expenditure": r.projected_expenditure}
        for r in approved_orders
    ]
    order_dataframe = pd.DataFrame(output_records)
    order_dataframe["generated_timestamp"] = pd.Timestamp.utcnow()

    total_spend = order_dataframe["projected_expenditure"].sum() if len(order_dataframe) else 0
    logger.info(
        "Replenishment cycle complete: %d orders, total projected spend $%.0f",
        len(order_dataframe),
        total_spend
    )

    if not simulate_only and len(order_dataframe):
        engine = create_engine(database_uri)
        order_dataframe.to_sql(
            "purchase_orders",
            engine,
            schema="supply",
            if_exists="append",
            index=False
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inventory replenishment engine")
    parser.add_argument("--dsn", required=True, help="Database connection string")
    parser.add_argument("--budget", type=float, default=750_000.0, help="Open-to-buy budget")
    parser.add_argument("--dry-run", action="store_true", help="Skip database writes")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    execute_replenishment_workflow(args.dsn, args.budget, args.dry_run)
