class UpliftPerformanceAnalyzer:
    """
    Computes key performance metrics for promotional uplift modeling initiatives.
    This module supports the product team in quantifying how well our uplift
    predictions drive incremental conversions and profitability when applied to
    customer targeting strategies. It evaluates ranking quality via the Qini
    curve, normalizes performance with AUUC, and recommends an optimal targeting
    threshold based on business economics.
    """

    def __init__(self, margin_per_conversion: float = 22.0, cost_per_promo: float = 1.80):
        self.margin_per_conversion = margin_per_conversion
        self.cost_per_promo = cost_per_promo

    def compute_ranking_curve(self, evaluation_set: pd.DataFrame, score_column: str = "uplift") -> pd.DataFrame:
        """Generate cumulative Qini statistics from a ranked holdout dataset."""
        ranked = evaluation_set.sort_values(score_column, ascending=False).reset_index(drop=True)
        treatment_indicator = ranked["treated"].to_numpy() == 1
        outcome = ranked["converted"].to_numpy()
        cumulative_treated = np.cumsum(treatment_indicator)
        cumulative_control = np.cumsum(~treatment_indicator)
        treated_conversions = np.cumsum(outcome * treatment_indicator)
        control_conversions = np.cumsum(outcome * ~treatment_indicator)
        qini_values = treated_conversions - np.where(
            cumulative_control > 0, control_conversions * cumulative_treated / cumulative_control, 0.0
        )
        return pd.DataFrame({
            "cumulative_fraction": np.arange(1, len(ranked) + 1) / len(ranked),
            "qini_value": qini_values,
            "treated_count": cumulative_treated,
            "control_count": cumulative_control,
            "score_value": ranked[score_column].to_numpy(),
        })

    def normalized_area_under_qini(self, curve: pd.DataFrame) -> float:
        """Compute AUUC normalized by population size for cross-campaign comparability."""
        final_qini = curve["qini_value"].iloc[-1]
        baseline = curve["cumulative_fraction"] * final_qini
        return float(np.trapz(curve["qini_value"] - baseline, curve["cumulative_fraction"]) / max(len(curve), 1))

    def optimal_targeting_threshold(self, curve: pd.DataFrame, total_eligible_customers: int) -> dict:
        """Determine the top-k targeting strategy maximizing expected incremental profit."""
        scale_factor = total_eligible_customers / len(curve)
        projected_incremental_conversions = curve["qini_value"] * scale_factor
        projected_targeted_customers = curve["cumulative_fraction"] * total_eligible_customers
        profit_projection = (
            projected_incremental_conversions * self.margin_per_conversion
            - projected_targeted_customers * self.cost_per_promo
        )
        best_index = int(np.argmax(profit_projection.to_numpy()))
        return {
            "target_fraction": float(curve["cumulative_fraction"].iloc[best_index]),
            "score_threshold": float(curve["score_value"].iloc[best_index]),
            "projected_incremental_conversions": float(projected_incremental_conversions.iloc[best_index]),
            "projected_profit": float(profit_projection.iloc[best_index]),
            "customers_to_target": int(projected_targeted_customers.iloc[best_index]),
        }
