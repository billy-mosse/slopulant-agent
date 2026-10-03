class DemandBacktestRunner:
    """
    Evaluates forecast accuracy using a rolling-origin cross-validation strategy.
    For each evaluation origin, the system trains on historical data up to that point,
    generates 12-week ahead predictions, and computes performance metrics—WAPE and bias—
    stratified by product category and forecast horizon bucket (1–4, 5–8, 9–12 weeks).
    Results are aggregated to support data-driven model selection and horizon-specific tuning.
    """

    def __init__(self, database_connection_string: str, num_validation_folds: int = 6, fold_step_weeks: int = 4) -> None:
        self._db_uri = database_connection_string
        self._num_folds = num_validation_folds
        self._step = fold_step_weeks

    def execute(self) -> None:
        engine = create_engine(self._db_uri)
        weekly_panel = load_weekly_panel(engine, reference_start_date="2022-01-03")
        origins = self._construct_rolling_origins(weekly_panel["week_start"])
        fold_results = []

        for evaluation_date in origins:
            fold_output = self._evaluate_single_origin(weekly_panel, evaluation_date)
            actuals = fold_output["units"].to_numpy()
            predictions = fold_output["p50"].to_numpy()
            log.info("origin %s: WAPE=%.3f bias=%+.3f", evaluation_date.date(), wape(actuals, predictions), bias(actuals, predictions))
            fold_results.append(fold_output)

        summary = self._aggregate_metrics(pd.concat(fold_results, ignore_index=True))
        print(summary.to_string(index=False))

    def _construct_rolling_origins(self, week_series: pd.Series) -> list[pd.Timestamp]:
        unique_weeks = np.sort(week_series.unique())
        last_valid_origin_index = len(unique_weeks) - HORIZON - 1
        return [
            pd.Timestamp(unique_weeks[last_valid_origin_index - i * self._step])
            for i in range(self._num_folds)
        ][::-1]

    def _evaluate_single_origin(self, full_panel: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        training_window = full_panel[full_panel["week_start"] <= cutoff]
        validation_window = full_panel[
            full_panel["week_start"].between(cutoff + pd.Timedelta(weeks=1), cutoff + pd.Timedelta(weeks=HORIZON))
        ]
        trained_models = train_models(build_features(training_window.copy()), num_rounds=400)
        predictions = recursive_forecast(trained_models, training_window, validation_window[["sku", "week_start", "markdown_pct"]].drop_duplicates())
        merged = predictions.merge(
            validation_window[KEYS + ["category", "week_start", "units"]],
            on=KEYS + ["week_start"],
            how="inner"
        )
        merged["origin_date"] = cutoff
        return merged

    def _aggregate_metrics(self, combined_results: pd.DataFrame) -> pd.DataFrame:
        combined_results["horizon_band"] = pd.cut(
            combined_results["horizon"], bins=[0, 4, 8, 12], labels=["1–4", "5–8", "9–12"]
        )
        metrics_rows = []
        for (category, band), subset in combined_results.groupby(["category", "horizon_band"], observed=True):
            y_true = subset["units"].to_numpy()
            y_pred = subset["p50"].to_numpy()
            interval_coverage = ((subset["units"] >= subset["p10"]) & (subset["units"] <= subset["p90"])).mean()
            metrics_rows.append({
                "category": category,
                "horizon_band": band,
                "wape": wape(y_true, y_pred),
                "bias": bias(y_true, y_pred),
                "interval_coverage": interval_coverage,
                "observation_count": len(subset),
            })
        return pd.DataFrame(metrics_rows).sort_values(["category", "horizon_band"])


def wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true).sum()
    return float(np.abs(y_true - y_pred).sum() / denom) if denom > 0 else np.nan


def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = y_true.sum()
    return float((y_pred.sum() - denom) / denom) if denom > 0 else np.nan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run rolling-origin backtest for demand forecasts")
    parser.add_argument("--dsn", required=True, help="Database connection string")
    parser.add_argument("--folds", type=int, default=6, help="Number of validation folds")
    parser.add_argument("--step-weeks", type=int, default=4, help="Weeks between origins")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    runner = DemandBacktestRunner(args.dsn, args.folds, args.step_weeks)
    runner.execute()
