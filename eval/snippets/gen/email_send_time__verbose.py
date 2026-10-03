class PersonalizedSendHourOptimizer:
    """
    Determines the optimal local hour to send marketing emails to each customer by
    combining historical engagement data with a global behavioral prior, while
    supporting exploration via Thompson sampling. For customers with limited send
    history, recommendations gracefully fallback to segment-level insights.
    
    The system processes the last 365 days of email events, aggregates engagement
    within three-hour local time buckets across the week, and applies empirical Bayes
    shrinking to stabilize estimates. Final recommendations prioritize robustness
    for low-volume customers while continuing to explore better-performing hours.
    """

    def __init__(self, engine, reference_time: pd.Timestamp, random_seed: int = 0, dry_run: bool = False):
        self.engine = engine
        self.reference_time = reference_time
        self.rng = np.random.default_rng(random_seed)
        self.dry_run = dry_run
        self.bucket_size_hours = 3
        self.num_buckets = 7 * 24 // self.bucket_size_hours
        self.min_personal_interactions = 8
        self.lookback_window = pd.Timedelta(days=365)
        self.open_window_hours = 48

    def execute(self) -> pd.DataFrame:
        events = self._fetch_events()
        events = self._standardize_timezones(events)
        events = self._assign_buckets_and_outcomes(events)
        global_prior = self._build_empirical_bayes_prior(events)
        recommendations = self._compute_customer_posteriors(events, global_prior)
        recommendations = self._apply_segment_fallback(recommendations, global_prior)
        output = self._prepare_final_output(recommendations)
        self._persist_or_log(output)
        return output

    def _fetch_events(self) -> pd.DataFrame:
        query = """
            SELECT customer_id, sent_ts_utc, opened_ts_utc, customer_tz, segment_name
            FROM marketing.email_events
            WHERE event_type = 'send'
              AND sent_ts_utc BETWEEN %(start)s AND %(asof)s
        """
        params = {"start": self.reference_time - self.lookback_window, "asof": self.reference_time}
        return pd.read_sql(query, self.engine, params=params, parse_dates=["sent_ts_utc", "opened_ts_utc"])

    def _standardize_timezones(self, df: pd.DataFrame) -> pd.DataFrame:
        df["customer_tz"] = df["customer_tz"].fillna("America/New_York")
        bad_zones = ~df["customer_tz"].str.contains("/", regex=False)
        if bad_zones.any():
            df.loc[bad_zones, "customer_tz"] = "America/New_York"
        df["sent_local"] = df.apply(
            lambda r: r["sent_ts_utc"].tz_convert(r["customer_tz"]) if r["sent_ts_utc"].tz else r["sent_ts_utc"].tz_localize("UTC").tz_convert(r["customer_tz"]),
            axis=1
        )
        df["dow"] = df["sent_local"].dt.dayofweek
        df["hour_local"] = df["sent_local"].dt.hour
        return df

    def _assign_buckets_and_outcomes(self, df: pd.DataFrame) -> pd.DataFrame:
        df["bucket"] = df["dow"] * (24 // self.bucket_size_hours) + df["hour_local"] // self.bucket_size_hours
        df["open_flag"] = (
            df["opened_ts_utc"].notna() &
            ((df["opened_ts_utc"].dt.tz_localize("UTC") - df["sent_ts_utc"].dt.tz_localize("UTC")) <= pd.Timedelta(hours=self.open_window_hours))
        ).astype(int)
        return df

    def _build_empirical_bayes_prior(self, df: pd.DataFrame) -> pd.DataFrame:
        # Aggregate per-customer per-bucket stats
        per_customer_bucket = (
            df.groupby(["customer_id", "bucket"])
            .agg(total_sends=("open_flag", "size"), total_opens=("open_flag", "sum"))
            .reset_index()
        )
        # Compute global bucket-level rates and variance components
        global_stats = per_customer_bucket.groupby("bucket").agg(
            sends=("total_sends", "sum"), opens=("total_opens", "sum")
        )
        global_stats["success_rate"] = (global_stats["opens"] + 1) / (global_stats["sends"] + 2)
        filtered = per_customer_bucket[per_customer_bucket["total_sends"] >= 3].copy()
        filtered["observed_rate"] = filtered["total_opens"] / filtered["total_sends"]
        variance_per_bucket = filtered.groupby("bucket")["observed_rate"].var()
        avg_sends_per_bucket = filtered.groupby("bucket")["total_sends"].mean()
        # Method-of-moments for Beta parameters
        binom_var = global_stats["success_rate"] * (1 - global_stats["success_rate"]) / avg_sends_per_bucket.fillna(3)
        between_var = (variance_per_bucket.fillna(0) - binom_var).clip(lower=1e-5)
        concentration = (global_stats["success_rate"] * (1 - global_stats["success_rate"]) / between_var - 1).clip(lower=2, upper=50)
        global_stats["alpha"] = global_stats["success_rate"] * concentration
        global_stats["beta"] = (1 - global_stats["success_rate"]) * concentration
        return global_stats.reindex(range(self.num_buckets), fill_value={"alpha": 1.0, "beta": 1.0})

    def _compute_customer_posteriors(self, events: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
        per_customer_bucket = (
            events.groupby(["customer_id", "bucket"])
            .agg(total_sends=("open_flag", "size"), total_opens=("open_flag", "sum"))
            .reset_index()
        )
        grid = pd.MultiIndex.from_product(
            [per_customer_bucket["customer_id"].unique(), range(self.num_buckets)],
            names=["customer_id", "bucket"]
        )
        extended = per_customer_bucket.set_index(["customer_id", "bucket"]).reindex(grid, fill_value=0).reset_index()
        extended["post_alpha"] = extended["bucket"].map(prior["alpha"]) + extended["total_opens"]
        extended["post_beta"] = extended["bucket"].map(prior["beta"]) + extended["total_sends"] - extended["total_opens"]
        extended["draw"] = self.rng.beta(extended["post_alpha"], extended["post_beta"])
        best_idx = extended.groupby("customer_id")["draw"].idxmax()
        picks = extended.loc[best_idx].set_index("customer_id")[["bucket", "post_alpha", "post_beta", "draw"]]
        picks.columns = ["recommendation_bucket", "alpha", "beta", "thompson_sample"]
        per_customer = events.groupby("customer_id").agg(
            tz=("customer_tz", "last"), segment=("segment_name", "last"), sends=("open_flag", "size")
        ).join(picks)
        per_customer["strategy"] = "personalized"
        return per_customer

    def _apply_segment_fallback(self, recommendations: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
        segment_stats = (
            recommendations.reset_index()
            .merge(self._build_segment_bucket_aggregates(prior), left_on="bucket", right_index=True)
            .groupby(["segment", "bucket"])[["opens", "sends"]].sum()
            .assign(rate=lambda d: d["opens"] / d["sends"])
        )
        segment_best = segment_stats.reset_index().loc[
            segment_stats.reset_index().groupby("segment")["rate"].idxmax()
        ].set_index("segment")
        thin_customers = recommendations["sends"] < self.min_personal_interactions
        recommendations.loc[thin_customers, "recommendation_bucket"] = recommendations.loc[thin_customers, "segment"].map(
            segment_best["recommendation_bucket"]
        ).fillna(prior["alpha"].idxmax())
        recommendations.loc[thin_customers, "strategy"] = "segment-based"
        return recommendations

    def _build_segment_bucket_aggregates(self, prior: pd.DataFrame) -> pd.DataFrame:
        # Helper to construct segment-level bucket stats
        pass

    def _prepare_final_output(self, df: pd.DataFrame) -> pd.DataFrame:
        result = df.reset_index()
        result["day_of_week"] = result["recommendation_bucket"] // (24 // self.bucket_size_hours)
        result["hour_local"] = (result["recommendation_bucket"] % (24 // self.bucket_size_hours)) * self.bucket_size_hours + self.bucket_size_hours // 2
        result["optimization_timestamp"] = self.reference_time
        return result[["customer_id", "tz", "day_of_week", "hour_local", "strategy", "optimization_timestamp"]]

    def _persist_or_log(self, df: pd.DataFrame) -> None:
        if self.dry_run:
            logging.info(f"Dry-run: would write {len(df):,} recommendations to marketing.send_times")
        else:
            df.to_sql("send_times", self.engine, schema="marketing", if_exists="replace", index=False, chunksize=50_000)
            logging.info(f"Wrote {len(df):,} records to marketing.send_times")
