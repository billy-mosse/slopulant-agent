class TrendingProductAnalyzer:
    """
    Computes hourly trending product rankings within each category by comparing
    recent user engagement against a 28-day historical baseline. The scoring
    methodology applies exponential decay to recent events (views, cart additions,
    purchases) and normalizes the result using a Poisson-aware z-score to ensure
    stability for low-volume items. The output powers the product recommendations
    engine’s trending section.
    """

    def __init__(self, database_url: str, stock_inventory: dict | None = None):
        self.engine = create_engine(database_url)
        self.stock = stock_inventory or {}

    def execute(self, reference_time: datetime) -> pd.DataFrame:
        raw_events = self._collect_events(reference_time)
        scored = self._compute_trending_scores(raw_events, reference_time)
        filtered = self._enforce_quality_gates(scored)
        ranked = self._generate_category_rankings(filtered)
        return ranked

    def _collect_events(self, now: datetime) -> pd.DataFrame:
        lookback_start = now - timedelta(days=BASELINE_DAYS)
        with self.engine.connect() as conn:
            events = pd.read_sql(text(EVENTS_SQL), conn, params={"since": lookback_start})
            purchases = pd.read_sql(text(PURCHASES_SQL), conn, params={"since": lookback_start})
        combined = pd.concat([events, purchases], ignore_index=True)
        combined["event_ts"] = pd.to_datetime(combined["event_ts"])
        combined["event_weight"] = combined["event_type"].map(EVENT_WEIGHTS)
        return combined

    def _compute_trending_scores(self, events: pd.DataFrame, now: datetime) -> pd.DataFrame:
        recent_threshold = now - timedelta(hours=RECENT_H)
        hours_old = (now - events["event_ts"]).dt.total_seconds() / 3600
        decay_factor = 0.5 ** (hours_old / HALF_LIFE_H)
        events = events.assign(decayed_score=events["event_weight"] * decay_factor, is_recent=events["event_ts"] >= recent_threshold)

        baseline_window_count = (BASELINE_DAYS * 24 - RECENT_H) / RECENT_H
        baseline_events = events[~events.is_recent]
        window_id = ((recent_threshold - baseline_events["event_ts"]).dt.total_seconds() // (RECENT_H * 3600)).astype(int)
        per_window = baseline_events.assign(window=window_id).groupby(["sku", "window"]).event_weight.sum()

        stats = defaultdict(lambda: [0.0, 0.0, 0])
        for (sku, _), weight_sum in per_window.items():
            s = stats[sku]
            s[0] += weight_sum
            s[1] += weight_sum ** 2
            s[2] += 1

        recent_agg = events[events.is_recent].groupby(["sku", "category_id"]).agg(
            decayed_sum=("decayed_score", "sum"), recent_count=("event_weight", "size")
        )
        baseline_counts = baseline_events.groupby("sku").size()

        normalization = HALF_LIFE_H / math.log(2) / RECENT_H * (1 - 0.5 ** (RECENT_H / HALF_LIFE_H))
        results = []
        for (sku, cat), row in recent_agg.iterrows():
            total, sq_sum, win_count = stats[sku]
            mean_baseline = total / baseline_window_count
            variance = max(sq_sum / baseline_window_count - mean_baseline ** 2, 0.0)
            std_dev = math.sqrt(variance + mean_baseline + 1.0)
            current_rate = row.decayed_sum / normalization
            z_score = (current_rate - mean_baseline) / std_dev if std_dev > 0 else 0.0
            results.append({
                "sku": sku, "category_id": cat, "current_rate": current_rate,
                "baseline_rate": mean_baseline, "z_score": z_score,
                "recent_event_count": row.recent_count, "historical_event_count": int(baseline_counts.get(sku, 0))
            })
        return pd.DataFrame(results)

    def _enforce_quality_gates(self, scores: pd.DataFrame) -> pd.DataFrame:
        mask = (
            (scores.recent_event_count >= MIN_RECENT_EVENTS) &
            (scores.historical_event_count >= MIN_BASELINE_EVENTS) &
            (scores.z_score > 0)
        )
        if self.stock:
            mask &= scores.sku.map(lambda s: self.stock.get(s, 0) >= MIN_STOCK)
        log.info("Quality gates retained %d of %d candidates", mask.sum(), len(scores))
        return scores[mask]

    def _generate_category_rankings(self, scored: pd.DataFrame) -> pd.DataFrame:
        output = []
        for cat, group in scored.sort_values("z_score", ascending=False).groupby("category_id", sort=False):
            for rank, (_, row) in enumerate(islice(group.iterrows(), TOP_K), start=1):
                output.append({
                    "category_id": cat, "sku": row.sku, "rank": rank,
                    "velocity_z": round(row.z_score, 3),
                    "current_rate": round(row.current_rate, 2),
                    "baseline_rate": round(row.baseline_rate, 2)
                })
        return pd.DataFrame(output) if output else pd.DataFrame(columns=["category_id", "sku", "rank", "velocity_z", "current_rate", "baseline_rate"])
