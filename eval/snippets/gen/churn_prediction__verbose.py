class ChurnFeatureEngineer:
    """Generates customer-level behavioral signals to support churn risk assessment.

    This module aggregates historical purchasing activity and web session behavior
    over a configurable lookback window, producing a rich feature set for downstream
    predictive modeling. Each output row corresponds to a unique customer, with
    features capturing recency, frequency, monetary value, category preferences,
    and digital engagement patterns as of a specified cutoff date.
    """

    def __init__(self, cutoff_date: pd.Timestamp, lookback_window_days: int = 730):
        self.cutoff = pd.Timestamp(cutoff_date)
        self.lookback_start = self.cutoff - pd.Timedelta(days=lookback_window_days)
        self.category_half_life = 90.0
        self.prominent_categories = ["bedding", "bath", "decor", "kitchen", "rugs", "lighting", "furniture"]

    def construct_feature_set(self, database_connection) -> pd.DataFrame:
        """Assemble the complete feature matrix for all customers as of the cutoff date."""
        params = {"start": self.lookback_start, "cutoff": self.cutoff}

        order_data = pd.read_sql(self._order_query(), database_connection, params=params, parse_dates=["order_ts"])
        session_data = pd.read_sql(self._session_query(), database_connection, params=params, parse_dates=["event_ts"])

        rfm_metrics = self._compute_rfm_metrics(order_data)
        category_mix = self._derive_category_mix(order_data)
        engagement_signals = self._extract_session_indicators(session_data)

        combined = rfm_metrics.join(category_mix, how="left").join(engagement_signals, how="left")
        combined = combined.fillna(0.0).drop(columns=["last_order_timestamp", "first_order_timestamp"], errors="ignore")
        return combined

    def _order_query(self) -> str:
        return """
            SELECT customer_id, order_id, order_ts, category_l1, quantity, net_revenue
            FROM orders.lines
            WHERE order_ts >= %(start)s AND order_ts < %(cutoff)s
              AND status NOT IN ('cancelled', 'fraud')
        """

    def _session_query(self) -> str:
        return """
            SELECT customer_id, session_id, event_ts, event_type
            FROM sessions.events
            WHERE event_ts >= %(start)s AND event_ts < %(cutoff)s
              AND customer_id IS NOT NULL
        """

    def _compute_rfm_metrics(self, order_lines: pd.DataFrame) -> pd.DataFrame:
        order_summary = (
            order_lines.groupby(["customer_id", "order_id"], as_index=False)
            .agg(order_timestamp=("order_ts", "min"), order_value=("net_revenue", "sum"), item_count=("quantity", "sum"))
        )

        rfm = order_summary.groupby("customer_id").agg(
            last_order_timestamp=("order_timestamp", "max"),
            first_order_timestamp=("order_timestamp", "min"),
            purchase_count=("order_id", "nunique"),
            total_spend=("order_value", "sum"),
            avg_order_value=("order_value", "mean"),
            avg_items_per_order=("item_count", "mean"),
        )

        rfm["days_since_last_order"] = (self.cutoff - rfm["last_order_timestamp"]).dt.total_seconds() / 86400.0
        rfm["customer_tenure_days"] = (self.cutoff - rfm["first_order_timestamp"]).dt.total_seconds() / 86400.0
        rfm["log_days_since_last_order"] = np.log1p(rfm["days_since_last_order"])
        rfm["log_purchase_count"] = np.log1p(rfm["purchase_count"])
        rfm["log_total_spend"] = np.log1p(rfm["total_spend"].clip(lower=0))
        rfm["log_avg_order_value"] = np.log1p(rfm["avg_order_value"].clip(lower=0))
        rfm["orders_per_month"] = rfm["purchase_count"] / np.maximum(rfm["customer_tenure_days"] / 30.0, 1.0)

        order_summary = order_summary.sort_values(["customer_id", "order_timestamp"])
        order_summary["inter_purchase_interval_days"] = (
            order_summary.groupby("customer_id")["order_timestamp"].diff().dt.total_seconds() / 86400.0
        )
        gap_stats = order_summary.groupby("customer_id")["inter_purchase_interval_days"].agg(
            avg_gap=("inter_purchase_interval_days", "mean"),
            gap_std=("inter_purchase_interval_days", "std"),
        )
        rfm = rfm.join(gap_stats)
        rfm["recency_ratio"] = rfm["days_since_last_order"] / np.maximum(rfm["avg_gap"].fillna(rfm["days_since_last_order"]), 1.0)
        return rfm

    def _derive_category_mix(self, order_lines: pd.DataFrame) -> pd.DataFrame:
        age_in_days = (self.cutoff - order_lines["order_ts"]).dt.total_seconds() / 86400.0
        decay_weights = np.power(0.5, age_in_days / self.category_half_life) * order_lines["net_revenue"].clip(lower=0)
        order_lines["category_group"] = order_lines["category_l1"].where(
            order_lines["category_l1"].isin(self.prominent_categories), "other"
        )
        category_spending = order_lines.groupby(["customer_id", "category_group"])["decay_weights"].sum().unstack(fill_value=0.0)
        category_totals = category_spending.sum(axis=1).replace(0, np.nan)
        category_shares = category_spending.div(category_totals, axis=0).fillna(0.0).add_prefix("category_share_")
        probabilities = category_shares.to_numpy().clip(1e-12, 1.0)
        category_shares["category_entropy"] = -(probabilities * np.log(probabilities)).sum(axis=1)
        category_shares["total_decay_weighted_spend"] = category_totals.fillna(0.0)
        return category_shares

    def _extract_session_indicators(self, session_events: pd.DataFrame) -> pd.DataFrame:
        session_summary = (
            session_events.groupby(["customer_id", "session_id"], as_index=False)
            .agg(session_start=("event_ts", "min"), event_count=("event_type", "size"), cart_additions=("event_type", lambda s: (s == "add_to_cart").sum()))
        )
        session_summary["days_since_session"] = (self.cutoff - session_summary["session_start"]).dt.total_seconds() / 86400.0

        session_metrics = session_summary.groupby("customer_id").agg(
            most_recent_session_days=("days_since_session", "min"),
            total_sessions=("session_id", "nunique"),
            avg_events_per_session=("event_count", "mean"),
            total_cart_additions=("cart_additions", "sum"),
        )

        for window in (7, 30, 90):
            recent_sessions = session_summary[session_summary["days_since_session"] <= window]
            session_metrics[f"sessions_in_last_{window}d"] = recent_sessions.groupby("customer_id")["session_id"].nunique()

        session_metrics = session_metrics.fillna({f"sessions_in_last_{w}d": 0 for w in (7, 30, 90)})
        session_metrics["visit_frequency_trend"] = (
            session_metrics["sessions_in_last_30d"] + 1
        ) / (session_metrics["sessions_in_last_90d"] / 3 + 1)
        session_metrics["log_days_since_last_session"] = np.log1p(session_metrics["most_recent_session_days"])
        return session_metrics


class ChurnLabeler:
    """Identifies customers who remained active within a defined future window.

    A customer is considered retained if they placed at least one qualifying order
    during the evaluation period beginning immediately after the scoring cutoff.
    All others are flagged as churned for modeling purposes.
    """

    def __init__(self, cutoff_date: pd.Timestamp, evaluation_horizon_days: int = 90):
        self.cutoff = pd.Timestamp(cutoff_date)
        self.evaluation_end = self.cutoff + pd.Timedelta(days=evaluation_horizon_days)

    def generate_active_customer_index(self, database_connection) -> pd.Index:
        """Return the set of customers who purchased during the label window."""
        params = {"start": self.cutoff, "cutoff": self.evaluation_end}
        active_buyers = pd.read_sql(self._label_query(), database_connection, params=params)
        return pd.Index(active_buyers["customer_id"].unique(), name="customer_id")

    def _label_query(self) -> str:
        return """
            SELECT DISTINCT customer_id
            FROM orders.lines
            WHERE order_ts >= %(start)s AND order_ts < %(cutoff)s
        """
