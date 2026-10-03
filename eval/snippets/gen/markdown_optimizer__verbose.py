class MarkdownStrategyOptimizer:
    """
    Computes optimal end-of-season pricing strategies for overstocked items.

    Given a catalog of items with remaining inventory and full-price sales
    projections, this module determines a sequence of permanent price
    reductions (a markdown ladder) that maximizes the net present value of
    revenue minus holding and clearance costs. The optimization uses a
    backward-induction dynamic programming approach over discrete stock
    levels and allowed discount tiers, respecting the business constraint
    that discounts may only increase (prices may only decrease) over time.

    The output is a week-by-week plan specifying the discount percentage,
    expected units sold, and projected leftover inventory for each item
    identified as having excess stock relative to the time remaining in the
    current selling season.
    """

    ALLOWED_DISCOUNTS = [0.00, 0.10, 0.20, 0.30, 0.40]
    STOCK_BUCKETS = 50
    WEEKLY_HOLDING_FRACTION = 0.004
    SALVAGE_RATE = 0.15
    CLEARANCE_PENALTY = 0.25
    DEFAULT_ELASTICITY = -1.5

    def __init__(self, database_connection):
        self._engine = database_connection

    def _fetch_inventory_and_elasticities(self):
        inventory_query = """
            SELECT sku, category, on_hand_units, full_price, baseline_weekly_units,
                   season_end_date
            FROM inventory.stock_levels
            WHERE season_end_date > CURRENT_DATE
        """
        elasticity_query = "SELECT category, elasticity FROM pricing.elasticities"
        inventory_df = pd.read_sql(inventory_query, self._engine)
        elasticity_df = pd.read_sql(elasticity_query, self._engine)
        return inventory_df, elasticity_df

    def _identify_candidates(self, inventory_df, reference_date):
        weeks_remaining = (pd.to_datetime(inventory_df["season_end_date"]) - pd.Timestamp(reference_date)).dt.days // 7
        inventory_df = inventory_df.assign(weeks_remaining=weeks_remaining.clip(lower=0))
        projected_sales = inventory_df["baseline_weekly_units"] * inventory_df["weeks_remaining"]
        mask = (inventory_df["weeks_remaining"] > 0) & (inventory_df["on_hand_units"] > projected_sales)
        return inventory_df[mask].copy()

    def _compute_demand(self, base_volume, discount, elasticity):
        return base_volume * ((1.0 - discount) ** elasticity)

    def _optimize_ladder(self, initial_stock, unit_price, base_volume, elasticity, horizon):
        bucket_size = max(initial_stock / self.STOCK_BUCKETS, 1.0)
        n_buckets = int(np.ceil(initial_stock / bucket_size)) + 1
        n_discounts = len(self.ALLOWED_DISCOUNTS)

        value = np.zeros((horizon + 1, n_buckets, n_discounts))
        decision = np.zeros((horizon, n_buckets, n_discounts), dtype=int)

        # Terminal value: salvage proceeds minus clearance penalty
        for b in range(n_buckets):
            leftover = b * bucket_size
            value[horizon, b, :] = leftover * unit_price * (self.SALVAGE_RATE - self.CLEARANCE_PENALTY)

        # Backward induction
        for week in range(horizon - 1, -1, -1):
            for stock_bucket in range(n_buckets):
                current_stock = stock_bucket * bucket_size
                for current_discount_idx in range(n_discounts):
                    best_value, best_discount_idx = -np.inf, current_discount_idx
                    for candidate_idx in range(current_discount_idx, n_discounts):
                        discount = self.ALLOWED_DISCOUNTS[candidate_idx]
                        expected_sales = min(self._compute_demand(base_volume, discount, elasticity), current_stock)
                        ending_stock = current_stock - expected_sales
                        revenue = expected_sales * unit_price * (1.0 - discount)
                        holding_cost = ending_stock * unit_price * self.WEEKLY_HOLDING_FRACTION
                        next_bucket = min(int(round(ending_stock / bucket_size)), n_buckets - 1)
                        total = revenue - holding_cost + value[week + 1, next_bucket, candidate_idx]
                        if total > best_value:
                            best_value, best_discount_idx = total, candidate_idx
                    value[week, stock_bucket, current_discount_idx] = best_value
                    decision[week, stock_bucket, current_discount_idx] = best_discount_idx

        # Forward reconstruction
        ladder, trajectory = [], []
        remaining_stock, discount_state = float(initial_stock), 0
        for week in range(horizon):
            bucket = min(int(round(remaining_stock / bucket_size)), n_buckets - 1)
            chosen_idx = decision[week, bucket, discount_state]
            discount = self.ALLOWED_DISCOUNTS[chosen_idx]
            sales = min(self._compute_demand(base_volume, discount, elasticity), remaining_stock)
            remaining_stock -= sales
            trajectory.append((week, discount, sales, remaining_stock))
            ladder.append(discount)
            discount_state = chosen_idx

        initial_bucket = min(int(round(initial_stock / bucket_size)), n_buckets - 1)
        return ladder, trajectory, float(value[0, initial_bucket, 0])

    def generate_plan(self, reference_date):
        inventory, elasticities = self._fetch_inventory_and_elasticities()
        candidates = self._identify_candidates(inventory, reference_date)
        elasticity_map = dict(zip(elasticities["category"], elasticities["elasticity"]))
        results = []

        for row in candidates.itertuples(index=False):
            elasticity = elasticity_map.get(row.category, self.DEFAULT_ELASTICITY)
            if elasticity > -0.2:
                elasticity = self.DEFAULT_ELASTICITY
            ladder, schedule, objective = self._optimize_ladder(
                int(row.on_hand_units),
                float(row.full_price),
                float(row.baseline_weekly_units),
                elasticity,
                int(row.weeks_remaining)
            )
            for week, discount, sold, leftover in schedule:
                results.append({
                    "sku": row.sku,
                    "week_start": reference_date + timedelta(weeks=week),
                    "discount_pct": round(discount * 100),
                    "markdown_price": round(row.full_price * (1 - discount), 2),
                    "expected_units": round(sold, 1),
                    "expected_remaining": round(leftover, 1),
                    "elasticity_used": elasticity,
                    "plan_value": round(objective, 2),
                })
        return pd.DataFrame(results)
