class MMRDiversifier:
    """Re-rank search result lists to maximize both relevance and attribute-level variety.

    When high-scoring items cluster heavily on attributes like brand, color, or
    price tier, users often benefit from a balanced mix. This class implements
    a greedy maximal marginal relevance algorithm that trades off original rank
    scores against a weighted attribute match similarity metric, with price
    bands normalized per category to avoid cross-category distortions.

    Outputs are written with positional metadata so前后 rank changes can be
    audited, and the lambda trade-off parameter can be tuned via configuration
    or CLI override.

    All data is sourced from the logging warehouse; no schema names appear
    directly to support future migrations.
    """

    INPUT_QUERY_LOGS: str = "search.query_logs"
    INPUT_PRODUCTS: str = "catalog.products"
    OUTPUT_RESULTS: str = "search.diverse_results"

    def __init__(self, configuration: "MMRConfiguration") -> None:
        self.config: MMRConfiguration = configuration
        self.product_attributes: dict[str, "AttributeSnapshot"] = {}
        self.log = get_logger("search_exp.mmr_diversifier")

    def run(self) -> int:
        """Execute the full pipeline: load, score, rerank, evaluate, and persist."""
        self._fetch_product_attributes()
        result_lists = self._collect_result_sets()

        if self.config.dry_run:
            self.log.info("dry-run mode; skipping write")
            return 0

        with timed_context(self.log, "full pipeline"):
            df = self._process_all_lists(result_lists)
            write_dataframe(df, self.OUTPUT_RESULTS, self.config.warehouse_uri, logger=self.log)

        self._log_summary(df)
        return 0

    def _fetch_product_attributes(self) -> None:
        sql = f"""
        SELECT product_id, brand, color_family AS color, category, price
        FROM {self.INPUT_PRODUCTS}
        WHERE is_active
        """
        df = read_dataframe(sql, self.config.warehouse_uri)
        df["brand"] = df["brand"].fillna("_unbranded").str.lower().str.strip()
        df["color"] = df["color"].fillna("_multi").str.lower()
        df["price_band"] = self._assign_price_bands(df)
        self.product_attributes = {
            row.product_id: AttributeSnapshot(row.brand, row.color, row.price_band)
            for row in df.itertuples(index=False)
        }
        self.log.info("cached attributes for %d products", len(self.product_attributes))

    def _assign_price_bands(self, df: pd.DataFrame) -> pd.Series:
        """Quantile-band price within category; very small categories get a single band."""
        def assign_band(group: pd.Series) -> pd.Series:
            unique_prices = group.nunique()
            if unique_prices < self.config.price_band_count:
                return pd.Series(0, index=group.index)
            return pd.qcut(group.rank(method="first"), self.config.price_band_count, labels=False)
        return df.groupby("category")["price"].transform(assign_band).fillna(0).astype(int)

    def _collect_result_sets(self) -> list["ResultListEnvelope"]:
        sql = f"""
        SELECT query_text, result_product_ids, result_scores, clicked_product_id
        FROM {self.INPUT_QUERY_LOGS}
        WHERE event_ts >= CURRENT_DATE - (:lookback_days * INTERVAL '1 day')
          AND cardinality(result_product_ids) > 0
        """
        params = {"lookback_days": self.config.lookback_days}
        df = read_dataframe(sql, self.config.warehouse_uri, params=params)
        df["query_norm"] = df["query_text"].apply(normalize_string)
        head_queries = (
            df["query_norm"]
            .value_counts()
            .reset_index()
            .query("count >= @self.config.minimum_query_frequency")
            .head(self.config.maximum_queries)["query_norm"]
        )
        df = df[df["query_norm"].isin(head_queries)]

        envelopes: list[ResultListEnvelope] = []
        for _, group in df.groupby("query_norm", sort=False):
            latest = group.iloc[-1]
            ids = latest["result_product_ids"][: self.config.pool_size]
            scores = [float(s) for s in latest["result_scores"][: self.config.pool_size]]
            if len(ids) != len(scores):
                self.log.warning("mismatch for %s (ids=%d, scores=%d)", latest["query_text"], len(ids), len(scores))
                continue
            clicks = group["clicked_product_id"].dropna().value_counts().to_dict()
            envelopes.append(ResultListEnvelope(latest["query_norm"], ids, scores, clicks, len(group)))
        self.log.info("loaded %d query result sets", len(envelopes))
        return envelopes

    def _process_all_lists(self, envelopes: list[ResultListEnvelope]) -> pd.DataFrame:
        frame_list: list[pd.DataFrame] = []
        for env in envelopes:
            if len(env.product_ids) < 2:
                continue
            df = self._rerank_one_list(env)
            frame_list.append(df)
        if not frame_list:
            return pd.DataFrame(columns=self._output_columns())
        out = pd.concat(frame_list, ignore_index=True)
        out["built_at"] = pd.Timestamp.utcnow()
        return out

    def _output_columns(self) -> list[str]:
        return [
            "query_norm", "position", "product_id", "original_position",
            "relevance", "mmr_score", "lambda"
        ]

    def _rerank_one_list(self, env: ResultListEnvelope) -> pd.DataFrame:
        scores = normalize_array(np.array(env.scores, dtype=float))
        reranked = self._greedy_mmr(env.product_ids, scores)

        orig_idx = {pid: i for i, pid in enumerate(env.product_ids)}
        rows = []
        for rank, (pid, score) in enumerate(reranked, start=1):
            orig = orig_idx[pid]
            rows.append({
                "query_norm": env.query_norm,
                "position": rank,
                "product_id": pid,
                "original_position": orig + 1,
                "relevance": float(scores[orig]),
                "mmr_score": float(score),
                "lambda": self.config.relevance_weight,
            })
        return pd.DataFrame(rows)

    def _greedy_mmr(self, ids: list[str], rel: np.ndarray) -> list[tuple[str, float]]:
        n = len(ids)
        remaining = np.ones(n, dtype=bool)
        max_sim = np.zeros(n)
        out: list[tuple[str, float]] = []

        for step in range(min(self.config.output_size, n)):
            if step < self.config.fixed_top:
                j = step
                val = float(rel[j])
            else:
                obj = self.config.relevance_weight * rel - (1 - self.config.relevance_weight) * max_sim
                obj[~remaining] = -np.inf
                j = int(np.argmax(obj))
                val = float(obj[j])
            out.append((ids[j], val))
            remaining[j] = False
            for i in np.flatnonzero(remaining):
                s = self._attribute_similarity(ids[i], ids[j])
                if s > max_sim[i]:
                    max_sim[i] = s
        return out

    def _attribute_similarity(self, a_id: str, b_id: str) -> float:
        a = self.product_attributes.get(a_id)
        b = self.product_attributes.get(b_id)
        if a is None or b is None:
            return 0.0
        return (
            self.config.brand_weight * (a.brand == b.brand)
            + self.config.color_weight * (a.color == b.color)
            + self.config.price_weight * (a.band == b.band)
        )

    def _log_summary(self, df: pd.DataFrame) -> None:
        moved = (df["position"] != df["original_position"]).mean()
        self.log.info("processed %d result sets; %.1f%% of positions changed", len(df), 100 * moved)


@dataclass(frozen=True)
class MMRConfiguration(ExperimentConfig):
    lookback_days: int = 7
    minimum_query_frequency: int = 50
    relevance_weight: float = 0.70
    pool_size: int = 48
    output_size: int = 24
    brand_weight: float = 0.45
    color_weight: float = 0.35
    price_weight: float = 0.20
    price_band_count: int = 4
    maximum_queries: int = 20_000
    fixed_top: int = 1

    def __post_init__(self) -> None:
        if not 0.0 <= self.relevance_weight <= 1.0:
            raise ValueError("relevance_weight must be between 0 and 1")
        total = self.brand_weight + self.color_weight + self.price_weight
        if abs(total - 1.0) > 1e-6:
            raise ValueError("attribute weights must sum to 1")
        if self.output_size > self.pool_size:
            raise ValueError("output_size cannot exceed pool_size")
        if not (0 <= self.fixed_top <= self.output_size):
            raise ValueError("fixed_top must be between 0 and output_size")


@dataclass(frozen=True)
class AttributeSnapshot:
    brand: str
    color: str
    band: int


@dataclass
class ResultListEnvelope:
    query_norm: str
    product_ids: list[str]
    scores: list[float]
    clicks: dict[str, int]
    impressions: int


def normalize_array(arr: np.ndarray) -> np.ndarray:
    minimum, maximum = arr.min(), arr.max()
    if maximum - minimum < 1e-12:
        return np.ones_like(arr, dtype=float)
    return (arr - minimum) / (maximum - minimum)
