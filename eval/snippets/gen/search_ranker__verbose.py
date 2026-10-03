class SearchRankingModelTrainer:
    """
    trains a learning-to-rank model for second-stage search re-ranking.
    
    this component combines multi-modal signals—semantic similarity via dual-tower embeddings,
    classical text relevance (BM25), query intent priors, historical user engagement (CTR),
    and product attributes—into a unified ranking function optimized for ndcg@10 using
    lambdarank with position-bias correction.
    """

    def __init__(self, config_path: str):
        self.config = self._load_configuration(config_path)
        self.feature_pipeline = FeatureTransformationPipeline(self.config)

    def execute_training_workflow(self, database_uri: str, output_directory: str) -> None:
        data_loader = HistoricalEventDataLoader(database_uri, self.config)
        raw_sessions = data_loader.fetch_training_data()
        feature_matrix = self.feature_pipeline.apply_all_transformations(raw_sessions)
        filtered_queries = self._filter_sparse_queries(feature_matrix)
        train_set, val_set = self._split_train_and_validation(filtered_queries)

        position_bias = PositionBiasEstimator.estimate_from(train_set)
        dataset_builder = RankerDatasetBuilder(position_bias)
        train_lgb, val_lgb = dataset_builder.build_train_and_val_sets(train_set, val_set)

        trainer = LightGBMRanker(self.config["lightgbm"])
        model = trainer.fit(train_lgb, val_lgb)
        metrics = {"ndcg@10": model.best_score["valid_0"]["ndcg@10"], "iteration": model.best_iteration}

        ModelArtifactManager.save(model, metrics, output_directory)
        logger.info("training complete — best ndcg@10: %.4f", metrics["ndcg@10"])

    def _load_configuration(self, config_path: str) -> dict:
        import yaml
        from pathlib import Path
        return yaml.safe_load(Path(config_path).read_text())

    def _filter_sparse_queries(self, frame: pd.DataFrame) -> pd.DataFrame:
        min_support = self.config["data"]["min_impressions_per_query"]
        query_counts = frame.groupby("query_id").size()
        valid_query_ids = query_counts[query_counts >= min_support].index
        return frame[frame["query_id"].isin(valid_query_ids)]

    def _split_train_and_validation(self, frame: pd.DataFrame) -> tuple:
        holdout = self.config["validation"]["holdout_days"]
        cutoff = frame["event_date"].max() - pd.Timedelta(days=holdout)
        return frame[frame["event_date"] < cutoff], frame[frame["event_date"] >= cutoff]


class FeatureTransformationPipeline:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.priors = cfg["features"]

    def apply_all_transformations(self, events: pd.DataFrame) -> pd.DataFrame:
        tables = self.cfg["tables"]
        embeddings = self._load_embeddings(tables["product_embeddings"], tables["query_intents"], events)
        enriched = events.copy()

        enriched = FeatureBuilders.cosine_similarity(enriched, embeddings["product"], embeddings["query"])
        enriched = FeatureBuilders.bm25_score(enriched, self.priors["bm25_k1"], self.priors["bm25_b"])
        enriched = FeatureBuilders.intent_match(enriched, embeddings["intents"])
        enriched = FeatureBuilders.smoothed_ctr(enriched, events, self.priors["ctr_prior_clicks"], self.priors["ctr_prior_impressions"])
        enriched = FeatureBuilders.price_z_score(enriched)
        enriched = FeatureBuilders.review_features(enriched)
        return enriched

    def _load_embeddings(self, emb_table: str, intent_table: str, events: pd.DataFrame) -> dict:
        from sqlalchemy import create_engine
        engine = create_engine(events.engine_uri)
        product_emb = pd.read_sql(f"SELECT sku, embedding FROM {emb_table}", engine)
        query_emb = events[["query_text", "query_embedding"]].drop_duplicates()
        query_emb = query_emb.rename(columns={"query_embedding": "embedding"})
        intent_df = pd.read_sql(f"SELECT query_text, top_categories FROM {intent_table}", engine)
        return {"product": product_emb, "query": query_emb, "intents": intent_df}


class FeatureBuilders:
    @staticmethod
    def cosine_similarity(events: pd.DataFrame, product_embs: pd.DataFrame, query_embs: pd.DataFrame) -> pd.DataFrame:
        p_vecs = product_embs.set_index("sku")["embedding"].map(np.asarray)
        q_vecs = query_embs.set_index("query_text")["embedding"].map(np.asarray)
        pv = np.stack(events["sku"].map(p_vecs).values)
        qv = np.stack(events["query_text"].map(q_vecs).values)
        dot = np.sum(pv * qv, axis=1)
        norm = np.linalg.norm(pv, axis=1) * np.linalg.norm(qv, axis=1) + 1e-9
        events["semantic_similarity"] = dot / norm
        return events

    @staticmethod
    def bm25_score(events: pd.DataFrame, k1: float, b: float) -> pd.DataFrame:
        titles = events.drop_duplicates("sku").set_index("sku")["title"].fillna("").str.lower().str.split()
        n = len(titles)
        avgdl = titles.map(len).mean() or 1.0
        df = Counter(term for toks in titles for term in set(toks))
        idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}
        def bm25_score_for_row(qry: str, sku: str) -> float:
            tokens = titles.get(sku, [])
            tf = Counter(tokens)
            norm = k1 * (1 - b + b * len(tokens) / avgdl)
            return sum(idf.get(t, 0.0) * tf[t] * (k1 + 1) / (tf[t] + norm) for t in qry.lower().split() if t in tf)
        events["text_relevance"] = [bm25_score_for_row(q, s) for q, s in zip(events["query_text"], events["sku"])]
        return events

    @staticmethod
    def intent_match(events: pd.DataFrame, intents: pd.DataFrame) -> pd.DataFrame:
        intent_probs = {
            row.query_text: json.loads(row.top_categories)
            for row in intents.itertuples()
        }
        events["intent_category_affinity"] = [
            intent_probs.get(q, {}).get(c, 0.0) for q, c in zip(events["query_text"], events["category_id"])
        ]
        return events

    @staticmethod
    def smoothed_ctr(events: pd.DataFrame, history: pd.DataFrame, a: float, b: float) -> pd.DataFrame:
        agg = history.groupby(["query_text", "sku"]).agg(clicks=("clicked", "sum"), imps=("clicked", "size")).reset_index()
        agg["empirical_ctr"] = (agg["clicks"] + a) / (agg["imps"] + a + b)
        events = events.merge(agg[["query_text", "sku", "empirical_ctr"]], on=["query_text", "sku"], how="left")
        events["empirical_ctr"] = events["empirical_ctr"].fillna(a / (a + b))
        return events

    @staticmethod
    def price_z_score(events: pd.DataFrame) -> pd.DataFrame:
        g = events.groupby("query_id")["price"]
        events["price_normalized"] = (
            (events["price"] - g.transform("mean")) / g.transform("std").replace(0, np.nan)
        ).fillna(0.0)
        return events

    @staticmethod
    def review_features(events: pd.DataFrame) -> pd.DataFrame:
        median_rating = events["avg_rating"].median()
        events["product_quality"] = events["avg_rating"].fillna(median_rating)
        events["review_scarcity"] = np.log1p(events["review_count"].fillna(0))
        return events


class PositionBiasEstimator:
    @staticmethod
    def estimate_from(train_data: pd.DataFrame) -> pd.Series:
        baseline = train_data.groupby("position")["clicked"].mean()
        smoothed = baseline.rolling(3, min_periods=1, center=True).mean()
        propensity = (smoothed / smoothed.iloc[0]).clip(lower=0.02)
        logger.info("learned position propensity: %s", propensity.head(10).round(3).to_dict())
        return propensity


class RankerDatasetBuilder:
    def __init__(self, pos_bias: pd.Series):
        self.bias = pos_bias

    def build_train_and_val_sets(self, train: pd.DataFrame, val: pd.DataFrame) -> tuple:
        for subset in (train, val):
            subset.sort_values(["query_id", "position"], inplace=True)
        groups = train.groupby("query_id", sort=False).size().values
        ipw_weights = self._compute_inverse_propensity_weights(train)
        train_set = lgb.Dataset(
            train[["semantic_similarity", "text_relevance", "intent_category_affinity", "empirical_ctr", "price_normalized", "product_quality", "review_scarcity"]],
            label=train["clicked"].astype(int), group=groups, weight=ipw_weights
        )
        val_groups = val.groupby("query_id", sort=False).size().values
        val_ipw = self._compute_inverse_propensity_weights(val)
        val_set = lgb.Dataset(
            val[["semantic_similarity", "text_relevance", "intent_category_affinity", "empirical_ctr", "price_normalized", "product_quality", "review_scarcity"]],
            label=val["clicked"].astype(int), group=val_groups, weight=val_ipw
        )
        return train_set, val_set

    def _compute_inverse_propensity_weights(self, frame: pd.DataFrame) -> np.ndarray:
        ipw = np.where(frame["clicked"] == 1, 1.0 / frame["position"].map(self.bias).fillna(self.bias.min()), 1.0)
        return np.clip(ipw, 1.0, 20.0)


class LightGBMRanker:
    def __init__(self, hyperparameters: dict):
        self.params = {k: v for k, v in hyperparameters.items() if k not in ("num_boost_round", "early_stopping_rounds")}

    def fit(self, train_set: lgb.Dataset, val_set: lgb.Dataset) -> lgb.Booster:
        return lgb.train(
            self.params,
            train_set,
            num_boost_round=hyperparameters["num_boost_round"],
            valid_sets=[val_set],
            callbacks=[
                lgb.early_stopping(hyperparameters["early_stopping_rounds"]),
                lgb.log_evaluation(50)
            ]
        )


class HistoricalEventDataLoader:
    def __init__(self, dsn: str, cfg: dict):
        self.engine = create_engine(dsn)
        self.cfg = cfg

    def fetch_training_data(self) -> pd.DataFrame:
        t = self.cfg["tables"]
        days = self.cfg["data"]["lookback_days"]
        max_pos = self.cfg["data"]["max_position"]
        query = f"""
            SELECT query_id, query_text, sku, position, clicked, title, category_id,
                   price, avg_rating, review_count, query_embedding, event_date
            FROM {t["click_logs"]}
            WHERE event_date >= CURRENT_DATE - {days}
              AND position <= {max_pos}
        """
        df = pd.read_sql(query, self.engine)
        df.engine_uri = self.engine.url.render_as_string(hide_password=False)
        logger.info("loaded %d impressions", len(df))
        return df


class ModelArtifactManager:
    @staticmethod
    def save(model: lgb.Booster, metrics: dict, out_dir: str) -> None:
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        model.save_model(str(out_path / "ranking_model.txt"))
        importance = dict(zip(
            ["semantic_similarity", "text_relevance", "intent_category_affinity", "empirical_ctr", "price_normalized", "product_quality", "review_scarcity"],
            model.feature_importance("gain").round(2).tolist()
        ))
        (out_path / "feature_gains.json").write_text(json.dumps(importance, indent=2))
        # (model registration logic elided for brevity — writes metadata to central registry)
        logger.info("saved model artifact to %s", out_dir)
