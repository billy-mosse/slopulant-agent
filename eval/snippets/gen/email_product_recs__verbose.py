class PostPurchaseRecommendationEngine:
    """
    Generates personalized product suggestions for post-purchase email campaigns.
    
    After a customer completes a purchase, this system identifies the most relevant
    additional items from our active inventory by analyzing textual similarity between
    their recently purchased item and our full product catalog. We use TF-IDF vectorization
    with bigram features to capture semantic overlap in titles and descriptions, then
    filter recommendations to exclude items the customer has already purchased.
    
    The resulting recommendations are ranked by cosine similarity and delivered as part
    of our marketing email sequence to increase cross-sell opportunities.
    """

    def __init__(self, top_k: int = 6, min_similarity: float = 0.08, batch_size: int = 2048):
        self.top_k = top_k
        self.min_similarity = min_similarity
        self.batch_size = batch_size
        self._vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), min_df=2, max_df=0.5, sublinear_tf=True, stop_words="english"
        )
        self._catalog_vectors = None
        self._product_index = None
        self._eligible_mask = None

    def _prepare_catalog(self, catalog_df: pd.DataFrame) -> None:
        """Build TF-IDF representation of product content for similarity search."""
        text_columns = ["title", "title", "description"]
        combined_text = pd.concat([catalog_df[col].fillna("").str.lower() for col in text_columns], axis=0)
        combined_text = combined_text.str.replace(r"<[^>]+>", " ", regex=True)
        combined_text = combined_text.str.replace(r"\b\d+\s*(?:tc|thread count)\b", " threadcount ", regex=True)
        combined_text = combined_text.str.replace(r"[^a-z0-9 ]+", " ", regex=True).str.replace(r"\s+", " ", regex=True)
        
        self._catalog_vectors = normalize(self._vectorizer.fit_transform(combined_text))
        self._product_index = pd.Series(np.arange(len(catalog_df)), index=catalog_df["product_id"])
        self._eligible_mask = (catalog_df["is_active"] & catalog_df["in_stock"]).to_numpy()

    def _compute_similarities(self, anchor_ids: np.ndarray) -> dict:
        """Calculate similarity scores between anchor items and all catalog items."""
        row_indices = self._product_index.loc[anchor_ids].to_numpy()
        similarity_matrix = (self._catalog_vectors[row_indices] @ self._catalog_vectors.T).toarray()
        similarity_matrix[:, ~self._eligible_mask] = -1.0
        np.fill_diagonal(similarity_matrix, -1.0)  # exclude anchor items themselves

        recommendations = {}
        for i, pid in enumerate(anchor_ids):
            candidates = np.argpartition(-similarity_matrix[i], kth=min(self.top_k * 4, len(similarity_matrix[i]) - 1))[:self.top_k * 4]
            sorted_candidates = candidates[np.argsort(-similarity_matrix[i, candidates])]
            valid = [(int(pid), float(similarity_matrix[i, j])) 
                     for j in sorted_candidates if similarity_matrix[i, j] >= self.min_similarity]
            recommendations[pid] = valid
        return recommendations

    def generate_recommendations(self, engine, reference_date: pd.Timestamp, history_window_days: int = 730) -> pd.DataFrame:
        """Orchestrate the full recommendation pipeline for recent customers."""
        catalog_df = pd.read_sql("SELECT product_id, title, description, is_active, in_stock FROM catalog.products", engine)
        self._prepare_catalog(catalog_df)

        recent_purchases = pd.read_sql(
            "SELECT customer_id, product_id, order_ts FROM orders.lines WHERE status = 'completed' AND order_ts >= %(start)s",
            engine, params={"start": reference_date - pd.Timedelta(days=history_window_days)}, parse_dates=["order_ts"]
        )
        customer_purchases = recent_purchases.groupby("customer_id")["product_id"].apply(set)
        recent_customers = recent_purchases[recent_purchases.order_ts >= reference_date - pd.Timedelta(days=30)]
        last_items = recent_customers.sort_values("order_ts").groupby("customer_id").tail(1)
        last_items = last_items[last_items.product_id.isin(self._product_index.index)]

        unique_anchors = last_items.product_id.unique()
        all_recommendations = {}
        for start in range(0, len(unique_anchors), self.batch_size):
            batch = unique_anchors[start:start + self.batch_size]
            all_recommendations.update(self._compute_similarities(batch))

        output_records = []
        for customer_id, row in last_items.iterrows():
            owned = customer_purchases.get(customer_id, set())
            candidates = [item for item in all_recommendations.get(row.product_id, []) if item[0] not in owned][:self.top_k]
            for rank, (product_id, score) in enumerate(candidates, 1):
                output_records.append((customer_id, row.product_id, product_id, rank, round(score, 4)))

        result = pd.DataFrame(output_records, columns=["customer_id", "anchor_product_id", "rec_product_id", "rank", "similarity"])
        result["generated_at"] = reference_date
        return result
