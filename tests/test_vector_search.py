"""The sqlite-vec topic index (watcher/db.py topic_vec) agrees with plain numpy and stays in
sync with the topics table.  python -m unittest tests.test_vector_search"""
import os
import sqlite3
import tempfile
import unittest

import numpy as np

TMP = tempfile.TemporaryDirectory()
os.environ["DATA_DIR"] = TMP.name  # before importing watcher.config

from watcher import config, db, keywords, topics  # noqa: E402

DIM = 384


def unit(v):
    return (v / np.linalg.norm(v)).astype(np.float32)


def entry(folder, vectors):
    return {"folder": folder, "n_chunks": 1, "func_names": [], "func_embeddings": np.zeros((0, DIM), np.float32),
            "topics": [{"name": f"{folder}-{i}", "description": f"{folder} topic {i}", "embedding": v,
                        "keywords": [folder, f"kw{i}"], "inputs": [], "outputs": []} for i, v in enumerate(vectors)]}


@unittest.skipIf(db.sqlite_vec is None, "sqlite-vec not installed")
class VectorSearchTest(unittest.TestCase):
    def setUp(self):
        if config.DB_PATH.exists():
            config.DB_PATH.unlink()
        self.conn = db.connect()
        self.rng = np.random.default_rng(0)
        self.vectors = {f"folder{i}": [unit(self.rng.normal(size=DIM)) for _ in range(1 + i % 3)] for i in range(12)}
        for i, (folder, vs) in enumerate(self.vectors.items()):
            db.put_folder(self.conn, f"tree{i}", entry(folder, vs))

    def tearDown(self):
        self.conn.close()

    def test_index_mirrors_topics(self):
        self.assertTrue(db.vector_search(self.conn))
        n = sum(len(v) for v in self.vectors.values())
        self.assertEqual(self.conn.execute("SELECT count(*) FROM topic_vec").fetchone()[0], n)

    def test_knn_equals_numpy_cosine(self):
        query = unit(self.rng.normal(size=DIM))
        sims = db.topic_similarities(self.conn, query)
        rows = self.conn.execute("SELECT rowid, embedding FROM topics").fetchall()
        self.assertEqual(set(sims), {r["rowid"] for r in rows})
        for r in rows:
            self.assertAlmostEqual(sims[r["rowid"]], float(np.frombuffer(r["embedding"], np.float32) @ query), places=5)

    def test_nearest_neighbour_is_the_near_duplicate(self):
        target = self.vectors["folder5"][0]
        query = unit(target + 0.05 * self.rng.normal(size=DIM))  # a slightly reworded description
        sims = db.topic_similarities(self.conn, query)
        best = max(sims, key=sims.get)
        name = self.conn.execute("SELECT name FROM topics WHERE rowid = ?", (best,)).fetchone()[0]
        self.assertEqual(name, "folder5-0")

    def test_rewriting_a_folder_replaces_its_vectors(self):
        new = [unit(self.rng.normal(size=DIM))]
        db.put_folder(self.conn, "tree0", entry("folder0", new))
        n = sum(len(v) for k, v in self.vectors.items() if k != "folder0") + 1
        self.assertEqual(self.conn.execute("SELECT count(*) FROM topic_vec").fetchone()[0], n)
        sims = db.topic_similarities(self.conn, new[0])
        rowid = self.conn.execute("SELECT rowid FROM topics WHERE tree_sha = 'tree0'").fetchone()[0]
        self.assertAlmostEqual(sims[rowid], 1.0, places=5)

    def test_existing_database_is_backfilled_on_connect(self):
        self.conn.execute("DROP TABLE topic_vec")
        self.conn.commit()
        conn = db.connect()
        self.assertEqual(conn.execute("SELECT count(*) FROM topic_vec").fetchone()[0],
                         conn.execute("SELECT count(*) FROM topics").fetchone()[0])

    def test_compare_gives_the_same_signals_with_and_without_the_index(self):
        base = {f: db.get_folder(self.conn, f"tree{i}") for i, f in enumerate(self.vectors)}
        pr = entry("new", [unit(self.vectors["folder3"][0] + 0.3 * self.rng.normal(size=DIM)), unit(self.rng.normal(size=DIM))])
        idf = keywords.TfidfIndex([t["keywords"] for e in base.values() for t in e["topics"]])
        sims = [db.topic_similarities(self.conn, t["embedding"]) for t in pr["topics"]]
        for folder, other in base.items():
            with_index, plain = topics.compare(pr, other, idf, sims), topics.compare(pr, other, idf)
            self.assertEqual(with_index["repo_topic"], plain["repo_topic"])
            self.assertAlmostEqual(with_index["desc_score"], plain["desc_score"], places=5)
            self.assertAlmostEqual(with_index["score"], plain["score"], places=5)


if __name__ == "__main__":
    unittest.main()
