import sqlite3
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from rss_pipeline import (
    build_automatic_query,
    build_passage_text,
    clear_vector_db,
    embed_unembedded_articles,
    embed_query_text,
    extract_html_text,
    get_sql_stats,
    ingest_rss,
    init_database,
    load_embedding_model,
    load_pipeline_config,
    normalize_article_url,
    resolve_embedding_device,
    run_automatic_retrieval,
    search_similar_articles,
)


class RssPipelineTest(unittest.TestCase):
    def test_normalize_article_url_removes_tracking_parameters(self):
        self.assertEqual(
            normalize_article_url(
                "HTTPS://Example.TEST/news/?utm_source=rss&id=42#section"
            ),
            "https://example.test/news?id=42",
        )

    def test_pipeline_config_supports_embedding_device_override(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "config.json"
            config_path.write_text(
                '{"embedding_model": "nvidia/Nemotron-3-Embed-1B-BF16", "embedding_device": "auto"}',
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"EMBEDDING_DEVICE": "cpu"}):
                config = load_pipeline_config(config_path)
        self.assertEqual(config["embedding_model"], "nvidia/Nemotron-3-Embed-1B-BF16")
        self.assertEqual(config["embedding_device"], "cpu")

    def test_load_embedding_model_uses_configured_device_without_changing_model(self):
        calls = []

        class FakeSentenceTransformer:
            def __init__(self, model_name, **kwargs):
                calls.append((model_name, kwargs))

        fake_module = types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer)
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            load_embedding_model(
                "nvidia/Nemotron-3-Embed-1B-BF16",
                local_files_only=True,
                device="cpu",
            )

        self.assertEqual(calls[0][0], "nvidia/Nemotron-3-Embed-1B-BF16")
        self.assertTrue(calls[0][1]["local_files_only"])
        self.assertEqual(calls[0][1]["device"], "cpu")

    def test_resolve_embedding_device_rejects_unknown_values(self):
        with self.assertRaises(ValueError):
            resolve_embedding_device("fallback-small-cpu-model")

    def test_init_database_creates_article_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "articles.sqlite3"
            init_database(database_path)
            connection = sqlite3.connect(database_path)
            try:
                table = connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name='Article'"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(table[0], "Article")
            self.assertEqual(get_sql_stats(database_path), {"articles": 0, "embedded": 0, "pending": 0})

            connection = sqlite3.connect(database_path)
            try:
                columns = {
                    row[1]
                    for row in connection.execute("PRAGMA table_info(Article)")
                }
            finally:
                connection.close()
            self.assertNotIn("cleanup_status", columns)
            self.assertNotIn("cleanup_error", columns)

    def test_init_database_migrates_legacy_schema(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "legacy.sqlite3"
            connection = sqlite3.connect(database_path)
            try:
                connection.execute(
                    """
                    CREATE TABLE Article(
                        Article_URL TEXT PRIMARY KEY,
                        Feed_URL TEXT NOT NULL,
                        Article_Title TEXT,
                        Article_Description TEXT,
                        published_date TEXT,
                        Article_Text TEXT,
                        embedded INTEGER NOT NULL DEFAULT 0,
                        ingested_at TEXT NOT NULL,
                        cleanup_status TEXT NOT NULL DEFAULT 'unknown',
                        cleanup_error TEXT
                    )
                    """
                )
                connection.commit()
            finally:
                connection.close()

            init_database(database_path)
            connection = sqlite3.connect(database_path)
            try:
                columns = {row[1] for row in connection.execute("PRAGMA table_info(Article)")}
            finally:
                connection.close()
            self.assertNotIn("cleanup_status", columns)
            self.assertNotIn("cleanup_error", columns)

    def test_extracts_html_without_semantic_rewriting(self):
        self.assertEqual(extract_html_text(""), "")
        self.assertEqual(
            extract_html_text(
                "<article>Hello <strong>world</strong>.</article>"
                "<script>tracking()</script>"
            ),
            "Hello world.",
        )

    def test_ingestion_stores_extracted_content_without_semantic_rewriting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "articles.sqlite3"
            with patch(
                "rss_pipeline._fetch_feed",
                return_value=[
                    {
                        "title": "Raw article",
                        "link": "https://example.test/article",
                        "published": "2026-01-01",
                        "summary": "<p>Raw <strong>summary</strong>.</p>",
                    }
                ],
            ), patch(
                "rss_pipeline.fetch_article_text",
                return_value="Raw article text without semantic rewriting.",
            ), patch("rss_pipeline._configure_logger"):
                result = ingest_rss(
                    ["https://example.test/feed.xml"],
                    database_path=database_path,
                )

            self.assertEqual(result["inserted_count"], 1)
            connection = sqlite3.connect(database_path)
            try:
                article = connection.execute(
                    "SELECT Article_Description, Article_Text FROM Article"
                ).fetchone()
            finally:
                connection.close()
            self.assertEqual(article, ("Raw summary.", "Raw article text without semantic rewriting."))

    def test_builds_passage_input_for_documents(self):
        self.assertEqual(
            build_passage_text("Titel", "Beschreibung"),
            "passage: Titel Beschreibung",
        )

    def test_uses_query_prompt_for_query_embeddings(self):
        class FakeVector:
            def tolist(self):
                return [1.0, 2.0]

        class FakeModel:
            def __init__(self):
                self.kwargs = None

            def encode(self, value, **kwargs):
                self.value = value
                self.kwargs = kwargs
                return FakeVector()

        model = FakeModel()
        self.assertEqual(embed_query_text(model, "Suchtext"), [1.0, 2.0])
        self.assertEqual(model.value, "Suchtext")
        self.assertEqual(model.kwargs["prompt_name"], "query")
        self.assertTrue(model.kwargs["normalize_embeddings"])

    def test_builds_automatic_queries_and_skips_incomplete_mode_rows(self):
        row = {
            "Comment_ID": "c-1",
            "Comment_Text": "Comment about the topic",
            "Video_Title": "Video about the topic",
            "Video_Description": "Video description",
        }
        self.assertEqual(build_automatic_query(row, "Q1"), "Comment about the topic")
        self.assertEqual(
            build_automatic_query(row, "Q2"),
            "Video about the topic\nVideo description",
        )
        self.assertEqual(
            build_automatic_query(row, "Q3"),
            "Comment about the topic\nVideo about the topic\nVideo description",
        )
        self.assertIsNone(build_automatic_query({**row, "Video_Description": ""}, "Q2"))
        self.assertIsNone(build_automatic_query({**row, "Comment_ID": ""}, "Q1"))

    def test_runs_automatic_retrieval_and_exports_best_match_fields(self):
        rows = [
            {
                "Comment_ID": "c-1",
                "Comment_Text": "Comment about the topic",
                "Video_Title": "",
                "Video_Description": "",
            },
            {
                "Comment_ID": "c-2",
                "Comment_Text": "",
                "Video_Title": "Video",
                "Video_Description": "",
            },
        ]
        match = {
            "rrf_score": 0.0123,
            "Article_Title": "Matching article",
            "Article_Description": "Article description",
            "Article_Text": "Article text",
        }
        with patch("rss_pipeline.search_similar_articles", return_value=[match]) as search:
            result = run_automatic_retrieval(
                rows,
                mode="Q1",
                vector_weight=0.3,
                bm25_weight=0.7,
                embedding_model=object(),
            )

        self.assertEqual(result["processed_count"], 1)
        self.assertEqual(result["skipped_count"], 1)
        self.assertEqual(result["results"][0]["Mode"], "Q1")
        self.assertEqual(result["results"][0]["VectorWeight"], 0.3)
        self.assertEqual(result["results"][0]["BM25Weight"], 0.7)
        self.assertEqual(result["results"][0]["ArticleTitle"], "Matching article")
        search.assert_called_once()
        self.assertEqual(search.call_args.args[0], "Comment about the topic")

    def test_clear_vector_db_resets_sql_embedding_flags(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "articles.sqlite3"
            init_database(database_path)
            connection = sqlite3.connect(database_path)
            try:
                connection.execute(
                    """
                    INSERT INTO Article(
                        Article_URL, Feed_URL, Article_Title, embedded, ingested_at
                    ) VALUES (?, ?, ?, 1, ?)
                    """,
                    ("https://example.test/article", "https://example.test/feed", "Titel", "now"),
                )
                connection.commit()
            finally:
                connection.close()

            result = clear_vector_db(
                vector_db_path=Path(temp_dir) / "missing-vdb",
                database_path=database_path,
            )
            self.assertFalse(result["dropped"])
            self.assertEqual(result["reset_sql_rows"], 1)
            self.assertEqual(get_sql_stats(database_path)["embedded"], 0)

    def test_embeddings_store_article_text_and_create_german_fts_index(self):
        class FakeModel:
            def encode(self, value, **kwargs):
                return [1.0, 0.0]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "articles.sqlite3"
            vector_db_path = root / "vectors"
            init_database(database_path)
            connection = sqlite3.connect(database_path)
            try:
                connection.execute(
                    """
                    INSERT INTO Article(
                        Article_URL, Feed_URL, Article_Title, Article_Description,
                        Article_Text, embedded, ingested_at
                    ) VALUES (?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        "https://example.test/article",
                        "https://example.test/feed",
                        "Berlin Wahl",
                        "Politik in Berlin",
                        "Die Wahl in Berlin fand gestern statt.",
                        "now",
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            result = embed_unembedded_articles(
                database_path=database_path,
                vector_db_path=vector_db_path,
                embedding_model=FakeModel(),
            )
            self.assertEqual(result["stored_count"], 1)
            self.assertTrue(result["fts_index_created"])

            import lancedb

            table = lancedb.connect(str(vector_db_path)).open_table("rss_articles")
            self.assertIn("article_text", table.schema.names)
            self.assertTrue(any(index.index_type == "FTS" for index in table.list_indices()))

            results = search_similar_articles(
                "Wahl Berlin",
                database_path=database_path,
                vector_db_path=vector_db_path,
                embedding_model=FakeModel(),
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0]["bm25_rank"], 1)
            self.assertAlmostEqual(results[0]["rrf_score"], 1.0 / 61.0, places=8)

    def test_embedding_run_rebuilds_legacy_vector_table(self):
        class FakeModel:
            def encode(self, value, **kwargs):
                return [1.0, 0.0]

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            database_path = root / "articles.sqlite3"
            vector_db_path = root / "vectors"
            init_database(database_path)
            connection = sqlite3.connect(database_path)
            try:
                connection.execute(
                    """
                    INSERT INTO Article(
                        Article_URL, Feed_URL, Article_Title, Article_Description,
                        Article_Text, embedded, ingested_at
                    ) VALUES (?, ?, ?, ?, ?, 1, ?)
                    """,
                    (
                        "https://example.test/legacy",
                        "https://example.test/feed",
                        "Legacy",
                        "Beschreibung",
                        "Full text for BM25",
                        "now",
                    ),
                )
                connection.commit()
            finally:
                connection.close()

            import lancedb

            db = lancedb.connect(str(vector_db_path))
            db.create_table("rss_articles", data=[{"article_url": "https://example.test/legacy", "vector": [1.0, 0.0]}])

            result = embed_unembedded_articles(
                database_path=database_path,
                vector_db_path=vector_db_path,
                embedding_model=FakeModel(),
            )
            self.assertTrue(result["rebuilt"])
            self.assertEqual(result["stored_count"], 1)
            self.assertEqual(get_sql_stats(database_path)["embedded"], 1)


if __name__ == "__main__":
    unittest.main()
