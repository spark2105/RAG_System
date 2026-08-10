from __future__ import annotations

import json
import logging
import math
import os
import re
import sqlite3
import ssl
from collections.abc import Callable, Iterable, Mapping
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATABASE_PATH = PROJECT_ROOT / "data" / "sql" / "articles.sqlite3"
VECTOR_DB_PATH = PROJECT_ROOT / "data" / "vector" / "lancedb_rss"
PIPELINE_CONFIG_PATH = CONFIG_DIR / "RAG_Pipeline_Config.json"
RSS_CONFIG_PATH = CONFIG_DIR / "RSS_URL.txt"
DEFAULT_TABLE_NAME = "rss_articles"
DEFAULT_EMBEDDING_MODEL = "nvidia/Nemotron-3-Embed-1B-BF16"
DOCUMENT_PROMPT_PREFIX = "passage: "
DEFAULT_FTS_COLUMN = "article_text"
DEFAULT_FTS_INDEX_NAME = "article_text_fts"
DEFAULT_FTS_LANGUAGE = "German"
DEFAULT_VECTOR_CANDIDATE_LIMIT = 25
DEFAULT_BM25_CANDIDATE_LIMIT = 25
DEFAULT_EMBEDDING_BATCH_SIZE = 32
DEFAULT_RRF_K = 60
DEFAULT_VECTOR_WEIGHT = 0.6
DEFAULT_BM25_WEIGHT = 0.4
AUTOMATIC_RETRIEVAL_MODES = ("Q1", "Q2", "Q3")
AUTOMATIC_INPUT_COLUMNS = (
    "Comment_ID",
    "Comment_Text",
    "Video_Title",
    "Video_Description",
)
AUTOMATIC_OUTPUT_COLUMNS = (
    "Mode",
    "VectorWeight",
    "BM25Weight",
    "Comment_ID",
    "Comment_Text",
    "Video_Title",
    "Video_Description",
    "RRFScore",
    "ArticleTitle",
    "ArticleDescription",
    "ArticleText",
)
ProgressCallback = Callable[[str], None]


def load_pipeline_config(path: str | Path = PIPELINE_CONFIG_PATH) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as config_file:
        value = json.load(config_file)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration must be a JSON object: {path}")
    if os.getenv("EMBEDDING_MODEL"):
        value["embedding_model"] = os.environ["EMBEDDING_MODEL"]
    if os.getenv("VECTOR_DB_PATH"):
        value["vector_db_path"] = os.environ["VECTOR_DB_PATH"]
    return value


def save_feed_urls(urls: Iterable[str], path: str | Path = RSS_CONFIG_PATH) -> Path:
    cleaned = []
    seen = set()
    for value in urls:
        url = str(value).strip()
        if not url or url.startswith("#") or url in seen:
            continue
        validate_feed_url(url)
        cleaned.append(url)
        seen.add(url)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(cleaned) + ("\n" if cleaned else ""), encoding="utf-8")
    return target


def load_feed_urls(path: str | Path = RSS_CONFIG_PATH) -> list[str]:
    target = Path(path)
    if not target.exists():
        return []
    return [
        line.strip()
        for line in target.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def validate_feed_url(url: str) -> str:
    cleaned = str(url).strip()
    parts = urlsplit(cleaned)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError(f"Invalid RSS URL: {url}")
    return cleaned


def normalize_article_url(url: str) -> str:
    cleaned = validate_feed_url(url)
    parts = urlsplit(cleaned)
    path = parts.path or "/"
    if path != "/":
        path = path.rstrip("/")
    query = [
        (key, value)
        for key, value in parse_qsl(parts.query, keep_blank_values=True)
        if not key.lower().startswith(("utm_", "fbclid", "gclid"))
    ]
    normalized = urlunsplit(
        (parts.scheme.lower(), parts.netloc.lower(), path, urlencode(query), "")
    )
    return normalized.rstrip("/") or normalized


def init_database(path: str | Path = DATABASE_PATH) -> Path:
    database_path = Path(path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    try:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS Article (
                Article_URL TEXT PRIMARY KEY,
                Feed_URL TEXT NOT NULL,
                Article_Title TEXT,
                Article_Description TEXT,
                published_date TEXT,
                Article_Text TEXT,
                embedded INTEGER NOT NULL DEFAULT 0 CHECK (embedded IN (0, 1)),
                ingested_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_article_embedded ON Article(embedded);
            CREATE INDEX IF NOT EXISTS idx_article_published ON Article(published_date);
            """
        )
        existing_columns = {
            str(row[1]) for row in connection.execute("PRAGMA table_info(Article)")
        }
        legacy_cleanup_columns = {"cleanup_status", "cleanup_error"} & existing_columns
        if legacy_cleanup_columns:
            connection.execute("DROP INDEX IF EXISTS idx_article_cleanup_status")
            for column in sorted(legacy_cleanup_columns):
                connection.execute(f"ALTER TABLE Article DROP COLUMN {column}")
        connection.commit()
    finally:
        connection.close()
    return database_path


def _configure_logger() -> logging.Logger:
    log_path = PROJECT_ROOT / "data" / "logs" / "rss_ingestion.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("rag_system.rss")
    if not logger.handlers:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def _fetch_feed(url: str, timeout: int) -> list[dict[str, str]]:
    import feedparser
    import certifi

    request = Request(url, headers={"User-Agent": "RAG_System/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    try:
        with urlopen(request, timeout=timeout, context=context) as response:
            payload = response.read()
    except URLError as exc:
        raise RuntimeError(f"RSS feed could not be loaded: {exc}") from exc

    parsed = feedparser.parse(payload)
    if getattr(parsed, "bozo", False) and not parsed.entries:
        raise RuntimeError(f"RSS feed could not be parsed: {parsed.bozo_exception}")

    articles = []
    for entry in parsed.entries:
        articles.append(
            {
                "title": str(entry.get("title") or ""),
                "link": str(entry.get("link") or ""),
                "published": str(entry.get("published") or entry.get("updated") or ""),
                "summary": str(entry.get("summary") or entry.get("description") or ""),
            }
        )
    return articles


def extract_html_text(payload: str | bytes) -> str:
    """Extract readable text while preserving the source content."""
    from lxml import html

    if isinstance(payload, bytes):
        if not payload.strip():
            return ""
    elif not str(payload).strip():
        return ""

    document = html.fromstring(payload)
    for element in document.xpath(
        "//script | //style | //noscript | //svg | //nav | //header | //footer | //aside | //form"
    ):
        element.drop_tree()

    candidates = document.xpath("//article | //main")
    if candidates:
        content = max(candidates, key=lambda element: len(element.text_content()))
    else:
        content = document
    text = re.sub(r"\s+", " ", content.text_content()).strip()
    return text


def fetch_article_text(url: str, timeout: int = 20) -> str:
    """Fetch readable text without semantic content changes."""
    import certifi

    request = Request(url, headers={"User-Agent": "RAG_System/1.0"})
    context = ssl.create_default_context(cafile=certifi.where())
    with urlopen(request, timeout=timeout, context=context) as response:
        payload = response.read()
    return extract_html_text(payload)


def ingest_rss(
    feed_urls: Iterable[str],
    *,
    article_limit_per_feed: int = 5,
    timeout: int = 20,
    fetch_full_text: bool = True,
    database_path: str | Path = DATABASE_PATH,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    if article_limit_per_feed < 0:
        raise ValueError("article_limit_per_feed must not be negative.")
    if timeout < 1:
        raise ValueError("timeout must be at least 1 second.")

    urls = []
    for value in feed_urls:
        url = validate_feed_url(value)
        if url not in urls:
            urls.append(url)
    if not urls:
        raise ValueError("At least one RSS URL is required.")

    logger = _configure_logger()
    database_path = init_database(database_path)
    with closing(sqlite3.connect(database_path)) as connection:
        known_urls = {
            row[0] for row in connection.execute("SELECT Article_URL FROM Article")
        }

    fetched_count = 0
    inserted_count = 0
    skipped_count = 0
    errors: list[dict[str, str]] = []
    now = datetime.now(timezone.utc).isoformat()

    for feed_index, feed_url in enumerate(urls, start=1):
        if progress_callback:
            progress_callback(f"Loading RSS feed {feed_index}/{len(urls)}: {feed_url}")
        try:
            entries = _fetch_feed(feed_url, timeout)[:article_limit_per_feed]
        except Exception as exc:
            logger.exception("Feed could not be loaded: %s", feed_url)
            errors.append({"feed_url": feed_url, "article_url": "", "error": str(exc)})
            continue

        fetched_count += len(entries)
        for entry_index, entry in enumerate(entries, start=1):
            raw_url = entry.get("link", "")
            try:
                article_url = normalize_article_url(raw_url)
            except ValueError as exc:
                errors.append({"feed_url": feed_url, "article_url": raw_url, "error": str(exc)})
                continue
            if article_url in known_urls:
                skipped_count += 1
                continue

            article_text = ""
            if fetch_full_text:
                if progress_callback:
                    progress_callback(
                        f"Reading article text {entry_index}/{len(entries)}: {article_url}"
                    )
                try:
                    article_text = fetch_article_text(article_url, timeout=timeout)
                except Exception as exc:
                    logger.exception("Article text could not be extracted: %s", article_url)
                    errors.append({"feed_url": feed_url, "article_url": article_url, "error": str(exc)})

            article_description = extract_html_text(entry.get("summary", ""))

            with closing(sqlite3.connect(database_path)) as connection:
                connection.execute(
                    """
                    INSERT OR IGNORE INTO Article(
                        Article_URL, Feed_URL, Article_Title, Article_Description,
                        published_date, Article_Text, embedded, ingested_at
                    ) VALUES (?, ?, ?, ?, ?, ?, 0, ?)
                    """,
                    (
                        article_url,
                        feed_url,
                        entry.get("title", ""),
                        article_description,
                        entry.get("published", ""),
                        article_text,
                        now,
                    ),
                )
                connection.commit()
            known_urls.add(article_url)
            inserted_count += 1

    return {
        "feed_count": len(urls),
        "fetched_count": fetched_count,
        "inserted_count": inserted_count,
        "skipped_count": skipped_count,
        "errors": errors,
        "database_path": str(database_path),
    }


def load_embedding_model(model_name: str, local_files_only: bool = False) -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name, local_files_only=local_files_only)


def build_passage_text(title: str | None, description: str | None) -> str:
    """Build the document input expected by Nemotron retrieval embeddings."""
    content = f"{title or ''} {description or ''}".strip()
    return f"{DOCUMENT_PROMPT_PREFIX}{content}"


def embed_query_text(embedding_model: Any, query: str) -> list[float]:
    """Encode a retrieval query with the model's configured ``query`` prompt."""
    cleaned_query = str(query).strip()
    if not cleaned_query:
        raise ValueError("The retrieval input must not be empty.")
    vector = embedding_model.encode(
        cleaned_query,
        prompt_name="query",
        normalize_embeddings=True,
    )
    return vector.tolist() if hasattr(vector, "tolist") else list(vector)


def _lancedb_table_names(database: Any) -> list[str]:
    page = database.list_tables()
    return list(getattr(page, "tables", page))


def _lancedb_table_columns(table: Any) -> set[str]:
    """Return the persisted LanceDB column names without loading table rows."""
    schema = getattr(table, "schema", None)
    if schema is not None and hasattr(schema, "names"):
        return {str(name) for name in schema.names}
    return {str(name) for name in table.to_arrow().schema.names}


def _has_full_text_index(table: Any, column_name: str = DEFAULT_FTS_COLUMN) -> bool:
    for index in table.list_indices():
        index_columns = {str(column) for column in getattr(index, "columns", [])}
        if str(getattr(index, "index_type", "")).upper() == "FTS" and column_name in index_columns:
            return True
    return False


def ensure_full_text_index(
    table: Any,
    column_name: str = DEFAULT_FTS_COLUMN,
    language: str = DEFAULT_FTS_LANGUAGE,
) -> None:
    """Create or refresh the German BM25 index for the article text."""
    from lancedb.index import FTS

    table.create_index(
        column_name,
        config=FTS(
            language=str(language or DEFAULT_FTS_LANGUAGE),
            base_tokenizer="simple",
            lower_case=True,
            stem=True,
            remove_stop_words=True,
            ascii_folding=True,
        ),
        replace=True,
        name=DEFAULT_FTS_INDEX_NAME,
    )


def embed_unembedded_articles(
    *,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    local_files_only: bool = False,
    database_path: str | Path = DATABASE_PATH,
    vector_db_path: str | Path = VECTOR_DB_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
    embedding_model: Any | None = None,
    progress_callback: ProgressCallback | None = None,
    fts_language: str = DEFAULT_FTS_LANGUAGE,
    batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
) -> dict[str, Any]:
    model_name = str(model_name).strip()
    if not model_name:
        raise ValueError("An embedding model is required.")
    if batch_size < 1:
        raise ValueError("Embedding batch size must be at least 1.")
    database_path = init_database(database_path)
    import lancedb

    vector_db_path = Path(vector_db_path)
    vector_db_path.mkdir(parents=True, exist_ok=True)
    database = lancedb.connect(str(vector_db_path))
    table_names = _lancedb_table_names(database)
    existing_urls: set[str] = set()
    table = None
    rebuilt = False
    if table_name in table_names:
        table = database.open_table(table_name)
        required_columns = {
            "article_url",
            "article_title",
            "article_description",
            "article_text",
            "vector",
        }
        if not required_columns.issubset(_lancedb_table_columns(table)):
            database.drop_table(table_name)
            table = None
            rebuilt = True
            with closing(sqlite3.connect(database_path)) as connection:
                connection.execute("UPDATE Article SET embedded = 0")
                connection.commit()

    with closing(sqlite3.connect(database_path)) as connection:
        rows = connection.execute(
            """
            SELECT Article_URL, Article_Title, Article_Description, Article_Text
            FROM Article WHERE embedded = 0 ORDER BY rowid
            """
        ).fetchall()

    if table is not None:
        existing_urls = {
            str(row.get("article_url"))
            for row in table.to_arrow().to_pylist()
            if row.get("article_url")
        }

    if not rows:
        if table is not None and not _has_full_text_index(table):
            ensure_full_text_index(table, language=fts_language)
        return {
            "selected_count": 0,
            "embedded_count": 0,
            "stored_count": 0,
            "errors": [],
            "fts_index_created": table is not None,
            "rebuilt": rebuilt,
        }

    if embedding_model is None:
        embedding_model = load_embedding_model(model_name, local_files_only=local_files_only)

    vectors = []
    errors: list[dict[str, str]] = []
    successful_urls: list[str] = []

    def store_encoded_row(row: Any, vector: Any) -> None:
        article_url, title, description, article_text = row
        vector = vector.tolist() if hasattr(vector, "tolist") else list(vector)
        if article_url not in existing_urls:
            vectors.append(
                {
                    "article_url": article_url,
                    "article_title": str(title or ""),
                    "article_description": str(description or ""),
                    "article_text": str(article_text or ""),
                    "vector": vector,
                }
            )
        successful_urls.append(article_url)

    for batch_start in range(0, len(rows), batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        if progress_callback:
            progress_callback(
                f"Creating embeddings {batch_start + 1}-{batch_start + len(batch_rows)}/{len(rows)}"
            )

        batch_texts = [build_passage_text(row[1], row[2]) for row in batch_rows]
        try:
            encoded = embedding_model.encode(
                batch_texts,
                normalize_embeddings=True,
                batch_size=batch_size,
                show_progress_bar=False,
            )
            encoded_vectors = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
            if len(encoded_vectors) != len(batch_rows):
                raise ValueError("Embedding model returned an unexpected batch size.")
            for row, vector in zip(batch_rows, encoded_vectors):
                store_encoded_row(row, vector)
        except Exception:
            for row in batch_rows:
                article_url, title, description, _article_text = row
                try:
                    encoded = embedding_model.encode(
                        build_passage_text(title, description),
                        normalize_embeddings=True,
                    )
                    store_encoded_row(row, encoded)
                except Exception as exc:
                    errors.append({"article_url": article_url, "error": str(exc)})

    stored_count = 0
    if vectors:
        if table is None:
            database.create_table(table_name, data=vectors)
            table = database.open_table(table_name)
        else:
            table.add(vectors)
        stored_count = len(vectors)

    if table is not None and (vectors or not _has_full_text_index(table)):
        ensure_full_text_index(table, language=fts_language)

    if successful_urls:
        with closing(sqlite3.connect(database_path)) as connection:
            connection.executemany(
                "UPDATE Article SET embedded = 1 WHERE Article_URL = ?",
                [(url,) for url in successful_urls],
            )
            connection.commit()

    return {
        "selected_count": len(rows),
        "embedded_count": len(successful_urls),
        "stored_count": stored_count,
        "errors": errors,
        "vector_db_path": str(vector_db_path),
        "table_name": table_name,
        "embedding_model": model_name,
        "fts_index_created": table is not None,
        "rebuilt": rebuilt,
    }


def get_sql_stats(database_path: str | Path = DATABASE_PATH) -> dict[str, int]:
    database_path = init_database(database_path)
    with closing(sqlite3.connect(database_path)) as connection:
        total, embedded = connection.execute(
            "SELECT COUNT(*), COALESCE(SUM(embedded), 0) FROM Article"
        ).fetchone()
    return {"articles": int(total), "embedded": int(embedded), "pending": int(total - embedded)}


def get_vector_stats(
    vector_db_path: str | Path = VECTOR_DB_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
) -> dict[str, Any]:
    path = Path(vector_db_path)
    if not path.exists():
        return {
            "table_exists": False,
            "fts_index_exists": False,
            "rows": 0,
            "dimensions": 0,
            "path": str(path),
        }
    import lancedb

    database = lancedb.connect(str(path))
    if table_name not in _lancedb_table_names(database):
        return {
            "table_exists": False,
            "fts_index_exists": False,
            "rows": 0,
            "dimensions": 0,
            "path": str(path),
        }
    table = database.open_table(table_name)
    first_rows = table.to_arrow().slice(0, 1).to_pylist()
    vector = first_rows[0].get("vector", []) if first_rows else []
    return {
        "table_exists": True,
        "fts_index_exists": _has_full_text_index(table),
        "rows": int(table.count_rows()),
        "dimensions": len(vector),
        "path": str(path),
    }


def search_similar_articles(
    query: str,
    *,
    limit: int = 10,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    local_files_only: bool = False,
    database_path: str | Path = DATABASE_PATH,
    vector_db_path: str | Path = VECTOR_DB_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
    embedding_model: Any | None = None,
    vector_candidate_limit: int = DEFAULT_VECTOR_CANDIDATE_LIMIT,
    bm25_candidate_limit: int = DEFAULT_BM25_CANDIDATE_LIMIT,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[dict[str, Any]]:
    """Return hybrid vector/BM25 results fused with weighted RRF."""
    if limit < 1:
        raise ValueError("limit must be at least 1.")
    if vector_candidate_limit < 1 or bm25_candidate_limit < 1:
        raise ValueError("Candidate limits must be at least 1.")
    if vector_weight < 0 or bm25_weight < 0 or vector_weight + bm25_weight <= 0:
        raise ValueError("RRF weights must be non-negative and must not both be zero.")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0.")

    vector_path = Path(vector_db_path)
    if not vector_path.exists():
        return []

    import lancedb

    database = lancedb.connect(str(vector_path))
    if table_name not in _lancedb_table_names(database):
        return []
    table = database.open_table(table_name)
    if DEFAULT_FTS_COLUMN not in _lancedb_table_columns(table):
        raise RuntimeError(
            "The LanceDB table does not contain Article_Text for BM25 search yet. "
            "Rebuild the vector database from the Embeddings page."
        )
    if not _has_full_text_index(table):
        raise RuntimeError(
            "The full-text index is missing. Rebuild the vector database from the Embeddings page."
        )

    if embedding_model is None:
        embedding_model = load_embedding_model(model_name, local_files_only=local_files_only)

    query_vector = embed_query_text(embedding_model, query)
    vector_hits = (
        table.search(query_vector)
        .distance_type("cosine")
        .limit(vector_candidate_limit)
        .to_list()
    )
    bm25_hits = (
        table.search(str(query).strip(), query_type="fts", fts_columns=DEFAULT_FTS_COLUMN)
        .limit(bm25_candidate_limit)
        .to_list()
    )

    vector_by_url: dict[str, dict[str, Any]] = {}
    for rank, hit in enumerate(vector_hits, start=1):
        article_url = str(hit.get("article_url") or "").strip()
        if article_url:
            vector_by_url[article_url] = {
                "rank": rank,
                "distance": float(hit.get("_distance", 0.0)),
            }

    bm25_by_url: dict[str, dict[str, Any]] = {}
    for rank, hit in enumerate(bm25_hits, start=1):
        article_url = str(hit.get("article_url") or "").strip()
        if article_url:
            bm25_by_url[article_url] = {
                "rank": rank,
                "score": float(hit.get("_score", 0.0)),
            }

    article_urls = list(dict.fromkeys([*vector_by_url, *bm25_by_url]))
    if not article_urls:
        return []

    fused = []
    for article_url in article_urls:
        vector_hit = vector_by_url.get(article_url)
        bm25_hit = bm25_by_url.get(article_url)
        rrf_score = 0.0
        if vector_hit is not None:
            rrf_score += vector_weight / (rrf_k + vector_hit["rank"])
        if bm25_hit is not None:
            rrf_score += bm25_weight / (rrf_k + bm25_hit["rank"])
        fused.append(
            {
                "article_url": article_url,
                "rrf_score": rrf_score,
                "vector_rank": vector_hit["rank"] if vector_hit else None,
                "vector_distance": vector_hit["distance"] if vector_hit else None,
                "bm25_rank": bm25_hit["rank"] if bm25_hit else None,
                "bm25_score": bm25_hit["score"] if bm25_hit else None,
            }
        )
    fused.sort(key=lambda row: (-row["rrf_score"], row["article_url"]))
    fused = fused[:limit]

    result_urls = [row["article_url"] for row in fused]
    placeholders = ", ".join("?" for _ in result_urls)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            f"""
            SELECT Article_URL, Feed_URL, Article_Title, Article_Description,
                   published_date, Article_Text, embedded
            FROM Article WHERE Article_URL IN ({placeholders})
            """,
            result_urls,
        ).fetchall()
    articles_by_url = {str(row["Article_URL"]): dict(row) for row in rows}

    results = []
    for rank, fusion in enumerate(fused, start=1):
        article_url = fusion["article_url"]
        article = articles_by_url.get(article_url, {"Article_URL": article_url})
        article.update(
            {
                "rank": rank,
                "rrf_score": fusion["rrf_score"],
                "vector_rank": fusion["vector_rank"],
                "vector_distance": fusion["vector_distance"],
                "bm25_rank": fusion["bm25_rank"],
                "bm25_score": fusion["bm25_score"],
            }
        )
        results.append(article)
    return results


def retrieve_best_article(
    query: str,
    **search_kwargs: Any,
) -> dict[str, Any] | None:
    """Return the single best retrieval result using the shared search path."""
    matches = search_similar_articles(query, limit=1, **search_kwargs)
    return matches[0] if matches else None


def _automatic_cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "<na>", "nat"}:
        return ""
    return text


def build_automatic_query(row: Mapping[str, Any], mode: str) -> str | None:
    """Build a mode-specific query or return ``None`` for an incomplete row."""
    normalized_mode = str(mode).strip().upper()
    if normalized_mode not in AUTOMATIC_RETRIEVAL_MODES:
        raise ValueError(f"Unknown automatic retrieval mode: {mode}")

    if not _automatic_cell_text(row.get("Comment_ID")):
        return None

    fields_by_mode = {
        "Q1": ("Comment_Text",),
        "Q2": ("Video_Title", "Video_Description"),
        "Q3": ("Comment_Text", "Video_Title", "Video_Description"),
    }
    values = [_automatic_cell_text(row.get(field)) for field in fields_by_mode[normalized_mode]]
    if any(not value for value in values):
        return None
    return "\n".join(values)


def run_automatic_retrieval(
    rows: Iterable[Mapping[str, Any]],
    *,
    mode: str,
    vector_weight: float = DEFAULT_VECTOR_WEIGHT,
    bm25_weight: float = DEFAULT_BM25_WEIGHT,
    rrf_k: int = DEFAULT_RRF_K,
    vector_candidate_limit: int = DEFAULT_VECTOR_CANDIDATE_LIMIT,
    bm25_candidate_limit: int = DEFAULT_BM25_CANDIDATE_LIMIT,
    model_name: str = DEFAULT_EMBEDDING_MODEL,
    local_files_only: bool = False,
    database_path: str | Path = DATABASE_PATH,
    vector_db_path: str | Path = VECTOR_DB_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
    embedding_model: Any | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run one best-result hybrid retrieval for every valid input row."""
    normalized_mode = str(mode).strip().upper()
    if normalized_mode not in AUTOMATIC_RETRIEVAL_MODES:
        raise ValueError(f"Unknown automatic retrieval mode: {mode}")
    if vector_weight < 0 or bm25_weight < 0 or vector_weight + bm25_weight <= 0:
        raise ValueError("RRF weights must be non-negative and must not both be zero.")
    if rrf_k <= 0:
        raise ValueError("rrf_k must be greater than 0.")

    input_rows = list(rows)
    prepared_rows: list[tuple[int, Mapping[str, Any], str]] = []
    for source_row_number, row in enumerate(input_rows, start=2):
        query = build_automatic_query(row, normalized_mode)
        if query is not None:
            prepared_rows.append((source_row_number, row, query))

    if not prepared_rows:
        return {
            "mode": normalized_mode,
            "results": [],
            "processed_count": 0,
            "skipped_count": len(input_rows),
        }

    if embedding_model is None:
        embedding_model = load_embedding_model(model_name, local_files_only=local_files_only)

    results: list[dict[str, Any]] = []
    for result_number, (source_row_number, row, query) in enumerate(prepared_rows, start=1):
        if progress_callback:
            progress_callback(
                f"Automatic Retrieval {result_number}/{len(prepared_rows)} "
                f"(Excel row {source_row_number})"
            )
        try:
            match = retrieve_best_article(
                query,
                model_name=model_name,
                local_files_only=local_files_only,
                database_path=database_path,
                vector_db_path=vector_db_path,
                table_name=table_name,
                embedding_model=embedding_model,
                vector_candidate_limit=vector_candidate_limit,
                bm25_candidate_limit=bm25_candidate_limit,
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                rrf_k=rrf_k,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Automatic retrieval failed at Excel row {source_row_number}: {exc}"
            ) from exc

        match = match or {}
        result = {
            "Mode": normalized_mode,
            "VectorWeight": float(vector_weight),
            "BM25Weight": float(bm25_weight),
            "Comment_ID": _automatic_cell_text(row.get("Comment_ID")),
            "Comment_Text": _automatic_cell_text(row.get("Comment_Text")),
            "Video_Title": _automatic_cell_text(row.get("Video_Title")),
            "Video_Description": _automatic_cell_text(row.get("Video_Description")),
            "RRFScore": match.get("rrf_score"),
            "ArticleTitle": _automatic_cell_text(match.get("Article_Title")),
            "ArticleDescription": _automatic_cell_text(match.get("Article_Description")),
            "ArticleText": _automatic_cell_text(match.get("Article_Text")),
        }
        results.append(result)

    return {
        "mode": normalized_mode,
        "results": results,
        "processed_count": len(prepared_rows),
        "skipped_count": len(input_rows) - len(prepared_rows),
    }


def clear_vector_db(
    *,
    vector_db_path: str | Path = VECTOR_DB_PATH,
    table_name: str = DEFAULT_TABLE_NAME,
    database_path: str | Path = DATABASE_PATH,
) -> dict[str, Any]:
    """Drop the configured LanceDB table and reset SQLite embedding flags."""
    vector_path = Path(vector_db_path)
    dropped = False
    previous_rows = 0

    if vector_path.exists():
        import lancedb

        database = lancedb.connect(str(vector_path))
        if table_name in _lancedb_table_names(database):
            table = database.open_table(table_name)
            previous_rows = int(table.count_rows())
            database.drop_table(table_name)
            dropped = True

    database_path = init_database(database_path)
    connection = sqlite3.connect(database_path)
    try:
        cursor = connection.execute("UPDATE Article SET embedded = 0 WHERE embedded != 0")
        reset_count = cursor.rowcount
        connection.commit()
    finally:
        connection.close()

    return {
        "dropped": dropped,
        "previous_vector_rows": previous_rows,
        "reset_sql_rows": int(reset_count),
        "table_name": table_name,
        "vector_db_path": str(vector_path),
    }


def list_articles(
    *,
    limit: int = 100,
    database_path: str | Path = DATABASE_PATH,
) -> list[dict[str, Any]]:
    database_path = init_database(database_path)
    with closing(sqlite3.connect(database_path)) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(
            """
            SELECT Article_URL, Feed_URL, Article_Title, Article_Description,
                   published_date, Article_Text, embedded, ingested_at
            FROM Article ORDER BY rowid DESC LIMIT ?
            """,
            (limit,),
        ).fetchall()
    return [dict(row) for row in rows]
