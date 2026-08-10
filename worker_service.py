from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from queue import Empty, Queue
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from job_store import JobConflictError, JobStore
from llm_service import (
    AUTOMATIC_LLM_INPUT_COLUMNS,
    AUTOMATIC_LLM_MODE_FIELDS,
    AUTOMATIC_LLM_OUTPUT_COLUMNS,
    build_manual_hate_speech_prompt,
    build_automatic_llm_values,
    evaluate_manual_hate_speech,
)
from rss_pipeline import (
    DATABASE_PATH,
    DEFAULT_BM25_CANDIDATE_LIMIT,
    DEFAULT_BM25_WEIGHT,
    DEFAULT_EMBEDDING_BATCH_SIZE,
    DEFAULT_RRF_K,
    DEFAULT_TABLE_NAME,
    DEFAULT_VECTOR_WEIGHT,
    DEFAULT_VECTOR_CANDIDATE_LIMIT,
    clear_vector_db,
    embed_unembedded_articles,
    get_sql_stats,
    get_vector_stats,
    ingest_rss,
    list_articles,
    load_embedding_model,
    load_pipeline_config,
    retrieve_best_article,
    run_automatic_retrieval,
    save_feed_urls,
    load_feed_urls,
    VECTOR_DB_PATH,
    validate_feed_url,
    build_automatic_query,
)


LOGGER = logging.getLogger("rag_system.worker")


class IngestionRequest(BaseModel):
    feed_urls: list[str] | None = None
    article_limit_per_feed: int = Field(default=5, ge=0)
    timeout: int = Field(default=20, ge=1)
    fetch_full_text: bool = True


class EmbeddingRequest(BaseModel):
    local_files_only: bool = False


class FeedConfigRequest(BaseModel):
    feed_urls: list[str]


class AutomaticRetrievalRequest(BaseModel):
    rows: list[dict[str, Any]]
    mode: str = "Q1"
    vector_weight: float = Field(default=DEFAULT_VECTOR_WEIGHT, ge=0)
    bm25_weight: float = Field(default=DEFAULT_BM25_WEIGHT, ge=0)


class ManualLLMEvaluationRequest(BaseModel):
    comment: str = Field(min_length=1)
    api_key: str | None = None
    model: str | None = None
    comment_timestamp: str | None = None
    video_title: str | None = None
    video_description: str | None = None
    video_upload_time: str | None = None
    article_title: str | None = None
    article_description: str | None = None
    article_text: str | None = None


class AutomaticLLMEvaluationRequest(BaseModel):
    rows: list[dict[str, Any]]
    input_columns: list[str] = Field(default_factory=list)
    retrieval_mode: str = "Q1"
    llm_query_mode: str = "Baseline1"
    vector_weight: float = Field(default=DEFAULT_VECTOR_WEIGHT, ge=0)
    bm25_weight: float = Field(default=DEFAULT_BM25_WEIGHT, ge=0)
    api_key: str | None = None
    model: str | None = None


class WorkerRuntime:
    def __init__(self) -> None:
        self.store = JobStore()
        self.queue: Queue[tuple[str, Callable[[dict[str, Any], Callable[[str], None]], dict[str, Any]]]] = Queue()
        self.embedding_model: Any | None = None
        self.embedding_model_name: str | None = None
        self.model_lock = threading.Lock()
        self._job_runtime_payloads: dict[str, dict[str, Any]] = {}
        self._job_runtime_payloads_lock = threading.Lock()
        self._job_secrets: dict[str, dict[str, Any]] = {}
        self._job_secrets_lock = threading.Lock()
        self.ready = False
        self._stop_event = threading.Event()
        self._job_thread = threading.Thread(target=self._job_loop, name="worker-jobs", daemon=True)
        self._scheduler_thread = threading.Thread(
            target=self._scheduler_loop,
            name="automatic-ingestion-scheduler",
            daemon=True,
        )

    def start(self) -> None:
        pipeline_config = load_pipeline_config()
        self.store.set_scheduler_interval(
            int(pipeline_config.get("automatic_ingestion_interval_seconds", 180))
        )
        self.ready = True
        self._job_thread.start()
        self._scheduler_thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def get_embedding_model(self, model_name: str, *, local_files_only: bool = False) -> Any:
        model_name = str(model_name).strip()
        if not model_name:
            raise RuntimeError("Embedding model is not configured.")
        with self.model_lock:
            if self.embedding_model is None or self.embedding_model_name != model_name:
                self.embedding_model = load_embedding_model(
                    model_name,
                    local_files_only=local_files_only,
                )
                self.embedding_model_name = model_name
            return self.embedding_model

    def submit_job(
        self,
        job_type: str,
        payload: dict[str, Any],
        handler: Callable[[dict[str, Any], Callable[[str], None]], dict[str, Any]],
        *,
        runtime_payload: dict[str, Any] | None = None,
        secret_payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            job = self.store.create_job(job_type, payload)
        except JobConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        job_id = str(job["job_id"])
        if runtime_payload:
            with self._job_runtime_payloads_lock:
                self._job_runtime_payloads[job_id] = dict(runtime_payload)
        if secret_payload:
            with self._job_secrets_lock:
                self._job_secrets[job_id] = dict(secret_payload)
        self.queue.put((job_id, handler))
        return job

    def _job_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                job_id, handler = self.queue.get(timeout=0.5)
            except Empty:
                continue
            job = self.store.get_job(job_id)
            job_type = str((job or {}).get("job_type") or "")
            payload = dict((job or {}).get("payload", {}))
            with self._job_runtime_payloads_lock:
                payload.update(self._job_runtime_payloads.get(job_id, {}))
            with self._job_secrets_lock:
                payload.update(self._job_secrets.get(job_id, {}))
            self.store.update_job(job_id, status="running", message="Job started", progress=0)

            last_progress_write = 0.0

            def progress(message: str, value: int | None = None) -> None:
                nonlocal last_progress_write
                now = time.monotonic()
                if value != 100 and now - last_progress_write < 0.75:
                    return
                self.store.update_job(job_id, message=message, progress=value)
                last_progress_write = now

            try:
                result = handler(payload, progress)
            except Exception as exc:
                LOGGER.exception("Worker job failed: %s", job_id)
                self.store.update_job(job_id, status="failed", error=str(exc), message="Job failed")
            else:
                result_errors = result.get("errors") if isinstance(result, Mapping) else None
                if job_type == "embeddings" and isinstance(result_errors, list) and result_errors:
                    details = []
                    for entry in result_errors[:3]:
                        if isinstance(entry, Mapping):
                            article_url = str(entry.get("article_url") or "unknown article")
                            error_text = str(entry.get("error") or "unknown error")
                            details.append(f"{article_url}: {error_text}")
                        else:
                            details.append(str(entry))
                    if len(result_errors) > len(details):
                        details.append(f"... plus {len(result_errors) - len(details)} more")
                    self.store.update_job(
                        job_id,
                        status="failed",
                        progress=100,
                        message="Embedding job failed",
                        error=f"{len(result_errors)} article embedding(s) failed: " + " | ".join(details),
                        result=result,
                    )
                else:
                    self.store.update_job(
                        job_id,
                        status="succeeded",
                        progress=100,
                        message="Job completed",
                        result=result,
                    )
            finally:
                with self._job_runtime_payloads_lock:
                    self._job_runtime_payloads.pop(job_id, None)
                with self._job_secrets_lock:
                    self._job_secrets.pop(job_id, None)
                self.queue.task_done()

    def _scheduler_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                state = self.store.scheduler_state()
                if state.get("enabled"):
                    last_run = state.get("last_run_at")
                    due = last_run is None or (time.time() - _parse_timestamp(last_run)) >= int(
                        state.get("interval_seconds", 180)
                    )
                    if due:
                        self._submit_automatic_ingestion()
            except Exception:
                LOGGER.exception("Automatic ingestion scheduler failed")
            self._stop_event.wait(1)

    def _submit_automatic_ingestion(self, *, raise_conflict: bool = False) -> None:
        feed_urls = load_feed_urls()
        if not feed_urls:
            if raise_conflict:
                raise HTTPException(status_code=400, detail="No RSS feed URLs are configured.")
            return
        payload = {"feed_urls": feed_urls}
        try:
            job = self.submit_job("automatic_ingestion", payload, self.handle_ingestion)
        except HTTPException as exc:
            if exc.status_code != 409:
                raise
            if raise_conflict:
                raise
            return
        self.store.update_scheduler_run(job_id=str(job["job_id"]))

    def handle_ingestion(
        self,
        payload: dict[str, Any],
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        config = load_pipeline_config()
        urls = payload.get("feed_urls") or load_feed_urls()
        return ingest_rss(
            urls,
            article_limit_per_feed=int(
                payload.get("article_limit_per_feed", config.get("article_limit_per_feed", 5))
            ),
            timeout=int(payload.get("timeout", config.get("request_timeout_seconds", 20))),
            fetch_full_text=bool(payload.get("fetch_full_text", config.get("article_text_enabled", True))),
            progress_callback=progress,
        )

    def handle_embeddings(
        self,
        payload: dict[str, Any],
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        config = load_pipeline_config()
        model_name = str(config.get("embedding_model", ""))
        model = self.get_embedding_model(
            model_name,
            local_files_only=bool(payload.get("local_files_only", config.get("local_files_only", False))),
        )
        return embed_unembedded_articles(
            model_name=model_name,
            local_files_only=bool(
                payload.get("local_files_only", config.get("local_files_only", False))
            ),
            vector_db_path=Path(str(config.get("vector_db_path", VECTOR_DB_PATH))),
            table_name=str(config.get("table_name", DEFAULT_TABLE_NAME)),
            fts_language=str(config.get("fts_language", "German")),
            embedding_model=model,
            progress_callback=progress,
            batch_size=int(config.get("embedding_batch_size", DEFAULT_EMBEDDING_BATCH_SIZE)),
        )

    def handle_automatic_llm_evaluation(
        self,
        payload: dict[str, Any],
        progress: Callable[[str], None],
    ) -> dict[str, Any]:
        request = AutomaticLLMEvaluationRequest.model_validate(payload)
        config = load_pipeline_config()
        return _run_automatic_llm_evaluation(
            request,
            pipeline_config=config,
            progress_callback=progress,
        )


def _parse_timestamp(value: str) -> float:
    from datetime import datetime

    return datetime.fromisoformat(value).timestamp()


def _automatic_cell_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "<na>", "nat"}:
        return ""
    return text


def _automatic_required_columns(llm_query_mode: str, retrieval_mode: str) -> set[str]:
    try:
        llm_fields = set(AUTOMATIC_LLM_MODE_FIELDS[llm_query_mode])
    except KeyError as exc:
        raise ValueError(f"Unknown automatic LLM query mode: {llm_query_mode}") from exc

    required = {"Comment_ID"}
    required.update(field for field in llm_fields if not field.startswith("RAG_"))
    if llm_query_mode in {"Rag1", "Rag2"}:
        retrieval_fields = {
            "Q1": {"Comment_Text"},
            "Q2": {"Video_Title", "Video_Description"},
            "Q3": {"Comment_Text", "Video_Title", "Video_Description"},
        }
        try:
            required.update(retrieval_fields[retrieval_mode])
        except KeyError as exc:
            raise ValueError(f"Unknown retrieval query mode: {retrieval_mode}") from exc
    return required


def _validate_automatic_llm_request(request: AutomaticLLMEvaluationRequest) -> None:
    required_columns = _automatic_required_columns(
        str(request.llm_query_mode).strip(),
        str(request.retrieval_mode).strip().upper(),
    )
    available_columns = set(request.input_columns)
    if not available_columns and request.rows:
        available_columns = set().union(*(row.keys() for row in request.rows))
    missing_columns = sorted(required_columns - available_columns)
    if missing_columns:
        raise ValueError("Missing required columns: " + ", ".join(missing_columns))
    if not request.api_key or not str(request.api_key).strip():
        raise ValueError("Gemini API key is required.")


def _automatic_result_row(
    row: Mapping[str, Any],
    mode: str,
    input_columns: list[str],
) -> dict[str, Any]:
    result = {column: _automatic_cell_text(row.get(column)) for column in input_columns}
    result.update(
        {
            "Mode": mode,
            "RAG_Article_Title": "",
            "RAG_Article_Description": "",
            "RAG_Article_Text": "",
            "Hate_Speech_Evaluation": "Fehler",
            "Analysis_Duration": 0.0,
        }
    )
    return result


def _run_automatic_llm_evaluation(
    request: AutomaticLLMEvaluationRequest,
    *,
    embedding_model: Any | None = None,
    pipeline_config: Mapping[str, Any] | None = None,
    progress_callback: Callable[..., None] | None = None,
) -> dict[str, Any]:
    llm_mode = str(request.llm_query_mode).strip()
    retrieval_mode = str(request.retrieval_mode).strip().upper()
    _automatic_required_columns(llm_mode, retrieval_mode)
    if not request.api_key or not str(request.api_key).strip():
        raise ValueError("Gemini API key is required.")
    if llm_mode in {"Rag1", "Rag2"} and request.vector_weight + request.bm25_weight <= 0:
        raise ValueError("RRF weights must be non-negative and must not both be zero.")

    config = dict(pipeline_config or load_pipeline_config())
    model_name = str(config.get("embedding_model", ""))
    if llm_mode in {"Rag1", "Rag2"} and embedding_model is None:
        embedding_model = runtime.get_embedding_model(
            model_name,
            local_files_only=bool(config.get("local_files_only", False)),
        )

    input_rows = list(request.rows)
    input_columns = list(request.input_columns)
    if not input_columns:
        input_columns = list(AUTOMATIC_LLM_INPUT_COLUMNS)
    additional_columns = [
        column
        for column in AUTOMATIC_LLM_OUTPUT_COLUMNS
        if column not in input_columns
    ]
    output_columns = [*input_columns, *additional_columns]
    started_all = time.perf_counter()

    def process_row(row_index: int, row: Mapping[str, Any]) -> tuple[int, dict[str, Any], dict[str, Any]]:
        row_number = row_index + 2
        started_at = time.perf_counter()
        result = _automatic_result_row(row, llm_mode, input_columns)
        status = "OK"
        detail = ""
        try:
            rag_article: dict[str, Any] | None = None
            if llm_mode in {"Rag1", "Rag2"}:
                query = build_automatic_query(row, retrieval_mode)
                if query is None:
                    raise ValueError("Required retrieval fields are missing or empty.")
                with retrieval_lock:
                    rag_article = retrieve_best_article(
                        query,
                        model_name=model_name,
                        local_files_only=True,
                        database_path=str(config.get("database_path", DATABASE_PATH)),
                        vector_db_path=str(config.get("vector_db_path", VECTOR_DB_PATH)),
                        table_name=str(config.get("table_name", DEFAULT_TABLE_NAME)),
                        embedding_model=embedding_model,
                        vector_candidate_limit=int(
                            config.get("vector_candidate_limit", DEFAULT_VECTOR_CANDIDATE_LIMIT)
                        ),
                        bm25_candidate_limit=int(
                            config.get("bm25_candidate_limit", DEFAULT_BM25_CANDIDATE_LIMIT)
                        ),
                        vector_weight=request.vector_weight,
                        bm25_weight=request.bm25_weight,
                        rrf_k=int(config.get("rrf_k", DEFAULT_RRF_K)),
                    )
                if rag_article is None:
                    raise LookupError("No retrieval article found for this comment.")
                result["RAG_Article_Title"] = _automatic_cell_text(rag_article.get("Article_Title"))
                result["RAG_Article_Description"] = _automatic_cell_text(
                    rag_article.get("Article_Description")
                )
                result["RAG_Article_Text"] = _automatic_cell_text(rag_article.get("Article_Text"))

            llm_values = build_automatic_llm_values(
                row,
                mode=llm_mode,
                api_key=str(request.api_key),
                model=request.model,
                rag_article=rag_article,
            )
            llm_result = evaluate_manual_hate_speech(llm_values)
            result["Hate_Speech_Evaluation"] = llm_result["result"]
        except Exception as exc:
            status = "Fehler"
            detail = str(exc)

        duration_seconds = round(time.perf_counter() - started_at, 3)
        result["Analysis_Duration"] = duration_seconds
        log_entry: dict[str, Any] = {
            "row_number": row_number,
            "comment_id": result.get("Comment_ID", ""),
            "step": "Kommentaranalyse",
            "status": status,
            "duration_ms": round(duration_seconds * 1000, 2),
        }
        if detail:
            log_entry["detail"] = detail
        return row_index, result, log_entry

    if not input_rows:
        return {
            "retrieval_query_mode": retrieval_mode,
            "llm_query_mode": llm_mode,
            "results": [],
            "evaluation_log": [],
            "processed_count": 0,
            "error_count": 0,
            "total_duration_ms": 0.0,
            "output_columns": output_columns,
        }

    configured_workers = int(config.get("llm_max_concurrency", 4))
    max_workers = max(1, min(configured_workers, len(input_rows)))
    retrieval_lock = threading.Lock()
    result_slots: list[dict[str, Any] | None] = [None] * len(input_rows)
    log_slots: list[dict[str, Any] | None] = [None] * len(input_rows)
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="llm-evaluation") as executor:
        futures = {
            executor.submit(process_row, row_index, row): row_index
            for row_index, row in enumerate(input_rows)
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            row_index, result, log_entry = future.result()
            result_slots[row_index] = result
            log_slots[row_index] = log_entry
            if progress_callback:
                progress_callback(
                    f"Automatic LLM evaluation {completed_count}/{len(input_rows)}",
                    round(completed_count / len(input_rows) * 100),
                )

    results = [result for result in result_slots if result is not None]
    evaluation_log = [entry for entry in log_slots if entry is not None]
    error_count = sum(entry.get("status") == "Fehler" for entry in evaluation_log)

    return {
        "retrieval_query_mode": retrieval_mode,
        "llm_query_mode": llm_mode,
        "results": results,
        "evaluation_log": evaluation_log,
        "processed_count": len(results),
        "error_count": error_count,
        "total_duration_ms": round((time.perf_counter() - started_all) * 1000, 2),
        "output_columns": output_columns,
    }


runtime = WorkerRuntime()
app = FastAPI(title="RAG_System Worker", version="1.0.0")


@app.on_event("startup")
def startup() -> None:
    runtime.start()


@app.on_event("shutdown")
def shutdown() -> None:
    runtime.stop()


@app.get("/health")
def health() -> dict[str, Any]:
    if not runtime.ready:
        raise HTTPException(status_code=503, detail="The worker is not ready.")
    return {
        "status": "ok",
        "service": "worker",
        "embedding_model_ready": runtime.embedding_model is not None,
    }


@app.get("/config/feeds")
def get_feeds() -> dict[str, Any]:
    return {"feed_urls": load_feed_urls()}


@app.put("/config/feeds")
def put_feeds(request: FeedConfigRequest) -> dict[str, Any]:
    try:
        return {"feed_urls": load_feed_urls(save_feed_urls(request.feed_urls))}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/jobs/manual-ingestion")
def manual_ingestion(request: IngestionRequest) -> dict[str, Any]:
    feed_urls = request.feed_urls if request.feed_urls is not None else load_feed_urls()
    if not feed_urls:
        raise HTTPException(status_code=400, detail="No RSS feed URLs are configured.")
    try:
        for feed_url in feed_urls:
            validate_feed_url(feed_url)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return runtime.submit_job(
        "manual_ingestion",
        request.model_dump(),
        runtime.handle_ingestion,
    )


@app.post("/jobs/embeddings")
def embeddings(request: EmbeddingRequest) -> dict[str, Any]:
    return runtime.submit_job(
        "embeddings",
        request.model_dump(),
        runtime.handle_embeddings,
    )


@app.get("/jobs/{job_id}")
def job_status(job_id: str) -> dict[str, Any]:
    job = runtime.store.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


@app.get("/jobs")
def jobs(limit: int = 200) -> dict[str, Any]:
    return {"jobs": runtime.store.list_jobs(limit=limit)}


@app.delete("/jobs")
def delete_finished_jobs() -> dict[str, Any]:
    return {"deleted_count": runtime.store.delete_finished_jobs()}


@app.delete("/jobs/{job_id}")
def delete_job(job_id: str) -> dict[str, Any]:
    try:
        deleted = runtime.store.delete_job(job_id)
    except JobConflictError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not deleted:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {"deleted": True, "job_id": job_id}


@app.post("/automatic-ingestion/start")
def start_automatic_ingestion() -> dict[str, Any]:
    if not load_feed_urls():
        raise HTTPException(status_code=400, detail="No RSS feed URLs are configured.")
    runtime.store.set_scheduler_enabled(True)
    try:
        runtime._submit_automatic_ingestion(raise_conflict=True)
    except HTTPException:
        runtime.store.set_scheduler_enabled(False)
        raise
    return runtime.store.scheduler_state()


@app.post("/automatic-ingestion/pause")
def pause_automatic_ingestion() -> dict[str, Any]:
    return runtime.store.set_scheduler_enabled(False)


@app.get("/automatic-ingestion/status")
def automatic_ingestion_status() -> dict[str, Any]:
    return runtime.store.scheduler_state()


@app.get("/stats")
def stats() -> dict[str, Any]:
    config = load_pipeline_config()
    return {
        "sql": get_sql_stats(),
        "vector": get_vector_stats(
            vector_db_path=str(config.get("vector_db_path", VECTOR_DB_PATH)),
            table_name=str(config.get("table_name", DEFAULT_TABLE_NAME)),
        ),
    }


@app.get("/articles")
def articles(limit: int = 200) -> dict[str, Any]:
    return {"articles": list_articles(limit=max(1, min(limit, 1000)))}


@app.post("/retrieval/automatic")
def automatic_retrieval(request: AutomaticRetrievalRequest) -> dict[str, Any]:
    config = load_pipeline_config()
    model = runtime.get_embedding_model(
        str(config.get("embedding_model", "")),
        local_files_only=bool(config.get("local_files_only", False)),
    )
    return run_automatic_retrieval(
        request.rows,
        mode=request.mode,
        vector_weight=request.vector_weight,
        bm25_weight=request.bm25_weight,
        rrf_k=int(config.get("rrf_k", DEFAULT_RRF_K)),
        vector_candidate_limit=int(
            config.get("vector_candidate_limit", DEFAULT_VECTOR_CANDIDATE_LIMIT)
        ),
        bm25_candidate_limit=int(
            config.get("bm25_candidate_limit", DEFAULT_BM25_CANDIDATE_LIMIT)
        ),
        model_name=str(config.get("embedding_model", "")),
        local_files_only=True,
        vector_db_path=str(config.get("vector_db_path", VECTOR_DB_PATH)),
        table_name=str(config.get("table_name", DEFAULT_TABLE_NAME)),
        embedding_model=model,
    )


@app.post("/llm/manual-evaluation/prompt")
def manual_llm_evaluation_prompt(request: ManualLLMEvaluationRequest) -> dict[str, str]:
    try:
        return {"prompt": build_manual_hate_speech_prompt(request.model_dump())}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/llm/manual-evaluation")
def manual_llm_evaluation(request: ManualLLMEvaluationRequest) -> dict[str, Any]:
    try:
        return evaluate_manual_hate_speech(request.model_dump())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/llm/automatic-evaluation")
def automatic_llm_evaluation(request: AutomaticLLMEvaluationRequest) -> dict[str, Any]:
    try:
        _validate_automatic_llm_request(request)
        return _run_automatic_llm_evaluation(
            request,
            pipeline_config=load_pipeline_config(),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/jobs/llm/automatic-evaluation")
def submit_automatic_llm_evaluation(request: AutomaticLLMEvaluationRequest) -> dict[str, Any]:
    try:
        _validate_automatic_llm_request(request)
        payload = request.model_dump(exclude={"api_key", "rows"})
        return runtime.submit_job(
            "automatic_llm_evaluation",
            payload,
            runtime.handle_automatic_llm_evaluation,
            runtime_payload={"rows": request.rows},
            secret_payload={"api_key": str(request.api_key or "")},
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/vector-db/clear")
def clear_vectors() -> dict[str, Any]:
    config = load_pipeline_config()
    return clear_vector_db(
        vector_db_path=str(config.get("vector_db_path", VECTOR_DB_PATH)),
        table_name=str(config.get("table_name", DEFAULT_TABLE_NAME)),
    )
