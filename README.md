# RAG_System

An English Streamlit application for RSS ingestion, article storage, RAG retrieval, and Gemini-based LLM evaluation.

## Services

Docker Compose starts two services:

- `app`: Streamlit frontend on port `9005`.
- `worker`: asynchronous FastAPI worker for ingestion, embeddings, retrieval, and Gemini API requests.

The app communicates with the worker through an internal HTTP API. Long-running ingestion, embedding, and automatic LLM evaluation tasks return a job ID and expose progress through `/jobs/{job_id}`.

## Persistent volumes

- `App_data`: SQLite, LanceDB, logs, jobs, scheduler state, and results.
- `App_config`: RSS URLs and pipeline configuration.
- `modell_cache`: Hugging Face and SentenceTransformer model cache.

## Start with Docker

```powershell
docker compose up -d --build
```

The worker and app become healthy as soon as the services are ready. The embedding model is loaded lazily when embeddings or RAG retrieval are first requested.

Existing SQLite databases are migrated on worker startup: legacy `cleanup_status` and `cleanup_error` columns are removed while article data remains intact.

Open [http://localhost:9005](http://localhost:9005).

The worker is configured to reserve one NVIDIA GPU for embedding workloads. On Windows, Docker Desktop must use the WSL2 backend with working NVIDIA GPU support. CPU-only hosts can remove the worker GPU reservation and continue to run the application on the CPU.

Useful commands:

```powershell
docker compose ps
docker compose logs -f worker app
docker compose down
```

## RSS ingestion

Manual ingestion lets users configure feeds and start a job on demand. Existing articles are skipped by normalized article URL. Article descriptions and full text are stored after technical HTML extraction and whitespace normalization only; no semantic rewriting or language model processing is performed.

Automatic ingestion can be started and paused from the frontend. While running, the worker checks configured feeds immediately and then every 180 seconds. Automatic ingestion never creates embeddings automatically; embeddings are started manually.

## LLM evaluation

The Gemini API key is configured once in the left sidebar and is kept only in the current Streamlit session. It is sent only with evaluation requests and is not persisted by the application.

The `Manual LLM Evaluation` page sends one comment and optional video/article context to the selected Gemini model through the official `google-genai` Python SDK. It shows the complete prompt before confirmation, the evaluation log with step durations, the raw Gemini output, the selected model, and the 0–10 hate-speech score.

The `Automatic LLM Evaluation` page reads an XLSX file, supports the eight Baseline, RAG, BestCase, and WorstCase query modes, and exposes configurable retrieval query modes and Vector/BM25 weighting. Each row is processed as an independent, stateless Gemini request with controlled concurrency and job progress. Retrieval is executed only for `Rag1` and `Rag2`, using the best matching article. The output preserves the input columns and adds the selected LLM mode, RAG article fields, the hate-speech score or `Fehler`, and `Analysis_Duration` in seconds. The frontend displays one log entry with duration and status for every input comment.

The `Jobs` page lists worker jobs and allows finished job records and their stored results to be deleted. Automatic-evaluation input rows and Gemini API keys are kept in worker memory while the job runs and are not stored in the job payload.

The default model is `gemini-3.5-flash-lite`. The frontend model selector offers `gemini-3.6-flash`, `gemini-3.5-flash-lite`, and `gemini-3.5-flash`. The default can be changed with `GEMINI_MODEL`.

## Local development

Install dependencies into the project virtual environment:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Start the frontend; it starts the local worker automatically:

```powershell
streamlit run streamlit_app.py
```

The frontend is available at `http://localhost:9005`. To run the worker separately, set `AUTO_START_WORKER=false` and start Uvicorn on port `9000`.
