FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1

WORKDIR /build

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --prefix=/install -r requirements.txt

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/models/sentence-transformers

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

COPY rss_pipeline.py streamlit_app.py worker_service.py job_store.py llm_service.py ./
COPY config ./config

RUN mkdir -p /app/data/sql /app/data/vector /app/data/logs \
    /models/huggingface /models/sentence-transformers

EXPOSE 9000 9005

CMD ["uvicorn", "worker_service:app", "--host", "0.0.0.0", "--port", "9000"]
