from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rss_pipeline import DATABASE_PATH, init_database


ACTIVE_JOB_STATUSES = ("pending", "running")
INGESTION_JOB_TYPES = ("manual_ingestion", "automatic_ingestion")
TERMINAL_JOB_STATUSES = ("succeeded", "failed", "cancelled")


class JobConflictError(RuntimeError):
    """Raised when a second conflicting job is requested."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class JobStore:
    """Persist worker jobs and automatic-ingestion state in SQLite."""

    def __init__(self, database_path: str | Path = DATABASE_PATH) -> None:
        self.database_path = init_database(database_path)
        self._ensure_schema()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _ensure_schema(self) -> None:
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA busy_timeout = 30000")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS Job (
                    job_id TEXT PRIMARY KEY,
                    job_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_json TEXT,
                    progress INTEGER NOT NULL DEFAULT 0,
                    message TEXT,
                    error TEXT,
                    created_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_job_status ON Job(status);

                CREATE TABLE IF NOT EXISTS SchedulerState (
                    name TEXT PRIMARY KEY,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    interval_seconds INTEGER NOT NULL DEFAULT 180,
                    last_run_at TEXT,
                    last_job_id TEXT,
                    last_error TEXT,
                    updated_at TEXT NOT NULL
                );
                """
            )
            connection.execute(
                """
                INSERT INTO SchedulerState(name, enabled, interval_seconds, updated_at)
                VALUES ('automatic_ingestion', 0, 180, ?)
                ON CONFLICT(name) DO UPDATE SET enabled = 0, updated_at = excluded.updated_at
                """,
                (_utc_now(),),
            )
            connection.execute(
                """
                UPDATE Job
                SET status = 'failed',
                    error = 'Worker restarted before the job completed.',
                    message = 'Job interrupted by worker restart.',
                    finished_at = ?
                WHERE status IN ('pending', 'running')
                """,
                (_utc_now(),),
            )
            connection.commit()

    def active_job(self, job_types: tuple[str, ...] | None = None) -> dict[str, Any] | None:
        query = "SELECT * FROM Job WHERE status IN (?, ?)"
        parameters: list[Any] = list(ACTIVE_JOB_STATUSES)
        if job_types:
            placeholders = ", ".join("?" for _ in job_types)
            query += f" AND job_type IN ({placeholders})"
            parameters.extend(job_types)
        query += " ORDER BY created_at LIMIT 1"
        with closing(self._connect()) as connection:
            row = connection.execute(query, parameters).fetchone()
        return self._row_to_dict(row) if row else None

    def create_job(self, job_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        if self.active_job():
            raise JobConflictError("Another worker job is already running or queued.")
        job_id = uuid.uuid4().hex
        created_at = _utc_now()
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO Job(job_id, job_type, status, payload_json, created_at)
                VALUES (?, ?, 'pending', ?, ?)
                """,
                (job_id, job_type, json.dumps(payload, ensure_ascii=False), created_at),
            )
            connection.commit()
        return self.get_job(job_id) or {}

    def update_job(
        self,
        job_id: str,
        *,
        status: str | None = None,
        progress: int | None = None,
        message: str | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        assignments: list[str] = []
        parameters: list[Any] = []
        if status is not None:
            assignments.append("status = ?")
            parameters.append(status)
            if status == "running":
                assignments.append("started_at = COALESCE(started_at, ?)")
                parameters.append(_utc_now())
            if status in {"succeeded", "failed", "cancelled"}:
                assignments.append("finished_at = ?")
                parameters.append(_utc_now())
        if progress is not None:
            assignments.append("progress = ?")
            parameters.append(max(0, min(100, int(progress))))
        if message is not None:
            assignments.append("message = ?")
            parameters.append(message)
        if result is not None:
            assignments.append("result_json = ?")
            parameters.append(json.dumps(result, ensure_ascii=False))
        if error is not None:
            assignments.append("error = ?")
            parameters.append(error)
        if not assignments:
            return self.get_job(job_id) or {}
        parameters.append(job_id)
        with closing(self._connect()) as connection:
            connection.execute(
                f"UPDATE Job SET {', '.join(assignments)} WHERE job_id = ?",
                parameters,
            )
            connection.commit()
        return self.get_job(job_id) or {}

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute("SELECT * FROM Job WHERE job_id = ?", (job_id,)).fetchone()
        return self._row_to_dict(row) if row else None

    def list_jobs(self, limit: int = 200) -> list[dict[str, Any]]:
        bounded_limit = max(1, min(int(limit), 1000))
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT job_id, job_type, status, progress, message, error,
                       created_at, started_at, finished_at
                FROM Job
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (bounded_limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_job(self, job_id: str) -> bool:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT status FROM Job WHERE job_id = ?",
                (job_id,),
            ).fetchone()
            if row is None:
                return False
            if str(row[0]) in ACTIVE_JOB_STATUSES:
                raise JobConflictError("Active jobs cannot be deleted.")
            connection.execute("DELETE FROM Job WHERE job_id = ?", (job_id,))
            connection.commit()
        return True

    def delete_finished_jobs(self) -> int:
        placeholders = ", ".join("?" for _ in TERMINAL_JOB_STATUSES)
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                f"DELETE FROM Job WHERE status IN ({placeholders})",
                TERMINAL_JOB_STATUSES,
            )
            connection.commit()
        return int(cursor.rowcount)

    def set_scheduler_enabled(self, enabled: bool) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE SchedulerState
                SET enabled = ?, updated_at = ?, last_error = NULL
                WHERE name = 'automatic_ingestion'
                """,
                (int(enabled), _utc_now()),
            )
            connection.commit()
        return self.scheduler_state()

    def set_scheduler_interval(self, interval_seconds: int) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE SchedulerState
                SET interval_seconds = ?, updated_at = ?
                WHERE name = 'automatic_ingestion'
                """,
                (max(1, int(interval_seconds)), _utc_now()),
            )
            connection.commit()
        return self.scheduler_state()

    def update_scheduler_run(
        self,
        *,
        job_id: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                UPDATE SchedulerState
                SET last_run_at = ?, last_job_id = ?, last_error = ?, updated_at = ?
                WHERE name = 'automatic_ingestion'
                """,
                (_utc_now(), job_id, error, _utc_now()),
            )
            connection.commit()
        return self.scheduler_state()

    def scheduler_state(self) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM SchedulerState WHERE name = 'automatic_ingestion'"
            ).fetchone()
        return self._row_to_dict(row) if row else {
            "name": "automatic_ingestion",
            "enabled": 0,
            "interval_seconds": 180,
        }

    @staticmethod
    def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
        result = dict(row)
        for key in ("payload_json", "result_json"):
            if result.get(key):
                result[key.removesuffix("_json")] = json.loads(result[key])
        result["enabled"] = bool(result.get("enabled", 0))
        return result
