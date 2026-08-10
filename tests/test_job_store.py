from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from job_store import JobConflictError, JobStore


class JobStoreTest(unittest.TestCase):
    def test_rejects_second_active_job(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir) / "jobs.sqlite3")
            first = store.create_job("manual_ingestion", {"feed_urls": []})
            self.assertEqual(first["status"], "pending")
            with self.assertRaises(JobConflictError):
                store.create_job("automatic_ingestion", {"feed_urls": []})

    def test_scheduler_is_paused_on_store_startup(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "jobs.sqlite3"
            first_store = JobStore(database_path)
            first_store.set_scheduler_enabled(True)
            self.assertTrue(first_store.scheduler_state()["enabled"])

            second_store = JobStore(database_path)
            self.assertFalse(second_store.scheduler_state()["enabled"])

    def test_job_result_is_serialized_and_read_back(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir) / "jobs.sqlite3")
            job = store.create_job("embeddings", {})
            completed = store.update_job(
                job["job_id"],
                status="succeeded",
                progress=100,
                result={"stored_count": 3},
            )
            self.assertEqual(completed["result"], {"stored_count": 3})

    def test_lists_jobs_without_loading_payload_or_result(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir) / "jobs.sqlite3")
            job = store.create_job("manual_ingestion", {"feed_urls": ["https://example.test/feed"]})
            store.update_job(job["job_id"], status="succeeded", result={"large": "result"})
            listed = store.list_jobs()
            self.assertEqual(len(listed), 1)
            self.assertNotIn("payload_json", listed[0])
            self.assertNotIn("result_json", listed[0])

    def test_deletes_finished_jobs_but_rejects_active_jobs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            store = JobStore(Path(temp_dir) / "jobs.sqlite3")
            active = store.create_job("manual_ingestion", {})
            with self.assertRaises(JobConflictError):
                store.delete_job(active["job_id"])
            store.update_job(active["job_id"], status="failed", error="test")
            self.assertTrue(store.delete_job(active["job_id"]))
            self.assertEqual(store.list_jobs(), [])


if __name__ == "__main__":
    unittest.main()
