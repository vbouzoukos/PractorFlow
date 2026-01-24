import asyncio
import io
from unittest.mock import AsyncMock, MagicMock

import pytest

import api.routes.agent.agent_task as agent_task


def test_create_and_get_job():
    job_id = agent_task.create_job("test-user")

    job = agent_task.get_job(job_id)

    assert job is not None
    assert job["user"] == "test-user"
    assert job["status"] == "pending"
    assert job["result"] is None
    assert job["error"] is None


@pytest.mark.asyncio
async def test_run_job_success():
    agent_task.JOBS.clear()

    fake_file = agent_task.JobFile(
        file=io.BytesIO(b"data"),
        filename="test.txt",
        content_type="text/plain",
    )

    job_id = agent_task.create_job("test-user", files=[fake_file])

    agent_service = MagicMock()
    agent_service.execute_task = AsyncMock(return_value={"ok": True})

    await agent_task._run_job(
        job_id,
        agent_service=agent_service,
        session_id="session-1",
        task="do something",
        user="test-user",
    )

    job = agent_task.JOBS[job_id]

    agent_service.execute_task.assert_awaited_once()
    _, kwargs = agent_service.execute_task.call_args
    assert kwargs["files"] is not None
    assert len(kwargs["files"]) == 1
    assert kwargs["files"][0].filename == "test.txt"

    assert job["status"] == "completed"
    assert job["result"] == {"ok": True}
    assert job["error"] is None
    assert job["files"] is None


@pytest.mark.asyncio
async def test_run_job_failure():
    agent_task.JOBS.clear()

    job_id = agent_task.create_job("test-user")

    agent_service = MagicMock()
    agent_service.execute_task = AsyncMock(side_effect=RuntimeError("boom"))

    await agent_task._run_job(
        job_id,
        agent_service=agent_service,
        session_id="session-1",
        task="do something",
        user="test-user",
    )

    job = agent_task.JOBS[job_id]

    assert job["status"] == "failed"
    assert job["result"] is None
    assert job["error"] == "boom"


@pytest.mark.asyncio
async def test_start_agent_job_schedules_task(monkeypatch):
    agent_task.JOBS.clear()
    class FakeUploadFile:
        def __init__(self, content: bytes, filename: str, content_type: str):
            self._content = content
            self.filename = filename
            self.content_type = content_type

        async def read(self):
            return self._content

    upload_files = [
        FakeUploadFile(b"hello", "file1.txt", "text/plain"),
        FakeUploadFile(b"world", "file2.txt", "text/plain"),
    ]

    def fake_create_job(*args, **kwargs):
        job_id = "job-1"
        agent_task.JOBS[job_id] = {
            "user": "test-user",
            "status": "pending",
            "result": None,
            "error": None,
        }
        return job_id

    def fake_create_task(coro):
        coro.close()
        return MagicMock()

    monkeypatch.setattr(agent_task, "create_job", fake_create_job)
    monkeypatch.setattr(asyncio, "create_task", fake_create_task)

    agent_service = MagicMock()

    job_id = await agent_task.start_agent_job(
        agent_service=agent_service,
        session_id="session-1",
        task="do something",
        user="test-user",
        files=upload_files,
    )

    assert job_id == "job-1"
    assert "job-1" in agent_task.JOBS
    assert agent_task.JOBS["job-1"]["status"] == "pending"


@pytest.mark.asyncio
async def test_read_upload_files():
    class FakeUploadFile:
        def __init__(self, content: bytes, filename: str, content_type: str):
            self._content = content
            self.filename = filename
            self.content_type = content_type

        async def read(self):
            return self._content

    upload_files = [
        FakeUploadFile(b"hello", "file1.txt", "text/plain"),
        FakeUploadFile(b"world", "file2.txt", "text/plain"),
    ]

    job_files = await agent_task._read_upload_files(upload_files)

    assert len(job_files) == 2
    assert job_files[0].filename == "file1.txt"
    assert job_files[0].content_type == "text/plain"
    assert job_files[0].file.read() == b"hello"
    assert job_files[1].filename == "file2.txt"
    assert job_files[1].file.read() == b"world"
