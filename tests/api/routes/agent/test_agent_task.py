import asyncio
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

    job_id = agent_task.create_job("test-user")

    agent_service = MagicMock()
    agent_service.execute_task = AsyncMock(return_value={"ok": True})

    await agent_task._run_job(
        job_id,
        agent_service=agent_service,
        session_id="session-1",
        task="do something",
        user="test-user",
        files=None,
    )

    job = agent_task.JOBS[job_id]

    assert job["status"] == "completed"
    assert job["result"] == {"ok": True}
    assert job["error"] is None

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
        files=None,
    )

    job = agent_task.JOBS[job_id]

    assert job["status"] == "failed"
    assert job["result"] is None
    assert job["error"] == "boom"

def test_start_agent_job_schedules_task(monkeypatch):
    agent_task.JOBS.clear()

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

    job_id = agent_task.start_agent_job(
        agent_service=agent_service,
        session_id="session-1",
        task="do something",
        user="test-user",
        files=None,
    )

    assert job_id == "job-1"
    assert "job-1" in agent_task.JOBS
    assert agent_task.JOBS["job-1"]["status"] == "pending"
