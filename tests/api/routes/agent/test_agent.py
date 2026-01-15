import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from api.routes.agent.agent import router
from api.dependencies import get_agent_service, get_session_history
from api.auth import get_current_user

from tests.api.common_api_fixtures import (
    app_base,
    override_auth,
    mock_current_user,
    mock_session,
    mock_session_history,
)


@pytest.fixture
def app(app_base, override_auth):
    app_base.include_router(router)
    app_base.dependency_overrides[get_current_user] = override_auth
    return app_base


@pytest.fixture
def client(app):
    return TestClient(app)


@pytest.fixture
def mock_agent_service():
    service = MagicMock()
    service.start_task = AsyncMock(return_value="agent-session-1")
    service.delete_task = AsyncMock(return_value=True)
    service.get_session.return_value = MagicMock(messages=[])
    service._session_store = MagicMock()
    return service


def test_agent_start_session(client, app, mock_agent_service):
    app.dependency_overrides[get_agent_service] = lambda: mock_agent_service

    resp = client.get("/agent")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "agent-session-1"


def test_agent_execute_task_failure(client, app, mock_agent_service, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr("api.routes.agent.agent.start_agent_job", boom)

    app.dependency_overrides[get_agent_service] = lambda: mock_agent_service

    resp = client.post(
        "/agent/session-1/execute",
        data={"task": "do something"},
    )

    assert resp.status_code == 500


def test_get_agent_job_success(client, app, mock_current_user, monkeypatch):
    monkeypatch.setattr(
        "api.routes.agent.agent.get_job",
        lambda job_id: {
            "user": mock_current_user.user_id,
            "status": "completed",
            "result": {"ok": True},
            "error": None,
        },
    )

    resp = client.get("/agent/jobs/job-1")

    assert resp.status_code == 200
    assert resp.json()["status"] == "completed"


def test_get_agent_job_not_found(client, app, monkeypatch):
    monkeypatch.setattr("api.routes.agent.agent.get_job", lambda job_id: None)

    resp = client.get("/agent/jobs/missing")

    assert resp.status_code == 404


def test_get_agent_job_forbidden(client, app, mock_current_user, monkeypatch):
    monkeypatch.setattr(
        "api.routes.agent.agent.get_job",
        lambda job_id: {
            "user": "other-user",
            "status": "completed",
            "result": None,
            "error": None,
        },
    )

    resp = client.get("/agent/jobs/job-1")

    assert resp.status_code == 403


def test_delete_agent_session_success(client, app, mock_agent_service):
    app.dependency_overrides[get_agent_service] = lambda: mock_agent_service

    resp = client.delete("/agent/session-1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True


def test_delete_agent_session_not_found(client, app, mock_agent_service):
    mock_agent_service.delete_task.return_value = False
    app.dependency_overrides[get_agent_service] = lambda: mock_agent_service

    resp = client.delete("/agent/missing")
    assert resp.status_code == 404


def test_agent_list_sessions(client, app, mock_session, mock_session_history):
    mock_session.metadata = {"type": "agent"}
    mock_session_history.list_sessions.return_value = [mock_session]

    app.dependency_overrides[get_session_history] = lambda: mock_session_history

    resp = client.get("/agent/sessions")

    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_agent_get_history_success(client, app, mock_session_history):
    app.dependency_overrides[get_session_history] = lambda: mock_session_history

    resp = client.get("/agent/session-1/history")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "session-1"


def test_agent_get_history_not_found(client, app, mock_session_history):
    mock_session_history.get_history.return_value = None
    app.dependency_overrides[get_session_history] = lambda: mock_session_history

    resp = client.get("/agent/missing/history")
    assert resp.status_code == 404


def test_agent_truncate_success(client, app, mock_agent_service):
    mock_session = MagicMock()
    mock_session.truncate_messages.return_value = 1
    mock_session.messages = ["m1"]

    store = MagicMock()
    store.exists.return_value = True
    store.get.return_value = mock_session

    mock_agent_service._session_store = store
    mock_agent_service.get_session.return_value = mock_session

    app.dependency_overrides[get_agent_service] = lambda: mock_agent_service

    resp = client.put(
        "/agent/session-1/truncate",
        json={"from_index": 1},
    )

    assert resp.status_code == 200
    assert resp.json()["truncated_count"] == 1


def test_agent_truncate_invalid_index(client, app, mock_agent_service):
    mock_session = MagicMock()
    mock_session.truncate_messages.side_effect = ValueError("bad index")

    store = MagicMock()
    store.exists.return_value = True
    store.get.return_value = mock_session

    mock_agent_service._session_store = store

    app.dependency_overrides[get_agent_service] = lambda: mock_agent_service

    resp = client.put(
        "/agent/session-1/truncate",
        json={"from_index": 0},
    )

    assert resp.status_code == 400


def test_agent_truncate_session_not_found(client, app, mock_agent_service):
    store = MagicMock()
    store.exists.return_value = False

    mock_agent_service._session_store = store
    mock_agent_service.get_session.return_value = None

    app.dependency_overrides[get_agent_service] = lambda: mock_agent_service

    resp = client.put(
        "/agent/missing/truncate",
        json={"from_index": 0},
    )

    assert resp.status_code == 404


def test_agent_truncate_session_not_found(client, app, mock_agent_service):
    store = MagicMock()
    store.exists.return_value = False

    mock_agent_service._session_store = store
    mock_agent_service.get_session.return_value = None

    app.dependency_overrides[get_agent_service] = lambda: mock_agent_service

    resp = client.put(
        "/agent/missing/truncate",
        json={"from_index": 0},
    )

    assert resp.status_code == 404

def test_agent_execute_task_success(
    client,
    app,
    mock_agent_service,
    monkeypatch,
    mock_current_user,
):
    # override dependencies
    app.dependency_overrides[get_current_user] = lambda: mock_current_user
    app.dependency_overrides[get_agent_service] = lambda: mock_agent_service

    # mock the async job scheduler
    async def mock_start_agent_job(*, agent_service, session_id, task, user, files):
        assert agent_service is mock_agent_service
        assert session_id == "session-1"
        assert task == "do something"
        assert user == mock_current_user.user_id
        assert files is not None
        return "job-123"

    monkeypatch.setattr(
        "api.routes.agent.agent.start_agent_job",
        mock_start_agent_job,
    )

    resp = client.post(
        "/agent/session-1/execute",
        data={"task": "do something"},
        files=[("files", ("test.txt", b"hello", "text/plain"))],
    )

    assert resp.status_code == 200

    body = resp.json()
    assert body["job_id"] == "job-123"
    assert body["status"] == "scheduled"


def test_get_agent_job_unexpected_exception(client, app, mock_current_user, monkeypatch):
    # override auth dependency
    app.dependency_overrides[get_current_user] = lambda: mock_current_user

    # force get_job to raise a non-HTTPException
    def boom(job_id):
        raise RuntimeError("error")

    monkeypatch.setattr(
        "api.routes.agent.agent.get_job",
        boom,
    )

    resp = client.get("/agent/jobs/job-123")

    assert resp.status_code == 500
    assert resp.json()["detail"] == "Failed to fetch job status: error"


