import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from api.routes.session import router
from api.dependencies import get_delete_session_service, get_session_history
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
def mock_delete_session_service():
    service = MagicMock()
    service.delete_chat = AsyncMock(return_value=True)
    service.get_session.return_value = MagicMock(messages=[])
    service.delete_session_document = AsyncMock(return_value=True)
    return service


@pytest.fixture
def override_delete_session_service(mock_delete_session_service):
    return lambda: mock_delete_session_service


@pytest.fixture
def override_session_history(mock_session_history):
    return lambda: mock_session_history


def test_delete_session_success(client, app, mock_delete_session_service):
    app.dependency_overrides[get_delete_session_service] = (
        lambda: mock_delete_session_service
    )

    resp = client.delete("/sessions/session-1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True


def test_delete_session_not_found(client, app, mock_delete_session_service):
    mock_delete_session_service.delete_chat.return_value = False
    app.dependency_overrides[get_delete_session_service] = (
        lambda: mock_delete_session_service
    )

    resp = client.delete("/sessions/missing")
    assert resp.status_code == 404


def test_list_sessions(client, app, override_session_history):
    app.dependency_overrides[get_session_history] = override_session_history

    resp = client.get("/sessions")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_get_history_success(client, app, override_session_history):
    app.dependency_overrides[get_session_history] = override_session_history

    resp = client.get("/sessions/session-1/history")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "session-1"


def test_get_history_not_found(client, app, mock_session_history):
    mock_session_history.get_history.return_value = None
    app.dependency_overrides[get_session_history] = lambda: mock_session_history

    resp = client.get("/sessions/missing/history")
    assert resp.status_code == 404


def test_get_history_forbidden(client, app, mock_session_history):
    # session exists but belongs to another user
    mock_session = MagicMock()
    mock_session.session_id = "session-1"
    mock_session.user = "other-user"
    mock_session.messages = []
    mock_session.documents = []
    mock_session.instructions = ""
    mock_session.created_at.isoformat.return_value = "2024-01-01T00:00:00"
    mock_session.updated_at.isoformat.return_value = "2024-01-01T00:00:00"

    mock_session_history.get_history.return_value = mock_session
    app.dependency_overrides[get_session_history] = lambda: mock_session_history

    resp = client.get("/sessions/session-1/history")

    assert resp.status_code == 403
    assert resp.json()["detail"] == "Forbidden"


def test_search_sessions_success(client, app, mock_session_history, mock_session):
    mock_session_history.sessions_by_title.return_value = [mock_session]
    app.dependency_overrides[get_session_history] = lambda: mock_session_history

    resp = client.get("/sessions/search", params={"term": "test"})

    assert resp.status_code == 200
    assert len(resp.json()) == 1
    assert resp.json()[0]["session_id"] == mock_session.session_id


def test_search_sessions_no_results(client, app, mock_session_history):
    mock_session_history.sessions_by_title.return_value = []
    app.dependency_overrides[get_session_history] = lambda: mock_session_history

    resp = client.get("/sessions/search", params={"term": "missing"})

    assert resp.status_code == 200
    assert resp.json() == []


def test_list_documents(client, app, override_session_history):
    app.dependency_overrides[get_session_history] = override_session_history

    resp = client.get("/sessions/session-1/documents")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_list_documents_session_not_found(client, app, mock_session_history):
    mock_session_history.get_history.return_value = None
    app.dependency_overrides[get_session_history] = lambda: mock_session_history

    resp = client.get("/sessions/missing/documents")

    assert resp.status_code == 404
    assert "Session not found" in resp.json()["detail"]


def test_delete_document_success(client, app, mock_delete_session_service):
    app.dependency_overrides[get_delete_session_service] = (
        lambda: mock_delete_session_service
    )

    resp = client.delete("/sessions/session-1/documents/doc-1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True


def test_delete_document_session_not_found(client, app, mock_delete_session_service):
    mock_delete_session_service.delete_session_document.return_value = None
    app.dependency_overrides[get_delete_session_service] = (
        lambda: mock_delete_session_service
    )

    resp = client.delete("/sessions/session-1/documents/doc-1")
    assert resp.status_code == 404


def test_delete_document_doc_not_found(client, app, mock_delete_session_service):
    mock_delete_session_service.delete_session_document.return_value = False
    app.dependency_overrides[get_delete_session_service] = (
        lambda: mock_delete_session_service
    )

    resp = client.delete("/sessions/session-1/documents/doc-1")
    assert resp.status_code == 404


def test_truncate_messages_success(client, app, mock_delete_session_service):
    mock_delete_session_service.truncate_messages = AsyncMock(return_value=1)

    app.dependency_overrides[get_delete_session_service] = (
        lambda: mock_delete_session_service
    )

    resp = client.put(
        "/sessions/session-1/truncate",
        json={"from_index": 1},
    )

    assert resp.status_code == 200
    assert resp.json()["truncated_count"] == 1


def test_truncate_messages_invalid_index(client, app, mock_delete_session_service):
    mock_delete_session_service.truncate_messages = AsyncMock(
        side_effect=ValueError("from_index must be non-negative")
    )

    app.dependency_overrides[get_delete_session_service] = (
        lambda: mock_delete_session_service
    )

    resp = client.put(
        "/sessions/session-1/truncate",
        json={"from_index": 0},
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "from_index must be non-negative"


def test_truncate_messages_session_not_found(client, app, mock_delete_session_service):
    mock_delete_session_service.truncate_messages = AsyncMock(return_value=None)

    app.dependency_overrides[get_delete_session_service] = (
        lambda: mock_delete_session_service
    )

    resp = client.put(
        "/sessions/missing/truncate",
        json={"from_index": 0},
    )

    assert resp.status_code == 404
    assert "Session not found" in resp.json()["detail"]
