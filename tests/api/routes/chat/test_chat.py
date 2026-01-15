import pytest
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient

from api.routes.chat import router
from api.dependencies import get_chat_service, get_session_history
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
def mock_chat_service():
    service = MagicMock()
    service.start_chat = AsyncMock(return_value="session-1")
    service.delete_chat = AsyncMock(return_value=True)
    service.get_session.return_value = MagicMock(messages=[])
    service.delete_session_document = AsyncMock(return_value=True)
    return service


@pytest.fixture
def override_chat_service(mock_chat_service):
    return lambda: mock_chat_service


@pytest.fixture
def override_session_history(mock_session_history):
    return lambda: mock_session_history


def test_start_session(client, app, override_chat_service):
    app.dependency_overrides[get_chat_service] = override_chat_service

    resp = client.get("/chat")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "session-1"


def test_delete_session_success(client, app, override_chat_service):
    app.dependency_overrides[get_chat_service] = override_chat_service

    resp = client.delete("/chat/session-1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True


def test_delete_session_not_found(client, app, mock_chat_service):
    mock_chat_service.delete_chat.return_value = False
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

    resp = client.delete("/chat/missing")
    assert resp.status_code == 404


def test_list_sessions(client, app, override_session_history):
    app.dependency_overrides[get_session_history] = override_session_history

    resp = client.get("/chat/sessions")
    assert resp.status_code == 200
    assert len(resp.json()) == 1


def test_get_history_success(client, app, override_session_history):
    app.dependency_overrides[get_session_history] = override_session_history

    resp = client.get("/chat/session-1/history")
    assert resp.status_code == 200
    assert resp.json()["session_id"] == "session-1"


def test_get_history_not_found(client, app, mock_session_history):
    mock_session_history.get_history.return_value = None
    app.dependency_overrides[get_session_history] = lambda: mock_session_history

    resp = client.get("/chat/missing/history")
    assert resp.status_code == 404


def test_list_documents(client, app, override_session_history):
    app.dependency_overrides[get_session_history] = override_session_history

    resp = client.get("/chat/session-1/documents")
    assert resp.status_code == 200
    assert resp.json()["count"] == 0


def test_list_documents_session_not_found(client, app, mock_session_history):
    mock_session_history.get_history.return_value = None
    app.dependency_overrides[get_session_history] = lambda: mock_session_history

    resp = client.get("/chat/missing/documents")

    assert resp.status_code == 404
    assert "Session not found" in resp.json()["detail"]


def test_delete_document_success(client, app, override_chat_service):
    app.dependency_overrides[get_chat_service] = override_chat_service

    resp = client.delete("/chat/session-1/documents/doc-1")
    assert resp.status_code == 200
    assert resp.json()["deleted"] is True


def test_delete_document_session_not_found(client, app, mock_chat_service):
    mock_chat_service.delete_session_document.return_value = None
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

    resp = client.delete("/chat/session-1/documents/doc-1")
    assert resp.status_code == 404


def test_delete_document_doc_not_found(client, app, mock_chat_service):
    mock_chat_service.delete_session_document.return_value = False
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

    resp = client.delete("/chat/session-1/documents/doc-1")
    assert resp.status_code == 404


def test_chat_stream_success(client, app, mock_chat_service):
    async def fake_stream(*args, **kwargs):
        yield MagicMock(text="hello", finished=False, finish_reason=None, usage=None)
        yield MagicMock(
            text="",
            finished=True,
            finish_reason="stop",
            usage={"total_tokens": 1},
        )

    mock_chat_service.chat_stream = fake_stream
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

    files = {
        "files": ("test.txt", b"dummy content", "text/plain"),
    }

    resp = client.post(
        "/chat/session-1",
        data={"message": "hi"},
        files=files,
    )

    assert resp.status_code == 200
    assert "hello" in resp.text
    assert "[DONE]" in resp.text


def test_chat_stream_value_error(client, app, mock_chat_service):
    async def error_stream(*args, **kwargs):
        raise ValueError("bad input")
        yield

    mock_chat_service.chat_stream = error_stream
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

    resp = client.post("/chat/session-1", data={"message": "hi"})
    assert resp.status_code == 200
    assert "bad input" in resp.text


def test_chat_stream_generic_error(client, app, mock_chat_service):
    async def error_stream(*args, **kwargs):
        raise RuntimeError("boom")
        yield

    mock_chat_service.chat_stream = error_stream
    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

    resp = client.post("/chat/session-1", data={"message": "hi"})
    assert resp.status_code == 200
    assert "Internal server error" in resp.text


def test_truncate_messages_success(client, app, mock_chat_service):
    # session with 2 messages
    mock_session = MagicMock()
    mock_session.truncate_messages.return_value = 1
    mock_session.messages = ["m1"]  # after truncation

    # session store behavior
    mock_store = MagicMock()
    mock_store.exists.return_value = True
    mock_store.get.return_value = mock_session

    mock_chat_service._session_store = mock_store
    mock_chat_service.get_session.return_value = mock_session

    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

    resp = client.put(
        "/chat/session-1/truncate",
        json={"from_index": 1},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["truncated_count"] == 1
    assert body["remaining_count"] == 1


def test_truncate_messages_invalid_index(client, app, mock_chat_service):
    # session.truncate_messages raises ValueError at runtime
    mock_session = MagicMock()
    mock_session.truncate_messages.side_effect = ValueError(
        "from_index must be non-negative"
    )

    mock_store = MagicMock()
    mock_store.exists.return_value = True
    mock_store.get.return_value = mock_session

    mock_chat_service._session_store = mock_store

    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

    resp = client.put(
        "/chat/session-1/truncate",
        json={"from_index": 0},  # valid payload, passes Pydantic
    )

    assert resp.status_code == 400
    assert resp.json()["detail"] == "from_index must be non-negative"


def test_truncate_messages_session_not_found(client, app, mock_chat_service):
    mock_store = MagicMock()
    mock_store.exists.return_value = False

    mock_chat_service._session_store = mock_store
    mock_chat_service.get_session.return_value = None

    app.dependency_overrides[get_chat_service] = lambda: mock_chat_service

    resp = client.put(
        "/chat/missing/truncate",
        json={"from_index": 0},
    )

    assert resp.status_code == 404
    assert "Session not found" in resp.json()["detail"]
