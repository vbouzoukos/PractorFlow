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
