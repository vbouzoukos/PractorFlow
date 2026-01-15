import pytest
from unittest.mock import MagicMock
from fastapi import FastAPI

from api.auth.schemas import UserContext


@pytest.fixture
def mock_current_user():
    return UserContext(user_id="test-user", is_authenticated=True)


@pytest.fixture
def override_auth(mock_current_user):
    async def _override():
        return mock_current_user
    return _override


@pytest.fixture
def mock_session():
    session = MagicMock()
    session.session_id = "session-1"
    session.user = "test-user"
    session.instructions = "instructions"
    session.documents = []
    session.messages = []
    session.created_at.isoformat.return_value = "2024-01-01T00:00:00"
    session.updated_at.isoformat.return_value = "2024-01-01T00:00:00"
    return session


@pytest.fixture
def mock_session_history(mock_session):
    history = MagicMock()
    history.list_sessions.return_value = [mock_session]
    history.get_history.return_value = mock_session
    return history


@pytest.fixture
def app_base():
    return FastAPI()
