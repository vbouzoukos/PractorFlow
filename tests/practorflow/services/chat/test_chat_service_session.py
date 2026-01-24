"""
Unit tests for ChatService session management.

Tests:
- start_chat
- get_session
- delete_chat
"""

import pytest

from practorflow.llm.base.session import Session

from tests.practorflow.common.fixtures import mock_knowledge_store
from tests.practorflow.services.chat.common_chat_service import (
    chat_service,
    mock_model_config,
    mock_model_pool,
    mock_session_store,
    mock_web_search_tool,
    sample_session,
)

# ============================================================================
# Test ChatService.start_chat
# ============================================================================


@pytest.mark.asyncio
async def test_start_chat(chat_service):
    """Test starting a new chat session."""
    session_id = await chat_service.start_chat()

    assert session_id is not None
    assert isinstance(session_id, str)
    assert len(session_id) > 0


@pytest.mark.asyncio
async def test_start_chat_unique_ids(chat_service):
    """Test start_chat generates unique session IDs."""
    session_id1 = await chat_service.start_chat()
    session_id2 = await chat_service.start_chat()

    assert session_id1 != session_id2


@pytest.mark.asyncio
async def test_start_chat_returns_valid_format(chat_service):
    """Test start_chat returns properly formatted session ID."""
    session_id = await chat_service.start_chat()

    assert session_id.startswith("session_")


# ============================================================================
# Test ChatService.get_session
# ============================================================================


def test_get_session(chat_service, mock_session_store, sample_session):
    """Test getting a session delegates to session store."""
    mock_session_store.exists.return_value = True
    mock_session_store.get.return_value = sample_session

    result = chat_service.get_session("session_test123")

    assert result == sample_session
    mock_session_store.get.assert_called_once_with("session_test123")


def test_get_session_returns_none_when_not_exists(chat_service, mock_session_store):
    """Test get_session returns None when session doesn't exist."""
    mock_session_store.exists.return_value = False

    result = chat_service.get_session("nonexistent")

    assert result is None
    mock_session_store.exists.assert_called_once_with("nonexistent")
    mock_session_store.get.assert_not_called()
