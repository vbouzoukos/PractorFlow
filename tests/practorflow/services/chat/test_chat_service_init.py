"""
Unit tests for ChatService initialization.

Tests:
- __init__ with all dependencies
- __init__ with default web search tool
- __init__ with default instructions
- _generate_session_id
"""

from practorflow.llm.tools.base_web_search import DuckDuckGoSearchTool
from practorflow.services.chat.chat_service import ChatService

from tests.practorflow.common.fixtures import mock_knowledge_store
from tests.practorflow.services.chat.common_chat_service import (
    mock_model_config,
    mock_model_pool,
    mock_session_store,
    mock_web_search_tool,
)


def test_chat_service_init(
    mock_model_pool,
    mock_model_config,
    mock_knowledge_store,
    mock_session_store,
    mock_web_search_tool,
):
    """Test ChatService initialization with all dependencies."""
    service = ChatService(
        model_pool=mock_model_pool,
        model_config=mock_model_config,
        knowledge_store=mock_knowledge_store,
        session_store=mock_session_store,
        web_search_tool=mock_web_search_tool,
        default_instructions="Test instructions",
    )

    assert service._model_pool == mock_model_pool
    assert service._model_config == mock_model_config
    assert service._knowledge_store == mock_knowledge_store
    assert service._session_store == mock_session_store
    assert service._web_search_tool == mock_web_search_tool
    assert "Test instructions" in service._instructions


def test_chat_service_init_default_web_search(
    mock_model_pool,
    mock_model_config,
    mock_knowledge_store,
    mock_session_store,
):
    """Test ChatService initialization with default web search tool."""
    service = ChatService(
        model_pool=mock_model_pool,
        model_config=mock_model_config,
        knowledge_store=mock_knowledge_store,
        session_store=mock_session_store,
    )

    assert isinstance(service._web_search_tool, DuckDuckGoSearchTool)


def test_chat_service_init_default_instructions(
    mock_model_pool,
    mock_model_config,
    mock_knowledge_store,
    mock_session_store,
):
    """Test ChatService initialization with default instructions."""
    service = ChatService(
        model_pool=mock_model_pool,
        model_config=mock_model_config,
        knowledge_store=mock_knowledge_store,
        session_store=mock_session_store,
    )

    assert service._instructions is not None
    assert len(service._instructions) > 0


def test_generate_session_id(
    mock_model_pool,
    mock_model_config,
    mock_knowledge_store,
    mock_session_store,
):
    """Test session ID generation produces unique IDs."""
    service = ChatService(
        model_pool=mock_model_pool,
        model_config=mock_model_config,
        knowledge_store=mock_knowledge_store,
        session_store=mock_session_store,
    )

    session_id1 = service._generate_session_id()
    session_id2 = service._generate_session_id()

    assert isinstance(session_id1, str)
    assert len(session_id1) > 0
    assert session_id1 != session_id2


def test_generate_session_id_format(
    mock_model_pool,
    mock_model_config,
    mock_knowledge_store,
    mock_session_store,
):
    """Test session ID has expected format."""
    service = ChatService(
        model_pool=mock_model_pool,
        model_config=mock_model_config,
        knowledge_store=mock_knowledge_store,
        session_store=mock_session_store,
    )

    session_id = service._generate_session_id()

    assert session_id.startswith("session_")
    assert len(session_id) > len("session_")