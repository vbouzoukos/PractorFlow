"""
Unit tests for ChatService tool registration.

Tests:
- _register_tools
"""

from unittest.mock import MagicMock

import pytest

from tests.practorflow.common.fixtures import mock_knowledge_store
from tests.practorflow.services.chat.common_chat_service import (
    chat_service,
    mock_model_config,
    mock_model_pool,
    mock_session_store,
    mock_web_search_tool,
)


def test_register_tools_registers_search_knowledge(chat_service):
    """Test that _register_tools registers the search_knowledge tool."""
    mock_agent = MagicMock()
    registered_tools = []

    def capture_tool(func):
        registered_tools.append(func.__name__)
        return func

    mock_agent.tool = capture_tool

    chat_service._register_tools(mock_agent)

    assert "search_knowledge" in registered_tools


def test_register_tools_registers_search_web(chat_service):
    """Test that _register_tools registers the search_web tool."""
    mock_agent = MagicMock()
    registered_tools = []

    def capture_tool(func):
        registered_tools.append(func.__name__)
        return func

    mock_agent.tool = capture_tool

    chat_service._register_tools(mock_agent)

    assert "search_web" in registered_tools


def test_register_tools_registers_exactly_two_tools(chat_service):
    """Test that _register_tools registers exactly two tools."""
    mock_agent = MagicMock()
    registered_tools = []

    def capture_tool(func):
        registered_tools.append(func.__name__)
        return func

    mock_agent.tool = capture_tool

    chat_service._register_tools(mock_agent)

    assert len(registered_tools) == 2
    assert set(registered_tools) == {"search_knowledge", "search_web"}


def test_register_tools_uses_agent_tool_decorator(chat_service):
    """Test that _register_tools uses the agent.tool decorator."""
    mock_agent = MagicMock()
    call_count = 0

    def counting_decorator(func):
        nonlocal call_count
        call_count += 1
        return func

    mock_agent.tool = counting_decorator

    chat_service._register_tools(mock_agent)

    assert call_count == 2