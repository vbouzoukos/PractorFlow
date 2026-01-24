"""
Unit tests for ChatService message history building.

Tests:
- build_message_history
"""

from practorflow.llm.base.session import Message, Session
from practorflow.services.history.builder import build_message_history

from tests.practorflow.common.fixtures import mock_knowledge_store
from tests.practorflow.services.chat.common_chat_service import (
    chat_service,
    mock_model_config,
    mock_model_pool,
    mock_session_store,
    mock_web_search_tool,
    sample_session,
)


def test_build_message_history_empty(chat_service):
    """Test building message history with no messages."""
    session = Session(session_id="test", instructions="test", user="test")

    result = build_message_history(session)

    assert result == []


def test_build_message_history_single_message(chat_service):
    """Test building message history with single message excludes last."""
    session = Session(
        session_id="test",
        instructions="test",
        user="test",
        messages=[Message(role="user", content="Hello")],
    )

    result = build_message_history(session)

    assert result == []


def test_build_message_history_two_messages(chat_service):
    """Test building message history with exactly two messages."""
    session = Session(
        session_id="test",
        instructions="test",
        user="test",
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi!"),
        ],
    )

    result = build_message_history(session)

    assert len(result) == 1


def test_build_message_history_multiple_messages(chat_service):
    """Test building message history with multiple messages."""
    session = Session(
        session_id="test",
        instructions="test",
        user="test",
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
            Message(role="user", content="How are you?"),
        ],
    )

    result = build_message_history(session)

    assert len(result) == 2


def test_build_message_history_excludes_last_message(chat_service):
    """Test that build_message_history excludes the last message."""
    session = Session(
        session_id="test",
        instructions="test",
        user="test",
        messages=[
            Message(role="user", content="First"),
            Message(role="assistant", content="Response"),
            Message(role="user", content="Last message"),
        ],
    )

    result = build_message_history(session)

    assert len(result) == 2


def test_build_message_history_preserves_order(chat_service):
    """Test that message history preserves chronological order."""
    session = Session(
        session_id="test",
        instructions="test",
        user="test",
        messages=[
            Message(role="user", content="Message 1"),
            Message(role="assistant", content="Response 1"),
            Message(role="user", content="Message 2"),
            Message(role="assistant", content="Response 2"),
            Message(role="user", content="Message 3"),
        ],
    )

    result = build_message_history(session)

    assert len(result) == 4


def test_build_message_history_long_conversation(chat_service):
    """Test building message history with long conversation."""
    messages = []
    for i in range(10):
        messages.append(Message(role="user", content=f"User message {i}"))
        messages.append(Message(role="assistant", content=f"Assistant response {i}"))
    messages.append(Message(role="user", content="Final user message"))

    session = Session(
        session_id="test",
        instructions="test",
        user="test",
        messages=messages,
    )

    result = build_message_history(session)

    assert len(result) == 20
