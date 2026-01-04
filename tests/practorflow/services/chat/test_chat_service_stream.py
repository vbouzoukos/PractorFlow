"""
Unit tests for ChatService chat streaming.

Tests:
- chat_stream
"""

from io import BytesIO
from unittest.mock import MagicMock, patch

import pytest

from practorflow.services.dto.chat_file import ChatFile

from tests.practorflow.common.fixtures import mock_knowledge_store
from tests.practorflow.services.chat.common_chat_service import (
    chat_service,
    create_mock_model_pool_context,
    create_mock_streaming_context,
    mock_model_config,
    mock_model_pool,
    mock_session_store,
    mock_web_search_tool,
    sample_session,
)

@pytest.mark.asyncio
async def test_chat_stream_new_session(
    chat_service,
    mock_session_store,
    mock_model_pool,
):
    """Test chat_stream creates new session when it doesn't exist."""
    mock_session_store.exists.return_value = False
    create_mock_model_pool_context(mock_model_pool)
    mock_stream_context = create_mock_streaming_context(["Hello", " world"])

    with patch("practorflow.services.chat.chat_service.create_runner"), \
         patch("practorflow.services.chat.chat_service.LocalLLMModel"), \
         patch("practorflow.services.chat.chat_service.Agent") as mock_agent_class:

        mock_agent = MagicMock()
        mock_agent.run_stream.return_value = mock_stream_context
        mock_agent_class.return_value = mock_agent

        chunks = []
        async for chunk in chat_service.chat_stream(
            session_id="new_session_123",
            message="Hello",
            user="test_user",
        ):
            chunks.append(chunk)

        assert len(chunks) == 3
        assert chunks[0].text == "Hello"
        assert chunks[1].text == " world"
        assert chunks[2].finished is True

        mock_session_store.save.assert_called_once()
        saved_session = mock_session_store.save.call_args[0][0]
        assert saved_session.session_id == "new_session_123"
        assert saved_session.user == "test_user"


@pytest.mark.asyncio
async def test_chat_stream_existing_session(
    chat_service,
    mock_session_store,
    mock_model_pool,
    sample_session,
):
    """Test chat_stream uses existing session when it exists."""
    mock_session_store.exists.return_value = True
    mock_session_store.get.return_value = sample_session
    create_mock_model_pool_context(mock_model_pool)
    mock_stream_context = create_mock_streaming_context(["Response"])

    with patch("practorflow.services.chat.chat_service.create_runner"), \
         patch("practorflow.services.chat.chat_service.LocalLLMModel"), \
         patch("practorflow.services.chat.chat_service.Agent") as mock_agent_class:

        mock_agent = MagicMock()
        mock_agent.run_stream.return_value = mock_stream_context
        mock_agent_class.return_value = mock_agent

        chunks = []
        async for chunk in chat_service.chat_stream(
            session_id="session_test123",
            message="New message",
            user="test_user",
        ):
            chunks.append(chunk)

        mock_session_store.get.assert_called_once_with("session_test123")


@pytest.mark.asyncio
async def test_chat_stream_with_files(
    chat_service,
    mock_session_store,
    mock_model_pool,
    mock_knowledge_store,
):
    """Test chat_stream indexes files when provided."""
    mock_session_store.exists.return_value = False
    create_mock_model_pool_context(mock_model_pool)
    mock_stream_context = create_mock_streaming_context(["OK"])

    chat_file = MagicMock(spec=ChatFile)
    chat_file.file = BytesIO(b"content")
    chat_file.filename = "uploaded.txt"
    chat_file.content_type = "text/plain"

    with patch("practorflow.services.chat.chat_service.create_runner"), \
         patch("practorflow.services.chat.chat_service.LocalLLMModel"), \
         patch("practorflow.services.chat.chat_service.Agent") as mock_agent_class:

        mock_agent = MagicMock()
        mock_agent.run_stream.return_value = mock_stream_context
        mock_agent_class.return_value = mock_agent

        chunks = []
        async for chunk in chat_service.chat_stream(
            session_id="session_with_files",
            message="Process this file",
            user="test_user",
            files=[chat_file],
        ):
            chunks.append(chunk)

        mock_knowledge_store.add_document_from_stream.assert_called_once()


@pytest.mark.asyncio
async def test_chat_stream_saves_messages(
    chat_service,
    mock_session_store,
    mock_model_pool,
):
    """Test chat_stream saves user and assistant messages to session."""
    mock_session_store.exists.return_value = False
    create_mock_model_pool_context(mock_model_pool)
    mock_stream_context = create_mock_streaming_context(["Assistant", " response"])

    with patch("practorflow.services.chat.chat_service.create_runner"), \
         patch("practorflow.services.chat.chat_service.LocalLLMModel"), \
         patch("practorflow.services.chat.chat_service.Agent") as mock_agent_class:

        mock_agent = MagicMock()
        mock_agent.run_stream.return_value = mock_stream_context
        mock_agent_class.return_value = mock_agent

        chunks = []
        async for chunk in chat_service.chat_stream(
            session_id="session_messages",
            message="User message",
            user="test_user",
        ):
            chunks.append(chunk)

        mock_session_store.save.assert_called_once()
        saved_session = mock_session_store.save.call_args[0][0]

        assert len(saved_session.messages) == 2
        assert saved_session.messages[0].role == "user"
        assert saved_session.messages[0].content == "User message"
        assert saved_session.messages[1].role == "assistant"
        assert saved_session.messages[1].content == "Assistant response"


@pytest.mark.asyncio
async def test_chat_stream_returns_usage(
    chat_service,
    mock_session_store,
    mock_model_pool,
):
    """Test chat_stream includes usage info in final chunk."""
    mock_session_store.exists.return_value = False
    create_mock_model_pool_context(mock_model_pool)
    mock_stream_context = create_mock_streaming_context(
        ["Done"], input_tokens=100, output_tokens=50
    )

    with patch("practorflow.services.chat.chat_service.create_runner"), \
         patch("practorflow.services.chat.chat_service.LocalLLMModel"), \
         patch("practorflow.services.chat.chat_service.Agent") as mock_agent_class:

        mock_agent = MagicMock()
        mock_agent.run_stream.return_value = mock_stream_context
        mock_agent_class.return_value = mock_agent

        chunks = []
        async for chunk in chat_service.chat_stream(
            session_id="session_usage",
            message="Test",
            user="test_user",
        ):
            chunks.append(chunk)

        final_chunk = chunks[-1]
        assert final_chunk.finished is True
        assert final_chunk.usage is not None
        assert final_chunk.usage["prompt_tokens"] == 100
        assert final_chunk.usage["completion_tokens"] == 50
        assert final_chunk.usage["total_tokens"] == 150


@pytest.mark.asyncio
async def test_chat_stream_enhanced_message_with_files(
    chat_service,
    mock_session_store,
    mock_model_pool,
    mock_knowledge_store,
):
    """Test chat_stream enhances message when files are attached."""
    mock_session_store.exists.return_value = False
    mock_knowledge_store.add_document_from_stream.return_value = {
        "id": "doc-new",
        "filename": "uploaded.txt",
    }
    create_mock_model_pool_context(mock_model_pool)
    mock_stream_context = create_mock_streaming_context(["OK"])

    chat_file = MagicMock(spec=ChatFile)
    chat_file.file = BytesIO(b"content")
    chat_file.filename = "uploaded.txt"
    chat_file.content_type = "text/plain"

    with patch("practorflow.services.chat.chat_service.create_runner"), \
         patch("practorflow.services.chat.chat_service.LocalLLMModel"), \
         patch("practorflow.services.chat.chat_service.Agent") as mock_agent_class:

        mock_agent = MagicMock()
        mock_agent.run_stream.return_value = mock_stream_context
        mock_agent_class.return_value = mock_agent

        chunks = []
        async for chunk in chat_service.chat_stream(
            session_id="session_enhanced",
            message="Process file",
            user="test_user",
            files=[chat_file],
        ):
            chunks.append(chunk)

        call_args = mock_agent.run_stream.call_args
        enhanced_message = call_args[0][0]
        assert "[User attached files:" in enhanced_message
        assert "uploaded.txt" in enhanced_message


@pytest.mark.asyncio
async def test_chat_stream_finish_reason(
    chat_service,
    mock_session_store,
    mock_model_pool,
):
    """Test chat_stream sets finish_reason to stop in final chunk."""
    mock_session_store.exists.return_value = False
    create_mock_model_pool_context(mock_model_pool)
    mock_stream_context = create_mock_streaming_context(["Response"])

    with patch("practorflow.services.chat.chat_service.create_runner"), \
         patch("practorflow.services.chat.chat_service.LocalLLMModel"), \
         patch("practorflow.services.chat.chat_service.Agent") as mock_agent_class:

        mock_agent = MagicMock()
        mock_agent.run_stream.return_value = mock_stream_context
        mock_agent_class.return_value = mock_agent

        chunks = []
        async for chunk in chat_service.chat_stream(
            session_id="session_finish",
            message="Test",
            user="test_user",
        ):
            chunks.append(chunk)

        final_chunk = chunks[-1]
        assert final_chunk.finish_reason == "stop"