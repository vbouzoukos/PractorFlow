"""
Unit tests for ChatService document handling.

Tests:
- _index_file
- _get_document_scope
"""

from io import BytesIO
from unittest.mock import MagicMock

import pytest

from practorflow.llm.base.session import Session
from practorflow.services.dto.chat_file import ChatFile

from tests.practorflow.services.chat.common_chat_service import (
    chat_service,
    mock_knowledge_store,
    mock_model_config,
    mock_model_pool,
    mock_session_store,
    mock_web_search_tool,
    sample_chat_file,
)


# ============================================================================
# Test ChatService._index_file
# ============================================================================


@pytest.mark.asyncio
async def test_index_file(chat_service, mock_knowledge_store, sample_chat_file):
    """Test file indexing."""
    doc_info = await chat_service._index_file(sample_chat_file)

    assert doc_info == {
        "id": "doc-123",
        "filename": "test.txt",
        "content": "test content",
    }
    mock_knowledge_store.add_document_from_stream.assert_called_once_with(
        file_stream=sample_chat_file.file,
        filename=sample_chat_file.filename,
        mime_type=sample_chat_file.content_type,
    )


@pytest.mark.asyncio
async def test_index_file_returns_store_response(chat_service, mock_knowledge_store):
    """Test _index_file returns exactly what knowledge store returns."""
    expected_doc_info = {
        "id": "custom-id",
        "filename": "custom.pdf",
        "size": 12345,
        "extra_field": "extra_value",
    }
    mock_knowledge_store.add_document_from_stream.return_value = expected_doc_info

    chat_file = MagicMock(spec=ChatFile)
    chat_file.file = BytesIO(b"content")
    chat_file.filename = "custom.pdf"
    chat_file.content_type = "application/pdf"

    result = await chat_service._index_file(chat_file)

    assert result == expected_doc_info


@pytest.mark.asyncio
async def test_index_file_passes_correct_mime_type(chat_service, mock_knowledge_store):
    """Test _index_file passes correct mime type to knowledge store."""
    chat_file = MagicMock(spec=ChatFile)
    chat_file.file = BytesIO(b"pdf content")
    chat_file.filename = "document.pdf"
    chat_file.content_type = "application/pdf"

    await chat_service._index_file(chat_file)

    mock_knowledge_store.add_document_from_stream.assert_called_once()
    call_kwargs = mock_knowledge_store.add_document_from_stream.call_args[1]
    assert call_kwargs["mime_type"] == "application/pdf"


# ============================================================================
# Test ChatService._get_document_scope
# ============================================================================


def test_get_document_scope_with_documents(chat_service):
    """Test getting document scope with documents."""
    session = Session(
        session_id="test",
        instructions="test",
        user="test",
        documents=[
            {"id": "doc-1", "filename": "file1.txt"},
            {"id": "doc-2", "filename": "file2.txt"},
        ],
    )

    result = chat_service._get_document_scope(session)

    assert result == {"doc-1", "doc-2"}
    assert isinstance(result, set)


def test_get_document_scope_no_documents(chat_service):
    """Test getting document scope with no documents."""
    session = Session(session_id="test", instructions="test", user="test")

    result = chat_service._get_document_scope(session)

    assert result is None


def test_get_document_scope_empty_documents_list(chat_service):
    """Test getting document scope with empty documents list."""
    session = Session(
        session_id="test",
        instructions="test",
        user="test",
        documents=[],
    )

    result = chat_service._get_document_scope(session)

    assert result is None


def test_get_document_scope_documents_without_ids(chat_service):
    """Test getting document scope with documents missing IDs."""
    session = Session(
        session_id="test",
        instructions="test",
        user="test",
        documents=[
            {"filename": "file1.txt"},
            {"id": "doc-2", "filename": "file2.txt"},
        ],
    )

    result = chat_service._get_document_scope(session)

    assert result == {"doc-2"}


def test_get_document_scope_all_documents_without_ids(chat_service):
    """Test getting document scope when all documents lack IDs."""
    session = Session(
        session_id="test",
        instructions="test",
        user="test",
        documents=[
            {"filename": "file1.txt"},
            {"filename": "file2.txt"},
        ],
    )

    result = chat_service._get_document_scope(session)

    assert result == set()


def test_get_document_scope_single_document(chat_service):
    """Test getting document scope with single document."""
    session = Session(
        session_id="test",
        instructions="test",
        user="test",
        documents=[{"id": "only-doc", "filename": "only.txt"}],
    )

    result = chat_service._get_document_scope(session)

    assert result == {"only-doc"}