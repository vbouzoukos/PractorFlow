import pytest
from unittest.mock import MagicMock

from practorflow.llm.base.session import Session

from tests.practorflow.common.fixtures import mock_knowledge_store
from tests.practorflow.services.chat.common_chat_service import (
    chat_service,
    mock_model_config,      
    mock_model_pool,        
    mock_session_store,
    mock_web_search_tool,
)


# ============================================================================
# Test ChatService.delete_session_document
# ============================================================================


@pytest.mark.asyncio
async def test_delete_session_document_session_not_found(
    chat_service,
    mock_session_store,
    mock_knowledge_store,
):
    """Returns None when session does not exist."""
    mock_session_store.exists.return_value = False

    result = await chat_service.delete_session_document(
        session_id="missing-session",
        document_id="doc-1",
    )

    assert result is None
    mock_session_store.exists.assert_called_once_with("missing-session")
    mock_session_store.get.assert_not_called()
    mock_session_store.save.assert_not_called()
    mock_knowledge_store.delete_document.assert_not_called()


@pytest.mark.asyncio
async def test_delete_session_document_document_not_found(
    chat_service,
    mock_session_store,
    mock_knowledge_store,
):
    """Returns False when document is not found in the session."""
    session = MagicMock(spec=Session)
    session.remove_document.return_value = False

    mock_session_store.exists.return_value = True
    mock_session_store.get.return_value = session

    result = await chat_service.delete_session_document(
        session_id="session-123",
        document_id="missing-doc",
    )

    assert result is False
    session.remove_document.assert_called_once_with("missing-doc")
    mock_knowledge_store.delete_document.assert_not_called()
    mock_session_store.save.assert_not_called()


@pytest.mark.asyncio
async def test_delete_session_document_success(
    chat_service,
    mock_session_store,
    mock_knowledge_store,
):
    """Successfully deletes document from session and knowledge store."""
    session = MagicMock(spec=Session)
    session.remove_document.return_value = True

    mock_session_store.exists.return_value = True
    mock_session_store.get.return_value = session

    result = await chat_service.delete_session_document(
        session_id="session-123",
        document_id="doc-1",
    )

    assert result is True
    session.remove_document.assert_called_once_with("doc-1")
    mock_knowledge_store.delete_document.assert_called_once_with("doc-1")
    mock_session_store.save.assert_called_once_with(session)


@pytest.mark.asyncio
async def test_delete_session_document_knowledge_store_failure(
    chat_service,
    mock_session_store,
    mock_knowledge_store,
):
    """Knowledge store deletion failure is swallowed and session is still saved."""
    session = MagicMock(spec=Session)
    session.remove_document.return_value = True

    mock_session_store.exists.return_value = True
    mock_session_store.get.return_value = session
    mock_knowledge_store.delete_document.side_effect = Exception("delete failed")

    result = await chat_service.delete_session_document(
        session_id="session-123",
        document_id="doc-1",
    )

    assert result is True
    session.remove_document.assert_called_once_with("doc-1")
    mock_knowledge_store.delete_document.assert_called_once_with("doc-1")
    mock_session_store.save.assert_called_once_with(session)
