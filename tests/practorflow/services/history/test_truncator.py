import pytest
from unittest.mock import MagicMock

from practorflow.llm.base.session import Message, Session
from practorflow.services.history.truncator import DeleteSessionService


@pytest.fixture
def sample_session():
    """Sample session for testing."""
    return Session(
        session_id="session_test123",
        instructions="Test instructions",
        user="user",
        messages=[
            Message(role="user", content="Hello"),
            Message(role="assistant", content="Hi there!"),
        ],
        documents=[
            {"id": "doc-1", "filename": "file1.txt"},
            {"id": "doc-2", "filename": "file2.txt"},
        ],
    )


@pytest.mark.asyncio
async def test_truncate_messages_session_not_found_returns_none():
    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = False
    service = DeleteSessionService(knowledge_store, session_store)
    result = await service.truncate_messages(
        session_id="missing", from_index=1, request_user="a"
    )

    assert result is None
    session_store.get.assert_not_called()
    session_store.save.assert_not_called()


@pytest.mark.asyncio
async def test_truncate_messages_successful_truncation():
    session = MagicMock()
    session.user = "user"
    session.truncate_messages.return_value = 3

    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = True
    session_store.get.return_value = session
    service = DeleteSessionService(knowledge_store, session_store)
    result = await service.truncate_messages(
        session_id="session_1", from_index=2, request_user="user"
    )

    assert result == 3
    session.truncate_messages.assert_called_once_with(2)
    session_store.save.assert_called_once_with(session)


@pytest.mark.asyncio
async def test_truncate_messages_propagates_value_error():
    session = MagicMock()
    session.truncate_messages.side_effect = ValueError("negative index")
    session.user = "user"
    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = True
    session_store.get.return_value = session

    with pytest.raises(ValueError):
        service = DeleteSessionService(knowledge_store, session_store)
        await service.truncate_messages(
            session_id="session_1", from_index=-1, request_user="user"
        )

    session_store.save.assert_not_called()


@pytest.mark.asyncio
async def test_truncate_messages_invalid_permission():
    session = MagicMock()
    session.user = "user"
    session.truncate_messages.return_value = 3

    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = True
    session_store.get.return_value = session

    with pytest.raises(PermissionError):
        service = DeleteSessionService(knowledge_store, session_store)
        await service.truncate_messages(
            session_id="session_1", from_index=2, request_user="hax"
        )

    session_store.save.assert_not_called()


@pytest.mark.asyncio
async def test_delete_session_document_session_not_found():
    """Returns None when session does not exist."""
    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = False

    service = DeleteSessionService(knowledge_store, session_store)
    result = await service.delete_session_document(
        session_id="missing-session", document_id="doc-1", request_user="user"
    )

    assert result is None
    session_store.exists.assert_called_once_with("missing-session")
    session_store.get.assert_not_called()
    session_store.save.assert_not_called()
    knowledge_store.delete_document.assert_not_called()


@pytest.mark.asyncio
async def test_delete_session_document_document_not_found():
    """Returns False when document is not found in the session."""
    session = MagicMock()
    session.user = "user"
    session.remove_document.return_value = False

    session_store = MagicMock()
    knowledge_store = MagicMock()

    session_store.exists.return_value = True
    session_store.get.return_value = session

    service = DeleteSessionService(knowledge_store, session_store)
    result = await service.delete_session_document(
        session_id="session-123", document_id="missing-doc", request_user="user"
    )

    assert result is False
    session.remove_document.assert_called_once_with("missing-doc")
    knowledge_store.delete_document.assert_not_called()
    session_store.save.assert_not_called()


@pytest.mark.asyncio
async def test_delete_session_document_success():
    """Successfully deletes document from session and knowledge store."""
    session = MagicMock()
    session.user = "user"
    session.remove_document.return_value = True
    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = True
    session_store.get.return_value = session

    service = DeleteSessionService(knowledge_store, session_store)
    result = await service.delete_session_document(
        session_id="session-123", document_id="doc-1", request_user="user"
    )

    assert result is True
    session.remove_document.assert_called_once_with("doc-1")
    knowledge_store.delete_document.assert_called_once_with("doc-1")
    session_store.save.assert_called_once_with(session)


@pytest.mark.asyncio
async def test_delete_session_no_permission():
    """No permission to delete document from session and knowledge store."""
    session = MagicMock()
    session.user = "user"
    session.remove_document.return_value = True
    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = True
    session_store.get.return_value = session

    with pytest.raises(PermissionError):
        service = DeleteSessionService(knowledge_store, session_store)
        await service.delete_session_document(
            session_id="session-123", document_id="doc-1", request_user="hax"
        )

    knowledge_store.delete_document.assert_not_called()


@pytest.mark.asyncio
async def test_delete_session_document_knowledge_store_failure():
    """Knowledge store deletion failure is swallowed and session is still saved."""
    session = MagicMock()
    session.user = "user"
    session.remove_document.return_value = True

    session_store = MagicMock()
    knowledge_store = MagicMock()

    session_store.exists.return_value = True
    session_store.get.return_value = session
    knowledge_store.delete_document.side_effect = Exception("delete failed")

    service = DeleteSessionService(knowledge_store, session_store)
    result = await service.delete_session_document(
        session_id="session-123", document_id="doc-1", request_user="user"
    )

    assert result is True
    session.remove_document.assert_called_once_with("doc-1")
    knowledge_store.delete_document.assert_called_once_with("doc-1")
    session_store.save.assert_called_once_with(session)


@pytest.mark.asyncio
async def test_delete_chat_success(sample_session):
    """Test successful chat deletion with document cleanup."""
    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = True
    session_store.get.return_value = sample_session

    service = DeleteSessionService(knowledge_store, session_store)
    result = await service.delete_chat("session_test123", "user")

    assert result is True
    session_store.exists.assert_called_once_with("session_test123")
    session_store.get.assert_called_once_with("session_test123")
    session_store.delete.assert_called_once_with("session_test123")

    assert knowledge_store.delete_document.call_count == 2
    knowledge_store.delete_document.assert_any_call("doc-1")
    knowledge_store.delete_document.assert_any_call("doc-2")


@pytest.mark.asyncio
async def test_delete_chat_no_permission(sample_session):
    """Chat deletion permission error."""
    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = True
    session_store.get.return_value = sample_session

    with pytest.raises(PermissionError):
        service = DeleteSessionService(knowledge_store, session_store)
        await service.delete_chat("session_test123", "hax")

    knowledge_store.delete_document.assert_not_called()


@pytest.mark.asyncio
async def test_delete_chat_not_found():
    """Test deleting a non-existent session."""
    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = False

    service = DeleteSessionService(knowledge_store, session_store)
    result = await service.delete_chat("nonexistent", "user")

    assert result is False
    session_store.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_chat_document_deletion_error(sample_session):
    """Test chat deletion continues even if document deletion fails."""
    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = True
    session_store.get.return_value = sample_session
    knowledge_store.delete_document.side_effect = Exception("Delete failed")

    service = DeleteSessionService(knowledge_store, session_store)
    result = await service.delete_chat("session_test123", "user")

    assert result is True
    session_store.delete.assert_called_once_with("session_test123")


@pytest.mark.asyncio
async def test_delete_chat_no_documents():
    """Test deleting a session with no documents."""
    session = Session(
        session_id="session_no_docs",
        instructions="Test",
        user="user",
        documents=[],
    )
    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = True
    session_store.get.return_value = session

    service = DeleteSessionService(knowledge_store, session_store)
    result = await service.delete_chat("session_no_docs", "user")

    assert result is True
    knowledge_store.delete_document.assert_not_called()
    session_store.delete.assert_called_once_with("session_no_docs")


@pytest.mark.asyncio
async def test_delete_chat_documents_without_ids():
    """Test deleting session with documents missing IDs."""
    session = Session(
        session_id="session_partial_docs",
        instructions="Test",
        user="user",
        documents=[
            {"filename": "no_id.txt"},
            {"id": "doc-with-id", "filename": "has_id.txt"},
        ],
    )
    session_store = MagicMock()
    knowledge_store = MagicMock()
    session_store.exists.return_value = True
    session_store.get.return_value = session

    service = DeleteSessionService(knowledge_store, session_store)
    result = await service.delete_chat("session_partial_docs", "user")

    assert result is True
    knowledge_store.delete_document.assert_called_once_with("doc-with-id")
