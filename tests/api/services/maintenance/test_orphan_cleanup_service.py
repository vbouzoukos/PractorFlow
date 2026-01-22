import pytest
from unittest.mock import Mock

from src.api.services.maintenance.orphan_cleanup_service import OrphanCleanupService


class DummySession:
    def __init__(self, documents):
        self.documents = documents


@pytest.fixture
def knowledge_store():
    return Mock()


@pytest.fixture
def session_history():
    return Mock()


@pytest.fixture
def service(knowledge_store, session_history):
    return OrphanCleanupService(
        knowledge_store=knowledge_store,
        session_history=session_history,
    )


def test_get_session_document_ids_empty(service, session_history):
    session_history.list_sessions.return_value = []

    result = service.get_session_document_ids()

    assert result == set()


def test_get_session_document_ids_with_documents(service, session_history):
    sessions = [
        DummySession(documents=[{"id": "doc1"}, {"id": "doc2"}]),
        DummySession(documents=[{"id": "doc3"}, {"id": None}, {}]),
    ]
    session_history.list_sessions.return_value = sessions

    result = service.get_session_document_ids()

    assert result == {"doc1", "doc2", "doc3"}


def test_get_knowledge_store_document_ids(service, knowledge_store):
    knowledge_store.list_documents.return_value = [
        {"id": "doc1"},
        {"id": "doc2"},
    ]

    result = service.get_knowledge_store_document_ids()

    assert result == {"doc1", "doc2"}


def test_find_orphan_document_ids(service, knowledge_store, session_history):
    session_history.list_sessions.return_value = [
        DummySession(documents=[{"id": "doc1"}]),
    ]
    knowledge_store.list_documents.return_value = [
        {"id": "doc1"},
        {"id": "doc2"},
        {"id": "doc3"},
    ]

    result = service.find_orphan_document_ids()

    assert result == {"doc2", "doc3"}


def test_cleanup_no_orphans(service):
    service.find_orphan_document_ids = Mock(return_value=set())

    result = service.cleanup()

    assert result == 0


def test_cleanup_successful_deletions(service, knowledge_store):
    service.find_orphan_document_ids = Mock(return_value={"doc1", "doc2"})
    knowledge_store.delete_document.side_effect = [True, True]

    result = service.cleanup()

    assert result == 2
    assert knowledge_store.delete_document.call_count == 2


def test_cleanup_document_not_found(service, knowledge_store):
    service.find_orphan_document_ids = Mock(return_value={"doc1"})
    knowledge_store.delete_document.return_value = False

    result = service.cleanup()

    assert result == 0
    knowledge_store.delete_document.assert_called_once_with("doc1")


def test_cleanup_delete_exception(service, knowledge_store):
    service.find_orphan_document_ids = Mock(return_value={"doc1"})
    knowledge_store.delete_document.side_effect = Exception("fail")

    result = service.cleanup()

    assert result == 0
    knowledge_store.delete_document.assert_called_once_with("doc1")
