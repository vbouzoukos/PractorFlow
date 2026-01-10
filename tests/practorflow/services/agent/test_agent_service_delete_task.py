import pytest
from unittest.mock import MagicMock

from tests.practorflow.common.fixtures import (
    mock_knowledge_store,
    mock_llm_config,
)

from practorflow.services.agent.agent_service import AgentService
from tests.practorflow.services.agent.common_agent_deps import make_session


@pytest.fixture
def mock_model_pool():
    return MagicMock()


@pytest.fixture
def mock_session_store():
    return MagicMock()


@pytest.fixture
def service(
    mock_model_pool,
    mock_llm_config,
    mock_knowledge_store,
    mock_session_store,
):
    return AgentService(
        model_pool=mock_model_pool,
        model_config=mock_llm_config,
        knowledge_store=mock_knowledge_store,
        session_store=mock_session_store,
    )


@pytest.mark.asyncio
async def test_delete_task_session_not_found(service, mock_session_store):
    mock_session_store.exists.return_value = False

    result = await service.delete_task("missing")

    assert result is False
    mock_session_store.delete.assert_not_called()


@pytest.mark.asyncio
async def test_delete_task_deletes_documents_and_session(
    service,
    mock_session_store,
    mock_knowledge_store,
):
    session = make_session("s1")
    session.documents = [{"id": "doc1"}, {"id": "doc2"}]

    mock_session_store.exists.return_value = True
    mock_session_store.get.return_value = session

    result = await service.delete_task("s1")

    assert result is True
    mock_knowledge_store.delete_document.assert_any_call("doc1")
    mock_knowledge_store.delete_document.assert_any_call("doc2")
    mock_session_store.delete.assert_called_once_with("s1")


@pytest.mark.asyncio
async def test_delete_task_document_delete_exception_is_swallowed(
    service,
    mock_session_store,
    mock_knowledge_store,
):
    session = make_session("s2")
    session.documents = [{"id": "doc1"}]

    mock_session_store.exists.return_value = True
    mock_session_store.get.return_value = session
    mock_knowledge_store.delete_document.side_effect = RuntimeError("boom")

    result = await service.delete_task("s2")

    assert result is True
    mock_session_store.delete.assert_called_once_with("s2")
