from unittest.mock import MagicMock, patch
import pytest

from tests.practorflow.common.fixtures import (
    mock_knowledge_store,
    mock_llm_config,
)

from practorflow.services.agent.agent_service import AgentService
from practorflow.llm.tools.tool_registry import ToolRegistry
from tests.practorflow.services.agent.common_agent_deps import make_session


@pytest.fixture
def mock_model_pool():
    return MagicMock()


@pytest.fixture
def mock_session_store():
    store = MagicMock()
    store.exists.return_value = False
    store.get.side_effect = lambda sid: make_session(sid)
    return store


def test_init_registers_default_tools_with_internal_registry(
    mock_model_pool,
    mock_llm_config,
    mock_knowledge_store,
    mock_session_store,
):
    with patch(
        "practorflow.services.agent.agent_service.register_default_tools"
    ) as register_default_tools:
        service = AgentService(
            model_pool=mock_model_pool,
            model_config=mock_llm_config,
            knowledge_store=mock_knowledge_store,
            session_store=mock_session_store,
            tool_registry=None,
        )

        assert service._tool_registry is not None
        register_default_tools.assert_called_once_with(
            service._tool_registry,
            mock_knowledge_store,
        )


def test_init_uses_provided_tool_registry(
    mock_model_pool,
    mock_llm_config,
    mock_knowledge_store,
    mock_session_store,
):
    provided_registry = MagicMock()  # MUST be truthy

    with patch(
        "practorflow.services.agent.agent_service.register_default_tools"
    ) as register_default_tools:
        service = AgentService(
            model_pool=mock_model_pool,
            model_config=mock_llm_config,
            knowledge_store=mock_knowledge_store,
            session_store=mock_session_store,
            tool_registry=provided_registry,
        )

        assert service._tool_registry is provided_registry
        register_default_tools.assert_called_once_with(
            provided_registry,
            mock_knowledge_store,
        )


@pytest.mark.asyncio
async def test_start_task_generates_agent_session_id(
    mock_model_pool,
    mock_llm_config,
    mock_knowledge_store,
    mock_session_store,
):
    service = AgentService(
        model_pool=mock_model_pool,
        model_config=mock_llm_config,
        knowledge_store=mock_knowledge_store,
        session_store=mock_session_store,
    )

    session_id = await service.start_task()

    assert session_id.startswith("agent_")


def test_get_session_returns_none_when_not_exists(
    mock_model_pool,
    mock_llm_config,
    mock_knowledge_store,
    mock_session_store,
):
    mock_session_store.exists.return_value = False

    service = AgentService(
        model_pool=mock_model_pool,
        model_config=mock_llm_config,
        knowledge_store=mock_knowledge_store,
        session_store=mock_session_store,
    )

    assert service.get_session("missing") is None


def test_get_session_returns_session_when_exists(
    mock_model_pool,
    mock_llm_config,
    mock_knowledge_store,
    mock_session_store,
):
    mock_session_store.exists.return_value = True

    service = AgentService(
        model_pool=mock_model_pool,
        model_config=mock_llm_config,
        knowledge_store=mock_knowledge_store,
        session_store=mock_session_store,
    )

    session = service.get_session("sess-1")

    assert session is not None
    assert session.session_id == "sess-1"
