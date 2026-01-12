import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from tests.practorflow.common.fixtures import (
    mock_knowledge_store,
    mock_llm_config,
)

from practorflow.services.agent.agent_service import AgentService
from practorflow.services.agent.schemas import AgentTaskResult
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
async def test_execute_task_planner_raises_value_error(service):
    with patch(
        "practorflow.services.agent.agent_service.run_planner",
        AsyncMock(side_effect=ValueError("planner failed")),
    ):
        result = await service.execute_task(
            session_id="s1",
            task="do something",
            user="u1",
        )

    assert isinstance(result, AgentTaskResult)
    assert result.success is False
    assert "Planning failed" in result.error


@pytest.mark.asyncio
async def test_execute_task_planner_error_persists_session(service):
    with patch(
        "practorflow.services.agent.agent_service.run_planner",
        AsyncMock(side_effect=ValueError("bad plan")),
    ), patch(
        "practorflow.services.agent.agent_service.persist_to_session"
    ) as persist:
        result = await service.execute_task(
            session_id="s2",
            task="x",
            user="u",
        )

    assert result.success is False
    persist.assert_called_once()

@pytest.mark.asyncio
async def test_run_planner_invalid_plan_structure_raises_value_error():
    from practorflow.services.agent.runners import run_planner
    from practorflow.services.agent.context import ExecutionContext

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="json"))

    invalid_plan = {
        "plan_id": "p1",
        "task": "task",
        # missing required fields -> ValidationError
    }

    tool_registry = MagicMock()
    tool_registry.get_schemas.return_value = []

    ctx = ExecutionContext(
        session=make_session("s1"),
        message_history=[],
        document_scope=None,
        document_context=None,
    )

    with (
        patch(
            "practorflow.services.agent.runners.create_runner",
            return_value=MagicMock(),
        ),
        patch(
            "practorflow.services.agent.runners.Agent",
            return_value=agent,
        ),
        patch(
            "practorflow.services.agent.runners.parse_json_from_response",
            return_value=invalid_plan,
        ),
    ):
        with pytest.raises(ValueError):
            await run_planner(
                task="task",
                ctx=ctx,
                model_pool=MagicMock(),
                model_config=MagicMock(),
                knowledge_store=MagicMock(),
                tool_registry=tool_registry,
            )
