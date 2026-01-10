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
    with patch.object(
        service,
        "_run_planner",
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
async def test_execute_task_planner_error_persists_session(service, mock_session_store):
    with patch.object(
        service,
        "_run_planner",
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
async def test_run_planner_invalid_plan_structure_raises_value_error(service):
    # Parsed JSON is present but structurally invalid
    invalid_plan = {
        "plan_id": "p1",
        "task": "test",
        # ❌ steps must be a list of objects, not strings
        "steps": ["not-a-step"],
        "success_criteria": ["ok"],
        "retry_policy": {"max_retries": 1},
    }

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="ignored"))

    with patch(
        "practorflow.services.agent.agent_service.create_runner",
        return_value=MagicMock(),
    ), patch(
        "practorflow.services.agent.agent_service.Agent",
        return_value=agent,
    ), patch(
        "practorflow.services.agent.agent_service.parse_json_from_response",
        return_value=invalid_plan,
    ):
        with pytest.raises(ValueError):
            await service._run_planner(
                task="t",
                document_context=None,
            )

