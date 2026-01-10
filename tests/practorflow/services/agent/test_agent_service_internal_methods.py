import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from tests.practorflow.common.fixtures import (
    mock_knowledge_store,
    mock_llm_config,
)

from practorflow.services.agent.agent_service import AgentService
from practorflow.services.agent.schemas import VerificationStatus
from tests.practorflow.services.agent.common_agent_deps import (
    make_plan,
    make_execution_result,
    make_verification_result,
    make_session,
)


@pytest.fixture
def mock_model_pool():
    pool = MagicMock()

    handle = MagicMock()
    handle.backend = "llama_cpp"

    class _AsyncContext:
        async def __aenter__(self):
            return handle

        async def __aexit__(self, exc_type, exc, tb):
            return None

    pool.acquire_context.return_value = _AsyncContext()
    return pool


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
async def test_run_planner_json_parse_failure_raises_value_error(service):
    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="not-json"))

    with patch(
        "practorflow.services.agent.agent_service.create_runner",
        return_value=MagicMock(),
    ), patch(
        "practorflow.services.agent.agent_service.Agent",
        return_value=agent,
    ), patch(
        "practorflow.services.agent.agent_service.parse_json_from_response",
        return_value=None,
    ):
        with pytest.raises(ValueError):
            await service._run_planner(
                task="t",
                document_context=None,
            )


@pytest.mark.asyncio
async def test_run_verifier_heuristic_fallback_on_invalid_json(service):
    plan = make_plan()
    execution = make_execution_result(plan)

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="not-json"))

    with patch(
        "practorflow.services.agent.agent_service.create_runner",
        return_value=MagicMock(),
    ), patch(
        "practorflow.services.agent.agent_service.Agent",
        return_value=agent,
    ), patch(
        "practorflow.services.agent.agent_service.parse_json_from_response",
        return_value=None,
    ), patch(
        "practorflow.services.agent.agent_service.heuristic_verification",
        return_value=make_verification_result(VerificationStatus.PASSED),
    ):
        result = await service._run_verifier(
            plan,
            execution,
        )

    assert result.verification_status == VerificationStatus.PASSED


@pytest.mark.asyncio
async def test_run_executor_happy_path(service):
    plan = make_plan()

    mock_agent_run = MagicMock()
    mock_agent_run.__aiter__.return_value = []
    mock_agent_run.result = MagicMock(output="ok")

    mock_iter_cm = MagicMock()
    mock_iter_cm.__aenter__.return_value = mock_agent_run
    mock_iter_cm.__aexit__.return_value = False

    agent = MagicMock()
    agent.iter.return_value = mock_iter_cm

    with patch(
        "practorflow.services.agent.agent_service.create_runner",
        return_value=MagicMock(),
    ), patch(
        "practorflow.services.agent.agent_service.Agent",
        return_value=agent,
    ), patch(
        "practorflow.services.agent.agent_service.parse_executor_results",
        return_value=make_execution_result(plan).step_results,
    ), patch(
        "practorflow.services.agent.agent_service.build_execution_log",
        return_value="log",
    ):
        result = await service._run_executor(
            plan,
            MagicMock(),  # deps
        )

    assert result.plan_id == plan.plan_id

@pytest.mark.asyncio
async def test_run_planner_success_hits_return_path(service, caplog):
    plan = make_plan()

    parsed = {
        "plan_id": plan.plan_id,
        "task": plan.task,
        "steps": [s.model_dump() for s in plan.steps],
        "success_criteria": plan.success_criteria,
        "retry_policy": {"max_retries": plan.retry_policy.max_retries},
    }

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="json"))

    with (
        patch(
            "practorflow.services.agent.agent_service.create_runner",
            return_value=MagicMock(),
        ),
        patch(
            "practorflow.services.agent.agent_service.Agent",
            return_value=agent,
        ),
        patch(
            "practorflow.services.agent.agent_service.parse_json_from_response",
            return_value=parsed,
        ),
    ):
        result = await service._run_planner(
            task=plan.task,
            document_context=None,
        )

    # forces execution past model_validate → logger → return
    assert isinstance(result, type(plan))
    assert result.plan_id == plan.plan_id
