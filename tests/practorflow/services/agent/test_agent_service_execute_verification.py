import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from practorflow.services.agent.agent_service import AgentService
from practorflow.services.agent.schemas import VerificationStatus

from tests.practorflow.common.fixtures import (
    mock_knowledge_store,
    mock_llm_config,
)
from tests.practorflow.services.agent.common_agent_deps import (
    make_plan,
    make_execution_result,
    make_verification_result,
    make_session,
)


@pytest.fixture
def mock_model_pool():
    pool = MagicMock()

    class _AsyncCtx:
        async def __aenter__(self):
            return MagicMock()

        async def __aexit__(self, exc_type, exc, tb):
            return None

    pool.acquire_context.return_value = _AsyncCtx()
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
async def test_execute_task_verification_failed_no_retry(service):
    plan = make_plan()
    execution = make_execution_result(plan)
    verification = make_verification_result(
        status=VerificationStatus.FAILED,
        retry_recommended=False,
    )

    with (
        patch(
            "practorflow.services.agent.agent_service.run_planner",
            AsyncMock(return_value=plan),
        ),
        patch(
            "practorflow.services.agent.agent_service.run_executor",
            AsyncMock(return_value=execution),
        ),
        patch(
            "practorflow.services.agent.agent_service.run_synthesizer",
            AsyncMock(return_value="test task:1"),
        ),
        patch(
            "practorflow.services.agent.agent_service.run_verifier",
            AsyncMock(return_value=verification),
        ),
        patch(
            "practorflow.services.agent.agent_service.build_failure_message",
            return_value="failure-msg",
        ),
        patch(
            "practorflow.services.agent.agent_service.persist_to_session",
        ) as persist,
    ):
        result = await service.execute_task(
            session_id="s1",
            task="task",
            user="user",
        )

    assert result.success is False
    assert result.error == "failure-msg"
    assert result.verification_result.verification_status == VerificationStatus.FAILED
    persist.assert_not_called()


@pytest.mark.asyncio
async def test_execute_task_verification_partial_no_retry(service):
    plan = make_plan()
    execution = make_execution_result(plan)
    verification = make_verification_result(
        status=VerificationStatus.PARTIAL,
        retry_recommended=False,
    )

    with (
        patch(
            "practorflow.services.agent.agent_service.run_planner",
            AsyncMock(return_value=plan),
        ),
        patch(
            "practorflow.services.agent.agent_service.run_executor",
            AsyncMock(return_value=execution),
        ),
        patch(
            "practorflow.services.agent.agent_service.run_synthesizer",
            AsyncMock(return_value="test task:1"),
        ),
        patch(
            "practorflow.services.agent.agent_service.run_verifier",
            AsyncMock(return_value=verification),
        ),
    ):
        result = await service.execute_task(
            session_id="s2",
            task="task",
            user="user",
        )

    assert result.success is True
    assert result.output == "test task:1"
    assert result.verification_result.verification_status == VerificationStatus.PARTIAL


@pytest.mark.asyncio
async def test_execute_task_verification_failed_after_retries_exhausted(service):
    plan = make_plan()
    plan.retry_policy.max_retries = 1
    execution = make_execution_result(plan)

    verification_retry = make_verification_result(
        status=VerificationStatus.FAILED,
        retry_recommended=True,
    )
    verification_final = make_verification_result(
        status=VerificationStatus.FAILED,
        retry_recommended=False,
    )

    with (
        patch(
            "practorflow.services.agent.agent_service.run_planner",
            AsyncMock(return_value=plan),
        ),
        patch(
            "practorflow.services.agent.agent_service.run_executor",
            AsyncMock(return_value=execution),
        ),
        patch(
            "practorflow.services.agent.agent_service.run_synthesizer",
            AsyncMock(return_value="test task:1"),
        ),
        patch(
            "practorflow.services.agent.agent_service.run_verifier",
            AsyncMock(side_effect=[verification_retry, verification_final]),
        ),
        patch(
            "practorflow.services.agent.agent_service.build_failure_message",
            return_value="final-failure",
        ),
    ):
        result = await service.execute_task(
            session_id="s3",
            task="task",
            user="user",
        )

    assert result.success is False
    assert result.error == "final-failure"
    assert result.verification_result.verification_status == VerificationStatus.FAILED


@pytest.mark.asyncio
async def test_execute_task_verifier_success_path(service):
    plan = make_plan()
    execution = make_execution_result(plan)
    verification = make_verification_result(
        status=VerificationStatus.PASSED,
        retry_recommended=False,
    )

    with (
        patch(
            "practorflow.services.agent.agent_service.run_planner",
            AsyncMock(return_value=plan),
        ),
        patch(
            "practorflow.services.agent.agent_service.run_executor",
            AsyncMock(return_value=execution),
        ),
        patch(
            "practorflow.services.agent.agent_service.run_synthesizer",
            AsyncMock(return_value="final-output"),
        ),
        patch(
            "practorflow.services.agent.agent_service.run_verifier",
            AsyncMock(return_value=verification),
        ),
    ):
        result = await service.execute_task(
            session_id="s1",
            task="task",
            user="user",
        )

    assert result.success is True
    assert result.output == "final-output"


@pytest.mark.asyncio
async def test_run_verifier_successful_json_parsing():
    from practorflow.services.agent.runners import run_verifier

    plan = make_plan()
    execution = make_execution_result(plan)

    verification = make_verification_result(VerificationStatus.PASSED)

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="json"))

    parsed = verification.model_dump()

    with (
        patch(
            "practorflow.services.agent.runners.verifier.create_runner",
            return_value=MagicMock(),
        ),
        patch(
            "practorflow.services.agent.runners.verifier.Agent",
            return_value=agent,
        ),
        patch(
            "practorflow.services.agent.runners.verifier.parse_json_from_response",
            return_value=parsed,
        ),
    ):
        result = await run_verifier(
            plan=plan,
            execution_result=execution,
            model_pool=MagicMock(),
            model_config=MagicMock(),
            knowledge_store=MagicMock(),
        )

    assert result.verification_status == VerificationStatus.PASSED
