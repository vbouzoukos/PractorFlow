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
    make_run_synthesizer_mock,
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


# ---------------------------------------------------------------------
# 1️⃣ Verification FAILED — no retry
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_execute_task_verification_failed_no_retry(service):
    plan = make_plan()
    execution = make_execution_result(plan)
    verification = make_verification_result(
        status=VerificationStatus.FAILED,
        retry_recommended=False,
    )

    def synth_logic(plan, execution_result):
        return f"{plan.task}:{len(execution_result.step_results)}"

    with (
        patch.object(service, "_run_planner", AsyncMock(return_value=plan)),
        patch.object(service, "_run_executor", AsyncMock(return_value=execution)),
        patch.object(
            service,
            "_run_synthesizer",
            make_run_synthesizer_mock(synth_logic),
        ),
        patch.object(service, "_run_verifier", AsyncMock(return_value=verification)),
        patch(
            "practorflow.services.agent.agent_service.build_failure_message",
            return_value="failure-msg",
        ) as build_failure,
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
    build_failure.assert_called_once()
    persist.assert_called()


# ---------------------------------------------------------------------
# 2️⃣ Verification PARTIAL — no retry
# ---------------------------------------------------------------------
@pytest.mark.asyncio
async def test_execute_task_verification_partial_no_retry(service):
    plan = make_plan()
    execution = make_execution_result(plan)
    verification = make_verification_result(
        status=VerificationStatus.PARTIAL,
        retry_recommended=False,
    )

    def synth_logic(plan, execution_result):
        return f"{plan.task}:{len(execution_result.step_results)}"

    with (
        patch.object(service, "_run_planner", AsyncMock(return_value=plan)),
        patch.object(service, "_run_executor", AsyncMock(return_value=execution)),
        patch.object(
            service,
            "_run_synthesizer",
            make_run_synthesizer_mock(synth_logic),
        ),
        patch.object(service, "_run_verifier", AsyncMock(return_value=verification)),
        patch(
            "practorflow.services.agent.agent_service.build_failure_message",
            return_value="partial-failure",
        ) as build_failure,
        patch(
            "practorflow.services.agent.agent_service.persist_to_session",
        ) as persist,
    ):
        result = await service.execute_task(
            session_id="s2",
            task="task",
            user="user",
        )

    assert result.success is True
    assert result.verification_result.verification_status == VerificationStatus.PARTIAL
    assert result.output == "test task:1"


# ---------------------------------------------------------------------
# 3️⃣ Verification FAILED after retries exhausted
# ---------------------------------------------------------------------
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

    def synth_logic(plan, execution_result):
        return f"{plan.task}:{len(execution_result.step_results)}"

    with (
        patch.object(service, "_run_planner", AsyncMock(return_value=plan)),
        patch.object(service, "_run_executor", AsyncMock(return_value=execution)),
        patch.object(
            service,
            "_run_synthesizer",
            make_run_synthesizer_mock(synth_logic),
        ),
        patch.object(
            service,
            "_run_verifier",
            AsyncMock(side_effect=[verification_retry, verification_final]),
        ) as run_verifier,
        patch(
            "practorflow.services.agent.agent_service.build_failure_message",
            return_value="final-failure",
        ),
        patch(
            "practorflow.services.agent.agent_service.persist_to_session",
        ),
    ):
        result = await service.execute_task(
            session_id="s3",
            task="task",
            user="user",
        )

    assert result.success is False
    assert result.error == "final-failure"
    assert run_verifier.call_count == 2


@pytest.mark.asyncio
async def test_execute_task_verifier_success_path(service):
    plan = make_plan()
    execution = make_execution_result(plan)

    agent = MagicMock()
    agent.run = AsyncMock(
        return_value=MagicMock(
            output={
                "verification_status": "PASSED",
                "failed_criteria": [],
                "issues": [],
                "retry_recommended": False,
            }
        )
    )

    with (
        patch.object(service, "_run_planner", AsyncMock(return_value=plan)),
        patch.object(service, "_run_executor", AsyncMock(return_value=execution)),
        patch(
            "practorflow.services.agent.agent_service.Agent",
            return_value=agent,
        ),
        patch(
            "practorflow.services.agent.agent_service.create_runner",
            return_value=MagicMock(),
        ),
        patch(
            "practorflow.services.agent.agent_service.parse_json_from_response",
            return_value={
                "verification_status": "PASSED",
                "failed_criteria": [],
                "issues": [],
                "retry_recommended": False,
            },
        ),
        patch(
            "practorflow.services.agent.agent_service.extract_final_output",
            return_value="final-output",
        ),
    ):
        result = await service.execute_task(
            session_id="s1",
            task="task",
            user="user",
        )

    assert result.success is True


@pytest.mark.asyncio
async def test_run_verifier_successful_json_parsing(service):
    """Test _run_verifier successfully parses valid JSON and returns VerificationResult."""
    plan = make_plan()
    execution = make_execution_result(plan)

    agent_mock = MagicMock()
    agent_mock.run = AsyncMock(return_value=MagicMock(output="some json string"))

    mock_parse = MagicMock(
        return_value={
            "verification_status": "passed",
            "failed_criteria": [],
            "issues": [],
            "retry_recommended": False,
        }
    )

    with (
        patch(
            "practorflow.services.agent.agent_service.Agent",
            return_value=agent_mock,
        ),
        patch(
            "practorflow.services.agent.agent_service.create_runner",
            return_value=MagicMock(),
        ),
        patch(
            "practorflow.services.agent.agent_service.LocalLLMModel",
            return_value=MagicMock(),
        ),
        patch(
            "practorflow.services.agent.agent_service.parse_json_from_response",
            mock_parse,
        ),
        patch(
            "practorflow.services.agent.agent_service.build_verifier_prompt",
            return_value="prompt",
        ),
    ):
        result = await service._run_verifier(plan, execution)

    assert mock_parse.called, "parse_json_from_response was not called"
    assert result.verification_status == VerificationStatus.PASSED
