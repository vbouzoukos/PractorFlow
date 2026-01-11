# tests/practorflow/services/agent/test_agent_service_execute_success.py
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
    make_run_synthesizer_mock,
    make_verification_result,
    make_session,
)


@pytest.fixture
def mock_model_pool():
    pool = MagicMock()
    ctx = AsyncMock()
    pool.acquire_context.return_value.__aenter__.return_value = ctx
    pool.acquire_context.return_value.__aexit__.return_value = None
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
async def test_execute_task_success_with_file_indexing(
    service,
    mock_knowledge_store,
):
    plan = make_plan()
    execution = make_execution_result(plan)
    verification = make_verification_result(VerificationStatus.PASSED)

    file_stream = MagicMock()
    file_stream.read.return_value = b"hello"

    file_mock = MagicMock()
    file_mock.file = file_stream
    file_mock.filename = "a.txt"
    file_mock.content_type = "text/plain"

    files = [file_mock]

    mock_knowledge_store.add_document_from_stream.return_value = {
        "id": "doc-1",
        "filename": "a.txt",
    }

    # ✅ dynamic synthesizer logic (USES plan + execution_result)
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
    ):
        result = await service.execute_task(
            session_id="s1",
            task="do something",
            user="user1",
            files=files,
        )

    assert result.success is True
    assert result.output == "test task:1"

    mock_knowledge_store.add_document_from_stream.assert_called_once_with(
        file_stream=file_stream,
        filename="a.txt",
        mime_type="text/plain",
    )


@pytest.mark.asyncio
async def test_execute_task_uses_existing_session(
    service,
    mock_session_store,
    mock_knowledge_store,
):
    # existing session path
    session = make_session("s1")
    mock_session_store.exists.return_value = True
    mock_session_store.get.return_value = session

    plan = make_plan()
    execution = make_execution_result(plan)
    verification = make_verification_result(VerificationStatus.PASSED)

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
    ):
        result = await service.execute_task(
            session_id="s1",
            task="do something",
            user="user1",
            files=None,
        )

    assert result.success is True
    assert result.output == "test task:1"

    mock_session_store.get.assert_called_once_with("s1")
    mock_session_store.exists.assert_called_once_with("s1")


@pytest.mark.asyncio
async def test_execute_task_retries_then_stops_on_no_more_retries(
    service,
    mock_session_store,
):
    session = make_session("s1")
    mock_session_store.exists.return_value = False

    plan = make_plan()
    plan.retry_policy.max_retries = 1
    execution = make_execution_result(plan)

    verification_retry = make_verification_result(
        status=VerificationStatus.FAILED,
        retry_recommended=True,
    )
    verification_stop = make_verification_result(
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
            AsyncMock(side_effect=[verification_retry, verification_stop]),
        ),
        patch(
            "practorflow.services.agent.agent_service.build_failure_message",
            return_value="failed-msg",
        ),
        patch(
            "practorflow.services.agent.agent_service.persist_to_session",
        ) as persist,
    ):
        result = await service.execute_task(
            session_id="s1",
            task="do something",
            user="user1",
        )

    assert result.success is False
    assert result.error == "failed-msg"
    assert persist.called


@pytest.mark.asyncio
async def test_execute_task_failure_message_and_persist_called(
    service,
    mock_session_store,
):
    session = make_session("s2")
    mock_session_store.exists.return_value = False

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
            return_value="final-failure",
        ),
        patch(
            "practorflow.services.agent.agent_service.persist_to_session",
        ) as persist,
    ):
        result = await service.execute_task(
            session_id="s2",
            task="do something",
            user="user2",
        )

    assert result.success is False
    assert result.error == "final-failure"
    persist.assert_called()


@pytest.mark.asyncio
async def test_run_executor_logs_each_node(service):
    plan = make_plan()

    node = MagicMock()
    node.name = "TestNode"

    mock_agent_run = MagicMock()
    mock_agent_run.__aiter__.return_value = [node]
    mock_agent_run.result = MagicMock(output="ok")

    mock_iter_cm = MagicMock()
    mock_iter_cm.__aenter__.return_value = mock_agent_run
    mock_iter_cm.__aexit__.return_value = False

    agent = MagicMock()
    agent.iter.return_value = mock_iter_cm

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
            "practorflow.services.agent.agent_service.parse_executor_results",
            return_value=make_execution_result(plan).step_results,
        ),
        patch(
            "practorflow.services.agent.agent_service.build_execution_log",
            return_value="log",
        ),
        patch("practorflow.services.agent.agent_service.logger") as mock_logger,
    ):
        await service._run_executor(
            plan,
            MagicMock(),  # deps
        )

    mock_logger.debug.assert_any_call("[AgentService][Executor] node=TestNode")
