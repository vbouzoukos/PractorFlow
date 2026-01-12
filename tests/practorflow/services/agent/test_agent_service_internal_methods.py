import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from practorflow.services.agent.agent_service import AgentService
from practorflow.llm.base.session import Session
from pydantic_ai.messages import ModelRequest, ModelResponse
from pydantic_ai.messages import UserPromptPart, TextPart

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
async def test_run_planner_json_parse_failure_raises_value_error():
    from practorflow.services.agent.runners import run_planner
    from practorflow.services.agent.context import ExecutionContext

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="not-json"))

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
            return_value=None,
        ),
    ):
        with pytest.raises(ValueError):
            await run_planner(
                task="t",
                ctx=ctx,
                model_pool=MagicMock(),
                model_config=MagicMock(),
                knowledge_store=MagicMock(),
                tool_registry=MagicMock(),
            )


@pytest.mark.asyncio
async def test_run_verifier_heuristic_fallback_on_invalid_json():
    from practorflow.services.agent.runners import run_verifier

    plan = make_plan()
    execution = make_execution_result(plan)

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="json"))

    invalid_verification = {"verification_status": "NOT_A_REAL_STATUS"}

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
            return_value=invalid_verification,
        ),
        patch(
            "practorflow.services.agent.runners.heuristic_verification",
            return_value=make_verification_result(VerificationStatus.PASSED),
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


@pytest.mark.asyncio
async def test_run_executor_happy_path():
    from practorflow.services.agent.runners import run_executor
    from practorflow.services.agent.context import ExecutionContext

    plan = make_plan()

    mock_agent_run = MagicMock()
    mock_agent_run.__aiter__.return_value = []
    mock_agent_run.result = MagicMock(output="ok")

    mock_iter_cm = MagicMock()
    mock_iter_cm.__aenter__.return_value = mock_agent_run
    mock_iter_cm.__aexit__.return_value = False

    agent = MagicMock()
    agent.iter.return_value = mock_iter_cm

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
            "practorflow.services.agent.runners.parse_executor_results",
            return_value=make_execution_result(plan).step_results,
        ),
        patch(
            "practorflow.services.agent.runners.build_execution_log",
            return_value="log",
        ),
    ):
        result = await run_executor(
            plan=plan,
            ctx=ctx,
            model_pool=MagicMock(),
            model_config=MagicMock(),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(),
        )

    assert result.plan_id == plan.plan_id


@pytest.mark.asyncio
async def test_run_planner_success_hits_return_path():
    from practorflow.services.agent.runners import run_planner
    from practorflow.services.agent.context import ExecutionContext

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

    tool_registry = MagicMock()
    tool_registry.get_schemas.return_value = [
        {"function": {"name": "knowledge_search"}},
        {"function": {"name": "calculator"}},
    ]

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
            return_value=parsed,
        ),
    ):
        result = await run_planner(
            task=plan.task,
            ctx=ctx,
            model_pool=MagicMock(),
            model_config=MagicMock(),
            knowledge_store=MagicMock(),
            tool_registry=tool_registry,
        )

    assert result.plan_id == plan.plan_id


@pytest.mark.asyncio
async def test_build_message_history_full_coverage():
    from practorflow.services.agent.context import build_message_history

    user_msg_1 = MagicMock()
    user_msg_1.role = "user"
    user_msg_1.get_text_content.return_value = "hello"

    assistant_msg = MagicMock()
    assistant_msg.role = "assistant"
    assistant_msg.get_text_content.return_value = "hi there"

    user_msg_2 = MagicMock()
    user_msg_2.role = "user"
    user_msg_2.get_text_content.return_value = "final question"

    session = Session(
        session_id="s1",
        user="u1",
        messages=[user_msg_1, assistant_msg, user_msg_2],
    )

    history = build_message_history(session)

    assert len(history) == 2

    assert isinstance(history[0], ModelRequest)
    assert isinstance(history[0].parts[0], UserPromptPart)
    assert history[0].parts[0].content == "hello"

    assert isinstance(history[1], ModelResponse)
    assert isinstance(history[1].parts[0], TextPart)
    assert history[1].parts[0].content == "hi there"
