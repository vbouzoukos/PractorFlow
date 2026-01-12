import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from pydantic_ai.messages import ModelRequest, UserPromptPart

from practorflow.services.agent.runners import (
    _estimate_tokens,
    _estimate_messages_tokens,
    _messages_to_text,
    _estimate_total_context,
    _prepare_history,
    run_planner,
    run_executor,
    run_synthesizer,
    run_verifier,
)
from practorflow.services.agent.context import ExecutionContext
from practorflow.services.agent.schemas import (
    StepStatus,
    VerificationStatus,
)
from practorflow.llm.base.session import Session


# ------------------------
# helpers
# ------------------------


def _make_msg(role: str, text: str):
    msg = MagicMock()
    msg.role = role
    msg.parts = [MagicMock(content=text)]
    return msg


def _make_ctx(history):
    session = MagicMock()
    session.session_id = "s1"
    return ExecutionContext(
        session=session,
        message_history=history,
        document_scope=None,
        document_context=None,
    )


# ------------------------
# token estimation helpers
# ------------------------


def test_estimate_tokens_empty():
    assert _estimate_tokens("") == 0


def test_estimate_tokens_non_empty():
    assert _estimate_tokens("abcd") == 1


def test_estimate_messages_tokens():
    msgs = [_make_msg("user", "abcd"), _make_msg("assistant", "abcdefgh")]
    assert _estimate_messages_tokens(msgs) == 1 + 2


def test_messages_to_text_user_and_assistant():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="hello")]),
        _make_msg("assistant", "hi"),
    ]

    text = _messages_to_text(msgs)

    assert "User: hello" in text
    assert "Assistant: hi" in text


def test_estimate_total_context():
    msgs = [_make_msg("user", "abcd")]
    total = _estimate_total_context("sys", "task", msgs)
    assert total > 0


# ------------------------
# _prepare_history
# ------------------------


@pytest.mark.asyncio
async def test_prepare_history_no_history():
    ctx = _make_ctx([])
    model_config = MagicMock(n_ctx=100)

    history = await _prepare_history(
        task="task",
        ctx=ctx,
        system_prompt="sys",
        task_prompt="task",
        model_pool=MagicMock(),
        model_config=model_config,
    )

    assert history == []


@pytest.mark.asyncio
async def test_prepare_history_fits_context():
    msgs = [_make_msg("user", "short")]
    ctx = _make_ctx(msgs)
    model_config = MagicMock(n_ctx=10_000)

    history = await _prepare_history(
        task="task",
        ctx=ctx,
        system_prompt="sys",
        task_prompt="task",
        model_pool=MagicMock(),
        model_config=model_config,
    )

    assert history == msgs


@pytest.mark.asyncio
async def test_prepare_history_exceeds_context_no_memory():
    msgs = [_make_msg("user", "x" * 100)]
    ctx = _make_ctx(msgs)
    model_config = MagicMock(n_ctx=1)

    with patch(
        "practorflow.services.agent.runners._extract_relevant_memory",
        AsyncMock(return_value=""),
    ):
        history = await _prepare_history(
            task="task",
            ctx=ctx,
            system_prompt="sys",
            task_prompt="task",
            model_pool=MagicMock(),
            model_config=model_config,
        )

    assert isinstance(history, list)


@pytest.mark.asyncio
async def test_prepare_history_exceeds_context_with_memory():
    msgs = [_make_msg("user", "x" * 100)]
    ctx = _make_ctx(msgs)
    model_config = MagicMock(n_ctx=10)

    with patch(
        "practorflow.services.agent.runners._extract_relevant_memory",
        AsyncMock(return_value="important"),
    ):
        history = await _prepare_history(
            task="task",
            ctx=ctx,
            system_prompt="sys",
            task_prompt="task",
            model_pool=MagicMock(),
            model_config=model_config,
        )

    assert isinstance(history[0], ModelRequest)
    assert "Relevant context" in history[0].parts[0].content


# ------------------------
# run_planner
# ------------------------


@pytest.mark.asyncio
async def test_run_planner_success():
    ctx = _make_ctx([])
    plan_dict = {
        "plan_id": "p1",
        "task": "t",
        "steps": [
            {
                "step_id": "s1",
                "description": "d",
                "tool": None,
                "tool_args": None,
                "expected_output": "x",
            }
        ],
        "success_criteria": ["done"],
        "retry_policy": {"max_retries": 1},
    }

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="json"))

    with (
        patch("practorflow.services.agent.runners.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.create_runner"),
        patch(
            "practorflow.services.agent.runners.parse_json_from_response",
            return_value=plan_dict,
        ),
    ):
        plan = await run_planner(
            task="t",
            ctx=ctx,
            model_pool=MagicMock(),
            model_config=MagicMock(n_ctx=100),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(get_schemas=lambda: []),
        )

    assert plan.plan_id == "p1"


@pytest.mark.asyncio
async def test_run_planner_parse_failure_raises():
    ctx = _make_ctx([])

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="bad"))

    with (
        patch("practorflow.services.agent.runners.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.create_runner"),
        patch(
            "practorflow.services.agent.runners.parse_json_from_response",
            return_value=None,
        ),
        patch("practorflow.services.agent.runners.repair_plan_json", return_value=None),
    ):
        with pytest.raises(ValueError):
            await run_planner(
                task="t",
                ctx=ctx,
                model_pool=MagicMock(),
                model_config=MagicMock(n_ctx=100),
                knowledge_store=MagicMock(),
                tool_registry=MagicMock(get_schemas=lambda: []),
            )


# ------------------------
# run_executor
# ------------------------

@pytest.mark.asyncio
async def test_run_executor_success():
    ctx = MagicMock()
    ctx.has_history = True
    ctx.message_history = [MagicMock()]
    ctx.document_scope = None

    plan = MagicMock(plan_id="p1", task="t", steps=[])

    agent_run = MagicMock()
    agent_run.__aiter__.return_value = []
    agent_run.result = MagicMock(output="ok")

    agent = MagicMock()
    agent.iter.return_value.__aenter__.return_value = agent_run
    agent.iter.return_value.__aexit__.return_value = False

    with (
        patch(
            "practorflow.services.agent.runners._prepare_history",
            AsyncMock(return_value=[MagicMock()]),  # <-- KEY
        ),
        patch("practorflow.services.agent.runners.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.create_runner"),
        patch(
            "practorflow.services.agent.runners.parse_executor_results", return_value=[]
        ),
        patch(
            "practorflow.services.agent.runners.build_execution_log", return_value="log"
        ),
    ):
        result = await run_executor(
            plan=plan,
            ctx=ctx,
            model_pool=MagicMock(),
            model_config=MagicMock(n_ctx=100),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(),
        )

    assert result.plan_id == "p1"


# ------------------------
# run_synthesizer
# ------------------------


@pytest.mark.asyncio
async def test_run_synthesizer_success():
    ctx = MagicMock()
    ctx.has_history = True
    ctx.message_history = [MagicMock()]

    plan = MagicMock(plan_id="p1", task="t")
    execution = MagicMock(step_results=[])

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="final"))

    with (
        patch(
            "practorflow.services.agent.runners._prepare_history",
            AsyncMock(return_value=[MagicMock()]),  # <-- KEY
        ),
        patch("practorflow.services.agent.runners.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.create_runner"),
    ):
        output = await run_synthesizer(
            plan=plan,
            execution_result=execution,
            ctx=ctx,
            model_pool=MagicMock(),
            model_config=MagicMock(n_ctx=100),
            knowledge_store=MagicMock(),
        )

    assert output == "final"

# ------------------------
# run_verifier
# ------------------------


@pytest.mark.asyncio
async def test_run_verifier_heuristic_fallback():
    plan = MagicMock()
    execution = MagicMock()

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="bad"))

    with (
        patch("practorflow.services.agent.runners.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.create_runner"),
        patch(
            "practorflow.services.agent.runners.parse_json_from_response",
            return_value=None,
        ),
        patch("practorflow.services.agent.runners.heuristic_verification") as hv,
    ):
        await run_verifier(
            plan=plan,
            execution_result=execution,
            model_pool=MagicMock(),
            model_config=MagicMock(n_ctx=100),
            knowledge_store=MagicMock(),
        )

    hv.assert_called_once()


@pytest.mark.asyncio
async def test_extract_relevant_memory_success():
    from practorflow.services.agent.runners import _extract_relevant_memory

    msg = MagicMock()
    msg.parts = [MagicMock(content="important context")]

    runner = MagicMock()
    runner.generate = AsyncMock(return_value={"reply": "This is relevant memory"})

    handle = MagicMock()
    handle.__aenter__.return_value = handle
    handle.__aexit__.return_value = False

    model_pool = MagicMock()
    model_pool.acquire_context.return_value = handle

    with patch(
        "practorflow.services.agent.runners.create_runner",
        return_value=runner,
    ):
        memory = await _extract_relevant_memory(
            task="task",
            messages=[msg],
            model_pool=model_pool,
            model_config=MagicMock(),
            target_tokens=10,
        )

    assert memory == "This is relevant memory"


@pytest.mark.asyncio
async def test_extract_relevant_memory_no_relevant_context():
    from practorflow.services.agent.runners import _extract_relevant_memory

    msg = MagicMock()
    msg.parts = [MagicMock(content="irrelevant chat")]

    runner = MagicMock()
    runner.generate = AsyncMock(return_value={"reply": "No relevant prior context."})

    handle = MagicMock()
    handle.__aenter__.return_value = handle
    handle.__aexit__.return_value = False

    model_pool = MagicMock()
    model_pool.acquire_context.return_value = handle

    with patch(
        "practorflow.services.agent.runners.create_runner",
        return_value=runner,
    ):
        memory = await _extract_relevant_memory(
            task="task",
            messages=[msg],
            model_pool=model_pool,
            model_config=MagicMock(),
            target_tokens=10,
        )

    assert memory == ""


@pytest.mark.asyncio
async def test_prepare_history_no_memory_collects_recent_messages():
    from practorflow.services.agent.runners import _prepare_history

    msg = MagicMock()
    msg.parts = [MagicMock(content="abcd")]  # 1 token

    ctx = MagicMock()
    ctx.has_history = True
    ctx.message_history = [msg]

    model_config = MagicMock(n_ctx=2)

    with patch(
        "practorflow.services.agent.runners._extract_relevant_memory",
        AsyncMock(return_value=""),
    ):
        history = await _prepare_history(
            task="task",
            ctx=ctx,
            system_prompt="",
            task_prompt="",
            model_pool=MagicMock(),
            model_config=model_config,
        )

    assert history == [msg]


@pytest.mark.asyncio
async def test_prepare_history_with_memory_collects_recent_messages():
    from practorflow.services.agent.runners import _prepare_history

    msg = MagicMock()
    msg.parts = [MagicMock(content="abcd")]  # 1 token

    ctx = MagicMock()
    ctx.has_history = True
    ctx.message_history = [msg]

    model_config = MagicMock(n_ctx=10)

    with (
        patch(
            "practorflow.services.agent.runners._estimate_total_context",
            return_value=100,  # FORCE memory path
        ),
        patch(
            "practorflow.services.agent.runners._extract_relevant_memory",
            AsyncMock(return_value="memory"),  # memory_tokens = 1
        ),
    ):
        history = await _prepare_history(
            task="task",
            ctx=ctx,
            system_prompt="",
            task_prompt="",
            model_pool=MagicMock(),
            model_config=model_config,
        )

    assert len(history) == 2
    assert isinstance(history[0], ModelRequest)
    assert history[1] is msg


@pytest.mark.asyncio
async def test_run_planner_logs_when_history_present():
    from practorflow.services.agent.runners import run_planner

    ctx = MagicMock()
    ctx.has_history = True
    ctx.message_history = [MagicMock()]
    ctx.has_documents = False
    ctx.document_context = None

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="json"))

    valid_plan = {
        "plan_id": "p1",
        "task": "t",
        "steps": [
            {
                "step_id": "s1",
                "description": "d",
                "tool": None,
                "tool_args": None,
                "expected_output": "x",
            }
        ],
        "success_criteria": ["done"],
        "retry_policy": {"max_retries": 1},
    }

    with (
        patch(
            "practorflow.services.agent.runners._prepare_history",
            AsyncMock(return_value=[MagicMock()]),
        ),
        patch(
            "practorflow.services.agent.runners.Agent",
            return_value=agent,
        ),
        patch(
            "practorflow.services.agent.runners.create_runner",
        ),
        patch(
            "practorflow.services.agent.runners.parse_json_from_response",
            return_value=valid_plan,
        ),
    ):
        plan = await run_planner(
            task="t",
            ctx=ctx,
            model_pool=MagicMock(),
            model_config=MagicMock(n_ctx=10),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(get_schemas=lambda: []),
        )

    assert plan.plan_id == "p1"

@pytest.mark.asyncio
async def test_prepare_history_no_memory_collects_recent_messages():
    from practorflow.services.agent.runners import _prepare_history

    msg = MagicMock()
    msg.parts = [MagicMock(content="abcd")]  # 1 token

    ctx = MagicMock()
    ctx.has_history = True
    ctx.message_history = [msg]

    model_config = MagicMock(n_ctx=2)

    with (
        patch(
            "practorflow.services.agent.runners._estimate_total_context",
            return_value=100,  # force memory path
        ),
        patch(
            "practorflow.services.agent.runners._extract_relevant_memory",
            AsyncMock(return_value=""),  # force no-memory branch
        ),
    ):
        history = await _prepare_history(
            task="task",
            ctx=ctx,
            system_prompt="",
            task_prompt="",
            model_pool=MagicMock(),
            model_config=model_config,
        )

    assert history == [msg]

