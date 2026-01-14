from pydantic_ai.messages import ModelMessage, ModelRequest
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from practorflow.services.agent.runners import (
    _prepare_history,
    run_planner,
    run_executor,
    run_synthesizer,
    run_verifier,
)
from practorflow.services.agent.context import ExecutionContext
from practorflow.services.history.estimator import estimate_tokens


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
        "practorflow.services.history.preparer.prepare_history",
        AsyncMock(
            return_value=MagicMock(
                messages=msgs[-1:],
                was_truncated=True,
                original_count=1,
                included_count=1,
                estimated_tokens=25,
            )
        ),
    ):
        history = await _prepare_history(
            task="task",
            ctx=ctx,
            system_prompt="sys",
            task_prompt="task",
            model_pool=MagicMock(),
            model_config=model_config,
        )

    assert history == msgs[-1:]


@pytest.mark.asyncio
async def test_prepare_history_exceeds_context_with_memory():
    msgs = [_make_msg("user", "x" * 100)]
    ctx = _make_ctx(msgs)
    model_config = MagicMock(n_ctx=10)

    with patch(
        "practorflow.services.history.preparer.extract_relevant_memory",
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

    # memory message is injected FIRST
    assert len(history) == 2
    assert "important" in history[0].parts[0].content
    assert history[1] is msgs[0]


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

    prepared_history = [MagicMock()]  # 👈 makes the branch execute

    with (
        patch(
            "practorflow.services.agent.runners._prepare_history",
            AsyncMock(return_value=prepared_history),
        ),
        patch("practorflow.services.agent.runners.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.create_runner"),
        patch(
            "practorflow.services.agent.runners.parse_json_from_response",
            return_value=plan_dict,
        ),
        patch("practorflow.services.agent.runners.logger.debug") as debug_logger,
    ):
        plan = await run_planner(
            task="t",
            ctx=ctx,
            model_pool=MagicMock(),
            model_config=MagicMock(n_ctx=100),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(get_schemas=lambda: []),
        )

    # ensures the uncovered line is executed
    debug_logger.assert_any_call("[Planner] Using 1 history messages")
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


@pytest.mark.asyncio
async def test_run_planner_invalid_plan_structure_raises_value_error():
    ctx = _make_ctx([])

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="json"))

    # Parsed JSON is truthy but structurally invalid for Plan
    invalid_plan = {
        "plan_id": "p1"
        # missing required fields: task, steps, success_criteria, retry_policy
    }

    with (
        patch("practorflow.services.agent.runners.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.create_runner"),
        patch(
            "practorflow.services.agent.runners.parse_json_from_response",
            return_value=invalid_plan,
        ),
        patch("practorflow.services.agent.runners.logger.error") as error_logger,
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

    # ensures the ValidationError branch is executed
    error_logger.assert_called_once()


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

    # fake node yielded by the async iterator
    node = MagicMock()
    node.name = "test_node"

    async def async_iter():
        yield node

    agent_run = MagicMock()
    agent_run.__aiter__.side_effect = async_iter
    agent_run.result = MagicMock(output="ok")

    agent = MagicMock()
    agent.iter.return_value.__aenter__.return_value = agent_run
    agent.iter.return_value.__aexit__.return_value = False

    with (
        patch(
            "practorflow.services.agent.runners._prepare_history",
            AsyncMock(return_value=[MagicMock()]),
        ),
        patch("practorflow.services.agent.runners.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.create_runner"),
        patch(
            "practorflow.services.agent.runners.parse_executor_results",
            return_value=[],
        ),
        patch(
            "practorflow.services.agent.runners.build_execution_log",
            return_value="log",
        ),
        patch("practorflow.services.agent.runners.logger.debug") as debug_logger,
    ):
        result = await run_executor(
            plan=plan,
            ctx=ctx,
            model_pool=MagicMock(),
            model_config=MagicMock(n_ctx=100),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(),
        )

    # ensures the uncovered logger.debug line is executed
    debug_logger.assert_any_call("[Executor] node=test_node")
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
            AsyncMock(return_value=[MagicMock()]),
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
async def test_run_verifier_success_path():
    from practorflow.services.agent.runners import run_verifier

    plan = MagicMock()
    plan.plan_id = "p1"

    execution = MagicMock()

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="json"))

    valid_verification = {
        "verification_status": "passed",
        "failed_criteria": [],
        "issues": [],
        "retry_recommended": False,
    }

    handle = MagicMock()
    handle.__aenter__.return_value = handle
    handle.__aexit__.return_value = False

    model_pool = MagicMock()
    model_pool.acquire_context.return_value = handle

    with (
        patch("practorflow.services.agent.runners.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.create_runner"),
        patch(
            "practorflow.services.agent.runners.parse_json_from_response",
            return_value=valid_verification,
        ),
        patch(
            "practorflow.services.agent.runners.logger.warning"
        ) as warning_logger,
        patch(
            "practorflow.services.agent.runners.logger.info"
        ) as info_logger,
    ):
        result = await run_verifier(
            plan=plan,
            execution_result=execution,
            model_pool=model_pool,
            model_config=MagicMock(n_ctx=100),
            knowledge_store=MagicMock(),
        )

    assert result.verification_status.value == "passed"
    warning_logger.assert_not_called()
    info_logger.assert_called_once()

@pytest.mark.asyncio
async def test_run_verifier_invalid_structure_uses_heuristic():
    from practorflow.services.agent.runners import run_verifier

    plan = MagicMock()
    plan.plan_id = "p1"

    execution = MagicMock()

    agent = MagicMock()
    agent.run = AsyncMock(
        return_value=MagicMock(
            output={"verification_status": "success"}  # ❌ missing required fields
        )
    )

    invalid_verification = {
        "verification_status": "success"
        # missing required fields → ValidationError
    }

    handle = MagicMock()
    handle.__aenter__.return_value = handle
    handle.__aexit__.return_value = False

    model_pool = MagicMock()
    model_pool.acquire_context.return_value = handle

    with (
        patch("practorflow.services.agent.runners.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.create_runner"),
        patch(
            "practorflow.services.agent.runners.parse_json_from_response",
            return_value=invalid_verification,
        ),
        patch(
            "practorflow.services.agent.runners.heuristic_verification",
            return_value="heuristic-result",
        ) as heuristic,
        patch(
            "practorflow.services.agent.runners.logger.warning"
        ) as warning_logger,
    ):
        result = await run_verifier(
            plan=plan,
            execution_result=execution,
            model_pool=model_pool,
            model_config=MagicMock(n_ctx=100),
            knowledge_store=MagicMock(),
        )

    assert result == "heuristic-result"
    warning_logger.assert_called_once()
    heuristic.assert_called_once()
