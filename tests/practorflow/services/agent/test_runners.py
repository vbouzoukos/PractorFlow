from pydantic_ai.messages import ModelMessage, ModelRequest
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from practorflow.services.agent.runners import (
    run_planner,
    adapt_plan,
    run_executor,
    run_synthesizer,
    run_verifier,
)
from practorflow.services.agent.runners.agent_context import (
    build_context_enhanced_prompt as _build_context_enhanced_prompt,
    extract_context as _extract_context,
    prepare_agent_history as _prepare_history,
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
        "practorflow.services.agent.runners.agent_context.prepare_history",
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
            "practorflow.services.agent.runners.planner.prepare_agent_history",
            AsyncMock(return_value=prepared_history),
        ),
        patch("practorflow.services.agent.runners.planner.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.planner.create_runner"),
        patch(
            "practorflow.services.agent.runners.planner.parse_json_from_response",
            return_value=plan_dict,
        ),
        patch("practorflow.services.agent.runners.planner.logger.debug") as debug_logger,
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
        patch("practorflow.services.agent.runners.planner.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.planner.create_runner"),
        patch(
            "practorflow.services.agent.runners.planner.parse_json_from_response",
            return_value=None,
        ),
        patch("practorflow.services.agent.runners.planner.repair_plan_json", return_value=None),
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
        patch("practorflow.services.agent.runners.planner.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.planner.create_runner"),
        patch(
            "practorflow.services.agent.runners.planner.parse_json_from_response",
            return_value=invalid_plan,
        ),
        patch("practorflow.services.agent.runners.planner.logger.error") as error_logger,
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
# adapt_plan
# ------------------------


@pytest.mark.asyncio
async def test_adapt_plan_success_with_history():
    ctx = _make_ctx([MagicMock()])  # non-empty history triggers debug log line 195
    plan_dict = {
        "plan_id": "p2",
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
        patch(
            "practorflow.services.agent.runners.planner.prepare_agent_history",
            AsyncMock(return_value=[MagicMock()]),
        ),
        patch("practorflow.services.agent.runners.planner.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.planner.create_runner"),
        patch(
            "practorflow.services.agent.runners.planner.parse_json_from_response",
            return_value=plan_dict,
        ),
        patch("practorflow.services.agent.runners.planner.logger.debug") as debug_logger,
    ):
        plan = await adapt_plan(
            task="t",
            failed_plan=MagicMock(),
            execution_result=MagicMock(),
            verification_result=MagicMock(),
            ctx=ctx,
            model_pool=MagicMock(),
            model_config=MagicMock(n_ctx=100),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(get_schemas=lambda: []),
            attempt_number=1,
        )

    debug_logger.assert_any_call("[AdaptPlan] Using 1 history messages")
    assert plan.plan_id == "p2"


@pytest.mark.asyncio
async def test_adapt_plan_parse_failure_raises():
    ctx = _make_ctx([])

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="bad"))

    with (
        patch(
            "practorflow.services.agent.runners.planner.prepare_agent_history",
            AsyncMock(return_value=[]),
        ),
        patch("practorflow.services.agent.runners.planner.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.planner.create_runner"),
        patch(
            "practorflow.services.agent.runners.planner.parse_json_from_response",
            return_value=None,
        ),
        patch("practorflow.services.agent.runners.planner.repair_plan_json", return_value=None),
    ):
        with pytest.raises(ValueError, match="valid JSON"):
            await adapt_plan(
                task="t",
                failed_plan=MagicMock(),
                execution_result=MagicMock(),
                verification_result=MagicMock(),
                ctx=ctx,
                model_pool=MagicMock(),
                model_config=MagicMock(n_ctx=100),
                knowledge_store=MagicMock(),
                tool_registry=MagicMock(get_schemas=lambda: []),
                attempt_number=1,
            )


@pytest.mark.asyncio
async def test_adapt_plan_invalid_structure_raises():
    ctx = _make_ctx([])

    agent = MagicMock()
    agent.run = AsyncMock(return_value=MagicMock(output="json"))

    invalid_plan = {"plan_id": "p1"}  # missing required fields

    with (
        patch(
            "practorflow.services.agent.runners.planner.prepare_agent_history",
            AsyncMock(return_value=[]),
        ),
        patch("practorflow.services.agent.runners.planner.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.planner.create_runner"),
        patch(
            "practorflow.services.agent.runners.planner.parse_json_from_response",
            return_value=invalid_plan,
        ),
        patch("practorflow.services.agent.runners.planner.logger.error") as error_logger,
    ):
        with pytest.raises(ValueError, match="Invalid plan structure"):
            await adapt_plan(
                task="t",
                failed_plan=MagicMock(),
                execution_result=MagicMock(),
                verification_result=MagicMock(),
                ctx=ctx,
                model_pool=MagicMock(),
                model_config=MagicMock(n_ctx=100),
                knowledge_store=MagicMock(),
                tool_registry=MagicMock(get_schemas=lambda: []),
                attempt_number=1,
            )

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
            "practorflow.services.agent.runners.executor.prepare_agent_history",
            AsyncMock(return_value=[MagicMock()]),
        ),
        patch("practorflow.services.agent.runners.executor.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.executor.create_runner"),
        patch(
            "practorflow.services.agent.runners.executor.parse_executor_results",
            return_value=[],
        ),
        patch(
            "practorflow.services.agent.runners.executor.build_execution_log",
            return_value="log",
        ),
        patch("practorflow.services.agent.runners.executor.logger.debug") as debug_logger,
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

    agent_run = MagicMock()
    agent_run.__aiter__.return_value = []
    agent_run.result = MagicMock(output="final")

    agent = MagicMock()
    agent.iter.return_value.__aenter__.return_value = agent_run
    agent.iter.return_value.__aexit__.return_value = False

    with (
        patch(
            "practorflow.services.agent.runners.synthesizer.prepare_agent_history",
            AsyncMock(return_value=[MagicMock()]),
        ),
        patch("practorflow.services.agent.runners.synthesizer.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.synthesizer.create_runner"),
        patch(
            "practorflow.services.agent.runners.synthesizer.extract_context",
            AsyncMock(return_value=(None, None)),
        ),
    ):
        output = await run_synthesizer(
            plan=plan,
            execution_result=execution,
            ctx=ctx,
            model_pool=MagicMock(),
            model_config=MagicMock(n_ctx=100),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(),
        )

    assert output == "final"


@pytest.mark.asyncio
async def test_run_synthesizer_with_instructions_success():
    ctx = MagicMock()
    ctx.has_history = True
    ctx.message_history = [MagicMock()]

    plan = MagicMock(plan_id="p1", task="t")
    execution = MagicMock(step_results=[])

    agent_run = MagicMock()
    agent_run.__aiter__.return_value = []
    agent_run.result = MagicMock(output="I am helpful")

    agent = MagicMock()
    agent.iter.return_value.__aenter__.return_value = agent_run
    agent.iter.return_value.__aexit__.return_value = False

    with (
        patch(
            "practorflow.services.agent.runners.synthesizer.prepare_agent_history",
            AsyncMock(return_value=[MagicMock()]),
        ),
        patch("practorflow.services.agent.runners.synthesizer.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.synthesizer.create_runner"),
        patch(
            "practorflow.services.agent.runners.synthesizer.extract_context",
            AsyncMock(return_value=(None, None)),
        ),
    ):
        output = await run_synthesizer(
            plan=plan,
            execution_result=execution,
            ctx=ctx,
            model_pool=MagicMock(),
            model_config=MagicMock(n_ctx=100),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(),
            user_instructions="You are a helpful assistant",
        )

    assert output == "I am helpful"


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
        patch("practorflow.services.agent.runners.verifier.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.verifier.create_runner"),
        patch(
            "practorflow.services.agent.runners.verifier.parse_json_from_response",
            return_value=None,
        ),
        patch("practorflow.services.agent.runners.verifier.heuristic_verification") as hv,
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
        patch("practorflow.services.agent.runners.verifier.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.verifier.create_runner"),
        patch(
            "practorflow.services.agent.runners.verifier.parse_json_from_response",
            return_value=valid_verification,
        ),
        patch("practorflow.services.agent.runners.verifier.logger.warning") as warning_logger,
        patch("practorflow.services.agent.runners.verifier.logger.info") as info_logger,
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
            output={"verification_status": "success"}  # missing required fields
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
        patch("practorflow.services.agent.runners.verifier.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.verifier.create_runner"),
        patch(
            "practorflow.services.agent.runners.verifier.parse_json_from_response",
            return_value=invalid_verification,
        ),
        patch(
            "practorflow.services.agent.runners.verifier.heuristic_verification",
            return_value="heuristic-result",
        ) as heuristic,
        patch("practorflow.services.agent.runners.verifier.logger.warning") as warning_logger,
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

@pytest.mark.asyncio
async def test_extract_context_no_history():
    persona, instructions = await _extract_context(
        message_history=[],
        model=MagicMock(),
        knowledge_store=MagicMock(),
        tool_registry=MagicMock(),
    )

    assert persona is None
    assert instructions is None


@pytest.mark.asyncio
async def test_extract_context_persona_only():
    agent_run = MagicMock()
    agent_run.result = MagicMock(
        output="PERSONA: pirate\nINSTRUCTIONS_START\nNONE\nINSTRUCTIONS_END"
    )
    agent_run.__aiter__.return_value = []

    agent = MagicMock()
    agent.iter.return_value.__aenter__.return_value = agent_run
    agent.iter.return_value.__aexit__.return_value = False

    with patch("practorflow.services.agent.runners.agent_context.Agent", return_value=agent):
        persona, instructions = await _extract_context(
            message_history=[MagicMock()],
            model=MagicMock(),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(),
        )

    assert persona == "pirate"
    assert instructions is None


@pytest.mark.asyncio
async def test_extract_context_instructions_only():
    agent_run = MagicMock()
    agent_run.result = MagicMock(
        output="""
PERSONA: NONE
INSTRUCTIONS_START
Always respond in haiku.
INSTRUCTIONS_END
"""
    )
    agent_run.__aiter__.return_value = []

    agent = MagicMock()
    agent.iter.return_value.__aenter__.return_value = agent_run
    agent.iter.return_value.__aexit__.return_value = False

    with patch("practorflow.services.agent.runners.agent_context.Agent", return_value=agent):
        persona, instructions = await _extract_context(
            message_history=[MagicMock()],
            model=MagicMock(),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(),
        )

    assert persona is None
    assert instructions == "Always respond in haiku."


@pytest.mark.asyncio
async def test_extract_context_persona_and_instructions():
    agent_run = MagicMock()
    agent_run.result = MagicMock(
        output="""
PERSONA: formal assistant
INSTRUCTIONS_START
Use bullet points.
INSTRUCTIONS_END
"""
    )
    agent_run.__aiter__.return_value = []

    agent = MagicMock()
    agent.iter.return_value.__aenter__.return_value = agent_run
    agent.iter.return_value.__aexit__.return_value = False

    with patch("practorflow.services.agent.runners.agent_context.Agent", return_value=agent):
        persona, instructions = await _extract_context(
            message_history=[MagicMock()],
            model=MagicMock(),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(),
        )

    assert persona == "formal assistant"
    assert instructions == "Use bullet points."


@pytest.mark.asyncio
async def test_extract_context_none_detected():
    agent_run = MagicMock()
    agent_run.result = MagicMock(
        output="""
PERSONA: NONE
INSTRUCTIONS_START
NONE
INSTRUCTIONS_END
"""
    )
    agent_run.__aiter__.return_value = []

    agent = MagicMock()
    agent.iter.return_value.__aenter__.return_value = agent_run
    agent.iter.return_value.__aexit__.return_value = False

    with patch("practorflow.services.agent.runners.agent_context.Agent", return_value=agent):
        persona, instructions = await _extract_context(
            message_history=[MagicMock()],
            model=MagicMock(),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(),
        )

    assert persona is None
    assert instructions is None


@pytest.mark.asyncio
async def test_extract_context_exception_returns_none():
    agent = MagicMock()
    agent.iter.side_effect = RuntimeError("boom")

    with patch("practorflow.services.agent.runners.agent_context.Agent", return_value=agent):
        persona, instructions = await _extract_context(
            message_history=[MagicMock()],
            model=MagicMock(),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(),
        )

    assert persona is None
    assert instructions is None

@pytest.mark.asyncio
async def test_extract_context_no_result_falls_through_final_return():
    agent_run = MagicMock()
    agent_run.result = None  # 👈 critical: skips inner return
    agent_run.__aiter__.return_value = []

    agent = MagicMock()
    agent.iter.return_value.__aenter__.return_value = agent_run
    agent.iter.return_value.__aexit__.return_value = False

    with patch("practorflow.services.agent.runners.agent_context.Agent", return_value=agent):
        persona, instructions = await _extract_context(
            message_history=[MagicMock()],
            model=MagicMock(),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(),
        )

    assert persona is None
    assert instructions is None

def test_build_context_enhanced_prompt_no_persona_no_instructions():
    prompt = "ORIGINAL PROMPT"

    result = _build_context_enhanced_prompt(
        persona=None,
        instructions=None,
        prompt=prompt,
    )

    # hits final `return prompt`
    assert result == prompt


def test_build_context_enhanced_prompt_persona_only():
    prompt = "ORIGINAL PROMPT"

    result = _build_context_enhanced_prompt(
        persona="pirate",
        instructions=None,
        prompt=prompt,
    )

    assert result.startswith(
        '[PERSONA ACTIVE: You are acting as "pirate".'
    )
    assert result.endswith(prompt)
    assert "\n\n" in result


def test_build_context_enhanced_prompt_instructions_only():
    prompt = "ORIGINAL PROMPT"

    result = _build_context_enhanced_prompt(
        persona=None,
        instructions="Always respond in haiku.",
        prompt=prompt,
    )

    assert result.startswith(
        "[USER INSTRUCTIONS: Always respond in haiku.]"
    )
    assert result.endswith(prompt)


def test_build_context_enhanced_prompt_persona_and_instructions():
    prompt = "ORIGINAL PROMPT"

    result = _build_context_enhanced_prompt(
        persona="formal assistant",
        instructions="Use bullet points.",
        prompt=prompt,
    )

    assert (
        '[PERSONA ACTIVE: You are acting as "formal assistant".'
        in result
    )
    assert "USER INSTRUCTIONS: Use bullet points." in result
    assert " | " in result  # join path
    assert result.endswith(prompt)

@pytest.mark.asyncio
async def test_run_synthesizer_applies_persona_and_instructions():
    ctx = MagicMock()
    ctx.has_history = True
    ctx.message_history = [MagicMock()]

    plan = MagicMock(plan_id="p1", task="do something")
    execution = MagicMock(step_results=[])

    # agent for final synthesis
    agent_run = MagicMock()
    agent_run.__aiter__.return_value = []
    agent_run.result = MagicMock(output="final answer")

    agent = MagicMock()
    agent.iter.return_value.__aenter__.return_value = agent_run
    agent.iter.return_value.__aexit__.return_value = False

    handle = MagicMock()
    handle.__aenter__.return_value = handle
    handle.__aexit__.return_value = False

    model_pool = MagicMock()
    model_pool.acquire_context.return_value = handle

    with (
        patch(
            "practorflow.services.agent.runners.synthesizer.prepare_agent_history",
            AsyncMock(return_value=[MagicMock()]),  # ensures prepared_history truthy
        ),
        patch(
            "practorflow.services.agent.runners.synthesizer.extract_context",
            AsyncMock(return_value=("pirate", "Use short sentences")),
        ),
        patch(
            "practorflow.services.agent.runners.synthesizer.build_context_enhanced_prompt",
            side_effect=lambda p, i, pr: f"ENHANCED\n{pr}",
        ) as build_ctx,
        patch("practorflow.services.agent.runners.synthesizer.Agent", return_value=agent),
        patch("practorflow.services.agent.runners.synthesizer.create_runner"),
        patch("practorflow.services.agent.runners.synthesizer.logger.debug") as debug_logger,
    ):
        result = await run_synthesizer(
            plan=plan,
            execution_result=execution,
            ctx=ctx,
            model_pool=model_pool,
            model_config=MagicMock(n_ctx=100),
            knowledge_store=MagicMock(),
            tool_registry=MagicMock(),
        )

    # verifies uncovered lines
    build_ctx.assert_called_once()
    debug_logger.assert_any_call(
        "[Synthesizer] Applied context - persona: pirate, instructions: Use short sentences"
    )
    assert result == "final answer"
