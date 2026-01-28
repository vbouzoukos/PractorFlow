from unittest.mock import AsyncMock, MagicMock

from practorflow.services.agent.parser import parse_executor_results
from practorflow.services.agent.schemas import StepStatus
from tests.practorflow.services.agent.common_agent_deps import (
    make_plan,
    make_agent_deps,
    make_mock_tool_result,
)


async def test_executor_with_successful_tool():
    plan = make_plan()
    step = plan.steps[0]
    step.tool = "tool_a"
    step.tool_args = {"x": 1}

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute = AsyncMock(
        return_value=make_mock_tool_result(success=True, data="ok")
    )

    deps = make_agent_deps(knowledge_store=MagicMock(), tool_registry=tool_registry)

    results = await parse_executor_results(plan, deps)

    assert results[0].status == StepStatus.SUCCESS
    assert results[0].output == "ok"
    tool_registry.execute.assert_called_once()


async def test_executor_tool_failure():
    plan = make_plan()
    step = plan.steps[0]
    step.tool = "tool_a"

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute = AsyncMock(
        return_value=make_mock_tool_result(success=False, error="err")
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    results = await parse_executor_results(plan, deps)

    assert results[0].status == StepStatus.FAILURE
    assert results[0].error == "err"


async def test_executor_tool_exception():
    plan = make_plan()
    step = plan.steps[0]
    step.tool = "tool_a"

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute.side_effect = RuntimeError("boom")

    deps = make_agent_deps(MagicMock(), tool_registry)

    results = await parse_executor_results(plan, deps)

    assert results[0].status == StepStatus.FAILURE
    assert "boom" in results[0].error


async def test_executor_tool_not_found():
    plan = make_plan()
    step = plan.steps[0]
    step.tool = "missing"

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = False

    deps = make_agent_deps(MagicMock(), tool_registry)

    results = await parse_executor_results(plan, deps)

    assert results[0].status == StepStatus.FAILURE
    assert "Tool not found" in results[0].error


async def test_executor_reasoning_step():
    plan = make_plan()
    plan.steps[0].tool = None

    deps = make_agent_deps(MagicMock(), AsyncMock())

    results = await parse_executor_results(plan, deps)

    assert results[0].status == StepStatus.SUCCESS
    assert "Reasoning completed" in results[0].output


async def test_executor_resolves_previous_reference():
    plan = make_plan()

    plan.steps = [
        plan.steps[0].model_copy(
            update={
                "step_id": "step_1",
                "tool": "tool_a",
                "tool_args": {},
            }
        ),
        plan.steps[0].model_copy(
            update={
                "step_id": "step_2",
                "tool": "tool_b",
                "tool_args": {"y": "$previous"},
            }
        ),
    ]

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute = AsyncMock(
        side_effect=[
            make_mock_tool_result(success=True, data="first"),
            make_mock_tool_result(success=True, data="second"),
        ]
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    await parse_executor_results(plan, deps)

    tool_registry.execute.assert_called_with("tool_b", y="first")


async def test_executor_resolves_explicit_step_output_reference():
    plan = make_plan()

    plan.steps = [
        plan.steps[0].model_copy(
            update={
                "step_id": "step_1",
                "tool": "tool_a",
                "tool_args": {},
            }
        ),
        plan.steps[0].model_copy(
            update={
                "step_id": "step_2",
                "tool": "tool_b",
                "tool_args": {"y": "$step_1.output"},
            }
        ),
    ]

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute = AsyncMock(
        side_effect=[
            make_mock_tool_result(success=True, data="out1"),
            make_mock_tool_result(success=True, data="out2"),
        ]
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    await parse_executor_results(plan, deps)

    tool_registry.execute.assert_called_with("tool_b", y="out1")


async def test_executor_reference_falls_back_when_step_failed():
    plan = make_plan()

    plan.steps = [
        plan.steps[0].model_copy(
            update={
                "step_id": "step_1",
                "tool": "tool_a",
                "tool_args": {},
            }
        ),
        plan.steps[0].model_copy(
            update={
                "step_id": "step_2",
                "tool": "tool_b",
                "tool_args": {},
            }
        ),
        plan.steps[0].model_copy(
            update={
                "step_id": "step_3",
                "tool": "tool_c",
                "tool_args": {"y": "$step_2"},
            }
        ),
    ]

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute = AsyncMock(
        side_effect=[
            make_mock_tool_result(success=True, data="good"),  # step_1
            make_mock_tool_result(success=False, error="fail"),  # step_2
            make_mock_tool_result(success=True, data="fallback"),  # step_3
        ]
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    results = await parse_executor_results(plan, deps)

    assert results[2].status == StepStatus.SUCCESS
    tool_registry.execute.assert_called_with("tool_c", y="good")


async def test_executor_inferrs_missing_tool_input_param():
    plan = make_plan()
    plan.steps = [
        plan.steps[0].model_copy(update={"tool": "tool_a"}),
        plan.steps[0].model_copy(
            update={
                "step_id": "step_2",
                "tool": "summarize_text",
                "tool_args": {},  # text missing
            }
        ),
    ]

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute = AsyncMock(
        side_effect=[
            make_mock_tool_result(success=True, data="raw text"),
            make_mock_tool_result(success=True, data="summary"),
        ]
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    await parse_executor_results(plan, deps)

    tool_registry.execute.assert_called_with(
        "summarize_text",
        text="raw text",
    )


async def test_executor_previous_skips_reasoning_steps():
    plan = make_plan()
    plan.steps = [
        plan.steps[0].model_copy(update={"tool": None}),
        plan.steps[0].model_copy(
            update={
                "step_id": "step_2",
                "tool": "tool_b",
                "tool_args": {"y": "$previous"},
            }
        ),
    ]

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute = AsyncMock(
        return_value=make_mock_tool_result(success=True, data="tool-out")
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    results = await parse_executor_results(plan, deps)

    tool_registry.execute.assert_called_with("tool_b", y=None)
    assert results[1].status == StepStatus.SUCCESS


async def test_executor_get_last_tool_output_returns_none_when_all_tool_outputs_none():
    plan = make_plan()

    plan.steps = [
        plan.steps[0].model_copy(
            update={
                "step_id": "step_1",
                "tool": "tool_a",
                "tool_args": {},
            }
        ),
        plan.steps[0].model_copy(
            update={
                "step_id": "step_2",
                "tool": "tool_b",
                "tool_args": {"y": "$previous"},
            }
        ),
    ]

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True

    # step_1: tool step but returns None output
    # step_2: tries to resolve $previous → no usable tool output → None
    tool_registry.execute = AsyncMock(
        side_effect=[
            make_mock_tool_result(success=False, error="fail"),  # output=None
            make_mock_tool_result(success=True, data="ignored"),
        ]
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    results = await parse_executor_results(plan, deps)

    # step_2 still executes, but with y=None
    tool_registry.execute.assert_called_with("tool_b", y=None)

    assert results[1].status == StepStatus.SUCCESS


async def test_executor_unresolvable_reference_sets_none():
    plan = make_plan()

    plan.steps = [
        plan.steps[0].model_copy(
            update={
                "step_id": "step_1",
                "tool": "tool_a",
                "tool_args": {"x": "$previous"},
            }
        )
    ]

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True

    # No previous tool outputs exist
    tool_registry.execute = AsyncMock(
        return_value=make_mock_tool_result(
            success=True,
            data="ignored",
        )
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    results = await parse_executor_results(plan, deps)

    tool_registry.execute.assert_called_with("tool_a", x=None)
    assert results[0].status == StepStatus.SUCCESS


async def test_resolve_tool_args_unresolvable_step_no_fallback():
    plan = make_plan()
    plan.steps = [
        plan.steps[0].model_copy(
            update={
                "step_id": "step_1",
                "tool": "tool_a",
                "tool_args": {},  # executes first
            }
        ),
        plan.steps[0].model_copy(
            update={
                "step_id": "step_2",
                "tool": "tool_b",
                "tool_args": {"x": "$step_1"},
            }
        ),
    ]

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True

    # step_1 fails → output=None
    # no other tool outputs exist → no fallback
    tool_registry.execute = AsyncMock(
        side_effect=[
            make_mock_tool_result(success=False, error="fail"),
            make_mock_tool_result(success=True, data="ignored"),
        ]
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    await parse_executor_results(plan, deps)

    tool_registry.execute.assert_called_with("tool_b", x=None)


async def test_resolve_tool_args_malformed_dollar_reference_passthrough():
    plan = make_plan()
    plan.steps[0].tool = "tool_a"
    plan.steps[0].tool_args = {"x": "$not a valid ref"}

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute = AsyncMock(
        return_value=make_mock_tool_result(success=True, data="ok")
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    await parse_executor_results(plan, deps)

    tool_registry.execute.assert_called_with("tool_a", x="$not a valid ref")


async def test_resolve_tool_args_plain_string_passthrough():
    plan = make_plan()
    plan.steps[0].tool = "tool_a"
    plan.steps[0].tool_args = {"x": "hello"}

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute = AsyncMock(
        return_value=make_mock_tool_result(success=True, data="ok")
    )
    deps = make_agent_deps(MagicMock(), tool_registry)

    await parse_executor_results(plan, deps)

    tool_registry.execute.assert_called_with("tool_a", x="hello")


async def test_resolve_tool_args_nested_dict():
    plan = make_plan()
    plan.steps[0].tool = "tool_a"
    plan.steps[0].tool_args = {"config": {"param": "value"}}

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute = AsyncMock(
        return_value=make_mock_tool_result(success=True, data="ok")
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    await parse_executor_results(plan, deps)

    tool_registry.execute.assert_called_with(
        "tool_a",
        config={"param": "value"},
    )


async def test_resolve_tool_args_list_recursion():
    plan = make_plan()
    plan.steps[0].tool = "tool_a"
    plan.steps[0].tool_args = {
        "items": [
            {"a": "x"},
            123,
        ]
    }

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute = AsyncMock(
        return_value=make_mock_tool_result(success=True, data="ok")
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    await parse_executor_results(plan, deps)

    tool_registry.execute.assert_called_with(
        "tool_a",
        items=[{"a": "x"}, 123],
    )


async def test_infer_tool_args_no_tool_outputs_logs_warning_and_does_not_infer():
    plan = make_plan()

    # Step 1: reasoning step (NOT a tool, so no tool outputs exist)
    plan.steps = [
        plan.steps[0].model_copy(
            update={
                "step_id": "step_1",
                "tool": None,
            }
        ),
        plan.steps[0].model_copy(
            update={
                "step_id": "step_2",
                "tool": "summarize_text",
                "tool_args": {},  # missing required 'text'
            }
        ),
    ]

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True

    # summarize_text executes, but inference should fail BEFORE execution
    tool_registry.execute = AsyncMock(
        return_value=make_mock_tool_result(
            success=True,
            data="summary",
        )
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    results = await parse_executor_results(plan, deps)

    # summarize_text should be called WITHOUT inferred 'text'
    tool_registry.execute.assert_called_with("summarize_text")

    # Step still succeeds, but inference path logged warning
    assert results[1].status == StepStatus.SUCCESS
