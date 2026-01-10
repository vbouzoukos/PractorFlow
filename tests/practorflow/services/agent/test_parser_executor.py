# tests/test_parser_executor.py
from unittest.mock import MagicMock

from practorflow.services.agent.parser import parse_executor_results
from practorflow.services.agent.schemas import StepStatus
from tests.practorflow.services.agent.common_agent_deps import make_plan, make_agent_deps, make_mock_tool_result


def test_executor_with_successful_tool():
    plan = make_plan()
    step = plan.steps[0]
    step.tool = "tool_a"
    step.tool_args = {"x": 1}

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute.return_value = make_mock_tool_result(success=True, data="ok")

    deps = make_agent_deps(knowledge_store=MagicMock(), tool_registry=tool_registry)

    results = parse_executor_results(plan, "response", deps)

    assert results[0].status == StepStatus.SUCCESS
    assert results[0].output == "ok"
    tool_registry.execute.assert_called_once()


def test_executor_tool_failure():
    plan = make_plan()
    step = plan.steps[0]
    step.tool = "tool_a"

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute.return_value = make_mock_tool_result(success=False, error="err")

    deps = make_agent_deps(MagicMock(), tool_registry)

    results = parse_executor_results(plan, "", deps)

    assert results[0].status == StepStatus.FAILURE
    assert results[0].error == "err"


def test_executor_tool_exception():
    plan = make_plan()
    step = plan.steps[0]
    step.tool = "tool_a"

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute.side_effect = RuntimeError("boom")

    deps = make_agent_deps(MagicMock(), tool_registry)

    results = parse_executor_results(plan, "", deps)

    assert results[0].status == StepStatus.FAILURE
    assert "boom" in results[0].error


def test_executor_tool_not_found():
    plan = make_plan()
    step = plan.steps[0]
    step.tool = "missing"

    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = False

    deps = make_agent_deps(MagicMock(), tool_registry)

    results = parse_executor_results(plan, "", deps)

    assert results[0].status == StepStatus.FAILURE
    assert "Tool not found" in results[0].error


def test_executor_reasoning_step():
    plan = make_plan()
    plan.steps[0].tool = None

    deps = make_agent_deps(MagicMock(), MagicMock())

    results = parse_executor_results(plan, "do something", deps)

    assert results[0].status == StepStatus.SUCCESS
    assert "Reasoning completed" in results[0].output
