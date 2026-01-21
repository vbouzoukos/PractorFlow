from practorflow.services.agent.prompts import (
    _format_tools_for_prompt,
    build_planner_prompt,
    build_executor_prompt,
    build_synthesis_prompt,
    build_verifier_prompt,
)

from practorflow.services.agent.schemas import StepResult, StepStatus
from tests.practorflow.services.agent.common_agent_deps import (
    make_plan,
    make_execution_result,
)


def test_build_planner_prompt_with_docs_and_tools():
    prompt = build_planner_prompt(
        task="do x",
        tools_metadata=[
            {
                "function": {
                    "name": "tool1",
                    "description": "desc",
                    "parameters": {
                        "properties": {"a": {"type": "string"}},
                        "required": ["a"],
                    },
                }
            }
        ],
        document_context="docs here",
    )

    assert "<task>\ndo x\n</task>" in prompt
    assert "<documents>\ndocs here\n</documents>" in prompt
    assert "<available_tools>\nUse these in the \'tool\' and \'tool_args\' fields of your steps:\n\nTool: tool1\n  Description: desc\n  Parameters:\n    - a: string (required) - \n\n</available_tools>" in prompt


def test_build_planner_prompt_no_tools_no_docs():
    prompt = build_planner_prompt(
        task="do y",
        tools_metadata=[],
        document_context=None,
    )

    assert "<available_tools>None - use null for tool field in all steps</available_tools>" in prompt


def test_build_executor_prompt_with_tool_step():
    plan = make_plan()
    plan.steps[0].tool = "tool_a"
    plan.steps[0].tool_args = {"x": 1}

    prompt = build_executor_prompt(plan)

    assert plan.plan_id in prompt
    assert "Tool: tool_a" in prompt
    assert "Expected:" in prompt


def test_build_executor_prompt_reasoning_step():
    plan = make_plan()
    plan.steps[0].tool = None

    prompt = build_executor_prompt(plan)

    assert "Tool: None (use LLM knowledge)" in prompt


def test_build_verifier_prompt_includes_execution_results():
    plan = make_plan()
    execution = make_execution_result(plan)

    prompt = build_verifier_prompt(plan, execution)

    assert "VERIFICATION REQUEST" in prompt
    assert plan.plan_id in prompt
    assert execution.execution_log in prompt

def test_format_tools_for_prompt_no_tools():
    result = _format_tools_for_prompt([])
    assert result == "No tools available."
    
def test_build_executor_prompt_empty_tool_args():
    plan = make_plan()
    plan.steps[0].tool = "tool_x"
    plan.steps[0].tool_args = None

    prompt = build_executor_prompt(plan)

    assert "Tool: tool_x()" in prompt

def test_build_verifier_prompt_truncates_long_output():
    plan = make_plan()
    long_output = "x" * 600

    execution = make_execution_result(
        plan,
        step_results=[
            StepResult(
                step_id=plan.steps[0].step_id,
                status=StepStatus.SUCCESS,
                output=long_output,
                evidence=[],
                error=None,
            )
        ],
    )

    prompt = build_verifier_prompt(plan, execution)

    assert "... (truncated)" in prompt

def test_build_verifier_prompt_includes_evidence():
    plan = make_plan()

    execution = make_execution_result(
        plan,
        step_results=[
            StepResult(
                step_id=plan.steps[0].step_id,
                status=StepStatus.SUCCESS,
                output="ok",
                evidence=["tool:x"],
                error=None,
            )
        ],
    )

    prompt = build_verifier_prompt(plan, execution)

    assert "Evidence: tool:x" in prompt

def test_build_verifier_prompt_includes_error():
    plan = make_plan()

    execution = make_execution_result(
        plan,
        step_results=[
            StepResult(
                step_id=plan.steps[0].step_id,
                status=StepStatus.FAILURE,
                output=None,
                evidence=[],
                error="boom",
            )
        ],
    )

    prompt = build_verifier_prompt(plan, execution)

    assert "Error: boom" in prompt

def test_build_synthesis_prompt_skips_llm_reasoning_outputs():
    plan = make_plan()

    execution = make_execution_result(
        plan,
        step_results=[
            StepResult(
                step_id=plan.steps[0].step_id,
                status=StepStatus.SUCCESS,
                output="internal reasoning text",
                evidence=["llm_reasoning"],
                error=None,
            )
        ],
    )

    from practorflow.services.agent.prompts import build_synthesis_prompt

    prompt = build_synthesis_prompt(plan.task, execution)

    assert "internal reasoning text" not in prompt
    assert "No tool outputs available." in prompt

def test_build_synthesis_prompt_no_tool_outputs():
    plan = make_plan()

    execution = make_execution_result(
        plan,
        step_results=[
            StepResult(
                step_id=plan.steps[0].step_id,
                status=StepStatus.SUCCESS,
                output=None,
                evidence=[],
                error=None,
            )
        ],
    )

    prompt = build_synthesis_prompt(plan.task, execution)

    assert "No tool outputs available." in prompt

def test_build_synthesis_prompt_with_multiple_tool_outputs():
    plan = make_plan()

    execution = make_execution_result(
        plan,
        step_results=[
            StepResult(
                step_id="step_1",
                status=StepStatus.SUCCESS,
                output="output one",
                evidence=["tool:a"],
                error=None,
            ),
            StepResult(
                step_id="step_2",
                status=StepStatus.SUCCESS,
                output="output two",
                evidence=["tool:b"],
                error=None,
            ),
        ],
    )

    prompt = build_synthesis_prompt(plan.task, execution)

    assert "output one" in prompt
    assert "output two" in prompt
    assert "\n\n---\n\n" in prompt
