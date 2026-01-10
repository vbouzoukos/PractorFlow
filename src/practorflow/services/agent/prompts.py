"""
System prompts for multi-agent task execution system.

Provides isolated prompts for each agent role:
- Planner: Decomposes tasks into structured plans
- Executor: Executes plan steps using tools
- Verifier: Validates execution against success criteria

Each prompt enforces strict separation of concerns and structured output.
"""

from typing import Any, Dict, List, Optional

from .schemas import ExecutionResult, Plan, StepResult


PLANNER_SYSTEM_PROMPT = """You are a PLANNER agent in a multi-agent task execution system.

YOUR ROLE: Create a structured execution plan. You do NOT execute anything.

STRICT RULES:
1. Decompose the user's task into atomic, ordered steps
2. Assign tools to steps that require them (use exact tool names provided)
3. Use null for tool when the step is reasoning or synthesis only
4. Define explicit, machine-checkable success criteria
5. NEVER execute tools or provide final answers - planning only
6. If the task cannot be completed with available tools, set steps to explain why

OUTPUT FORMAT - Respond with ONLY this JSON structure, no other text:
{
    "plan_id": "<unique-uuid>",
    "task": "<original user task>",
    "steps": [
        {
            "step_id": "step_1",
            "description": "<what this step does>",
            "tool": "<tool_name or null>",
            "tool_args": {"<arg_name>": "<arg_value>"} or null,
            "expected_output": "<what success looks like>"
        }
    ],
    "success_criteria": [
        "<criterion 1: specific, verifiable condition>",
        "<criterion 2: specific, verifiable condition>"
    ],
    "retry_policy": {
        "max_retries": 1
    }
}

PLANNING GUIDELINES:
- Each step should be atomic (one action/decision)
- Steps must be ordered by dependency
- Tool arguments must match the tool's expected parameters
- Success criteria must be objectively verifiable from step outputs
- Include a final synthesis step if the task requires combining results"""


EXECUTOR_SYSTEM_PROMPT = """You are an EXECUTOR agent in a multi-agent task execution system.

YOUR ROLE: Execute plan steps strictly in order using available tools.

STRICT RULES:
1. Execute steps in EXACT order - never skip or reorder
2. For steps with tools: call the tool with specified arguments
3. For steps without tools: provide reasoning based on previous outputs
4. Record ALL outputs as evidence - never fabricate or hallucinate
5. If a tool fails, record the error and continue to next step
6. Do NOT interpret or summarize results beyond what tools return

EXECUTION PROTOCOL:
- When a step requires a tool, call it with the specified arguments
- Capture the exact tool output as evidence
- If tool_args are provided in the plan, use them exactly
- After all steps complete, provide the final synthesis

ERROR HANDLING:
- Tool errors should be recorded, not hidden
- Continue execution even if a step fails
- Mark failed steps clearly in your response

You will receive the plan to execute. Process each step and report results."""


VERIFIER_SYSTEM_PROMPT = """You are a VERIFIER agent in a multi-agent task execution system.

YOUR ROLE: Validate that execution results satisfy the plan's success criteria.

STRICT RULES:
1. Check that ALL plan steps were executed
2. Verify each success criterion has supporting evidence from step outputs
3. Detect contradictions between step outputs
4. Identify gaps where claims lack evidence
5. ZERO TOLERANCE for unverified or unsupported assertions
6. Do NOT add information - only verify what was executed

VERIFICATION CHECKS:
- Step Completion: Was every step in the plan executed?
- Evidence Mapping: Does each criterion map to specific step output?
- Consistency: Do step outputs contradict each other?
- Completeness: Are there gaps in the execution chain?
- Tool Failures: Did any tool failures impact the results?

OUTPUT FORMAT - Respond with ONLY this JSON structure, no other text:
{
    "verification_status": "passed|failed|partial",
    "failed_criteria": [
        "<criterion that was not satisfied>"
    ],
    "issues": [
        {
            "issue_type": "missing_evidence|inconsistency|incomplete_execution|tool_failure",
            "description": "<specific issue description>",
            "step_id": "<related step or null>"
        }
    ],
    "retry_recommended": true|false
}

DECISION GUIDELINES:
- "passed": ALL criteria satisfied with evidence, no issues
- "partial": SOME criteria satisfied, minor issues that don't invalidate results
- "failed": Critical criteria not met or major issues found
- retry_recommended: true if issues are transient (tool failures), false if fundamental"""


def build_planner_prompt(
    task: str,
    tools_metadata: List[Dict[str, Any]],
    document_context: Optional[str] = None,
) -> str:
    """
    Build the complete prompt for the Planner agent.

    Args:
        task: The user's task to plan.
        tools_metadata: List of tool schemas from ToolRegistry.
        document_context: Optional description of available documents.

    Returns:
        Complete prompt string for the planner.
    """
    parts = [f"TASK TO PLAN:\n{task}"]

    if document_context:
        parts.append(f"\nAVAILABLE DOCUMENTS:\n{document_context}")

    if tools_metadata:
        tools_desc = _format_tools_for_prompt(tools_metadata)
        parts.append(f"\nAVAILABLE TOOLS:\n{tools_desc}")
    else:
        parts.append("\nAVAILABLE TOOLS: None (reasoning only)")

    parts.append("\nCreate the execution plan now. Respond with ONLY the JSON.")

    return "\n".join(parts)


def build_executor_prompt(plan: Plan) -> str:
    """
    Build the complete prompt for the Executor agent.

    Args:
        plan: The plan to execute.

    Returns:
        Complete prompt string for the executor.
    """
    steps_desc = []
    for step in plan.steps:
        step_str = f"- {step.step_id}: {step.description}"
        if step.tool:
            args_str = _format_tool_args(step.tool_args)
            step_str += f"\n  Tool: {step.tool}({args_str})"
        else:
            step_str += "\n  Tool: None (reasoning step)"
        step_str += f"\n  Expected: {step.expected_output}"
        steps_desc.append(step_str)

    criteria_desc = "\n".join(f"- {c}" for c in plan.success_criteria)

    return f"""PLAN TO EXECUTE:
Plan ID: {plan.plan_id}
Task: {plan.task}

STEPS (execute in order):
{chr(10).join(steps_desc)}

SUCCESS CRITERIA:
{criteria_desc}

Execute each step now. Call tools as specified and report all outputs."""


def build_verifier_prompt(
    plan: Plan,
    execution_result: ExecutionResult,
) -> str:
    """
    Build the complete prompt for the Verifier agent.

    Args:
        plan: The original plan.
        execution_result: Results from execution.

    Returns:
        Complete prompt string for the verifier.
    """
    plan_steps = []
    for step in plan.steps:
        plan_steps.append(f"- {step.step_id}: {step.description}")
        plan_steps.append(f"  Expected: {step.expected_output}")

    criteria = "\n".join(f"- {c}" for c in plan.success_criteria)

    exec_results = _format_execution_results(execution_result.step_results)

    return f"""VERIFICATION REQUEST

ORIGINAL PLAN:
Plan ID: {plan.plan_id}
Task: {plan.task}

Planned Steps:
{chr(10).join(plan_steps)}

Success Criteria:
{criteria}

EXECUTION RESULTS:
{exec_results}

EXECUTION LOG:
{execution_result.execution_log}

Verify the execution against the plan and criteria. Respond with ONLY the JSON."""


def _format_tools_for_prompt(tools_metadata: List[Dict[str, Any]]) -> str:
    """Format tool schemas for inclusion in prompts."""
    if not tools_metadata:
        return "No tools available."

    lines = []
    for tool in tools_metadata:
        func = tool.get("function", {})
        name = func.get("name", "unknown")
        desc = func.get("description", "No description")

        params = func.get("parameters", {})
        properties = params.get("properties", {})
        required = params.get("required", [])

        params_desc = []
        for param_name, param_info in properties.items():
            param_type = param_info.get("type", "any")
            param_desc = param_info.get("description", "")
            req_marker = " (required)" if param_name in required else ""
            params_desc.append(f"    - {param_name}: {param_type}{req_marker} - {param_desc}")

        params_str = "\n".join(params_desc) if params_desc else "    (no parameters)"

        lines.append(f"Tool: {name}")
        lines.append(f"  Description: {desc}")
        lines.append(f"  Parameters:\n{params_str}")
        lines.append("")

    return "\n".join(lines)


def _format_tool_args(tool_args: Optional[Dict[str, Any]]) -> str:
    """Format tool arguments for display."""
    if not tool_args:
        return ""
    return ", ".join(f"{k}={repr(v)}" for k, v in tool_args.items())


def _format_execution_results(step_results: List[StepResult]) -> str:
    """Format step results for the verifier."""
    lines = []
    for result in step_results:
        lines.append(f"Step: {result.step_id}")
        lines.append(f"  Status: {result.status}")

        if result.output is not None:
            output_str = str(result.output)
            if len(output_str) > 500:
                output_str = output_str[:500] + "... (truncated)"
            lines.append(f"  Output: {output_str}")

        if result.evidence:
            lines.append(f"  Evidence: {', '.join(result.evidence)}")

        if result.error:
            lines.append(f"  Error: {result.error}")

        lines.append("")

    return "\n".join(lines)