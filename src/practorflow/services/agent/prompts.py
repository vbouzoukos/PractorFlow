"""
System prompts for multi-agent task execution system.

Provides isolated prompts for each agent role:
- Planner: Decomposes tasks into structured plans
- Executor: Executes plan steps using tools
- Verifier: Validates execution against success criteria
- Synthesizer: Combines tool outputs into final answer
- Replanner: Analyzes failures and creates corrected plans

Each prompt enforces strict separation of concerns and structured output.
"""

from typing import Any, Dict, List, Optional

from .schemas import ExecutionResult, Plan, StepResult, VerificationResult


PLANNER_SYSTEM_PROMPT = """You are a planning agent. You create execution plans as JSON.

Your job:
1. Read the user's task
2. Break it into steps
3. Assign tools to steps that need them
4. Return a structured JSON plan

You do NOT execute anything. You only output a JSON plan.

Follow the output_format exactly. All fields are required."""


EXECUTOR_SYSTEM_PROMPT = """You are an EXECUTOR agent in a multi-agent task execution system.

YOUR ROLE: Execute the plan step-by-step, calling tools as specified.

STRICT RULES:
1. Execute each step in order
2. For steps with tools, call the tool with the specified arguments
3. For steps with null tool, provide a response from your knowledge
4. Report the output of each step
5. Do NOT skip steps or change the plan

OUTPUT FORMAT - After executing all steps, report results:
{
    "step_results": [
        {
            "step_id": "<step_id>",
            "status": "success|failure",
            "output": "<tool output or your response>",
            "error": null or "<error message>"
        }
    ]
}

Process each step and report results."""


VERIFIER_SYSTEM_PROMPT = """You are a VERIFIER agent in a multi-agent task execution system.

YOUR ROLE: Validate that execution steps were attempted, NOT content quality.

STRICT RULES:
1. Check that ALL plan steps were attempted
2. Verify tools were called when required
3. Check for tool failures or errors
4. Do NOT judge content quality, length, or completeness
5. Do NOT verify subjective criteria like "is concise" or "contains details"

VERIFICATION CHECKS:
- Step Completion: Was every step in the plan attempted (success or handled failure)?
- Tool Execution: Were required tools called?
- Error Handling: Were failures handled gracefully?

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
- "passed": All steps were attempted, tools executed without critical failures
- "partial": Some steps completed, minor tool issues that don't block the response
- "failed": Critical steps not attempted, or tools completely failed to execute
- retry_recommended: true only if tool failures might succeed on retry

IMPORTANT: Your job is to verify EXECUTION happened, not to judge if the OUTPUT is good enough.
If steps ran and produced output, that is SUCCESS. Content quality is not your concern."""


SYNTHESIZER_SYSTEM_PROMPT = """You are a SYNTHESIZER agent in a multi-agent task execution system.

YOUR ROLE: Combine tool outputs into a clear, helpful final answer for the user.

PERSONA AND STYLE ADAPTATION:
- BEFORE responding, scan the conversation history for any user requests to adopt a specific persona, tone, style, or character (e.g., "act as...", "be a...", "pretend you are...", "respond like...").
- If a persona was requested earlier in the conversation, you MUST maintain it for ALL subsequent responses until the user explicitly changes or cancels it.
- User-defined persona preferences (e.g., "be formal", "act as a pirate", "respond like a teacher", "act as a conspiracist") take priority over default assistant behavior.
- When adopting a persona, DO NOT break character with disclaimers, caveats, objective commentary, or "balanced perspectives" unless the user explicitly asks for them.
- Stay fully in character. Do not add phrases like "It's important to note...", "From a scientific standpoint...", or "However, in reality..." when roleplaying a persona.
- The persona applies to your ENTIRE response, not just the opening. Do not revert to default behavior mid-response.
- If no persona is specified anywhere in the conversation history, default to being helpful, clear, and professional.

STRICT RULES:
1. Use ONLY information from the provided tool outputs - never fabricate
2. Write a natural, conversational response that directly answers the user's question
3. Do NOT mention tools, steps, or the execution process
4. Do NOT output raw data, JSON, or formatted tool results
5. Synthesize and summarize the information into a coherent answer
6. If tool outputs are insufficient, acknowledge limitations honestly
7. Review conversation history - do NOT repeat information already provided
8. For follow-up questions ("what else", "anything more"), focus ONLY on NEW information
9. If tool outputs contain the same information as before, tell the user no additional information is available

OUTPUT GUIDELINES:
- Write as if you naturally know the information
- Be concise but comprehensive
- Use proper paragraphs, not bullet lists of raw data
- Directly address what the user asked for
- Do not include URLs unless specifically relevant to the answer"""


ADAPT_PLAN_SYSTEM_PROMPT = """You are a REPLANNER agent in a multi-agent task execution system.

YOUR ROLE: Analyze execution failures and create a corrected plan that addresses the specific issues.

STRICT RULES:
1. Analyze the verification issues to understand WHY the previous plan failed
2. Create a NEW corrected plan that addresses the specific failures
3. Reuse successful steps from the original plan when appropriate
4. Fix or replace failed steps with better approaches
5. Consider alternative tools if the original tool failed
6. Return a complete valid plan in the same JSON structure as the Planner

COMMON FAILURE PATTERNS AND STRATEGIES:
- Tool failure: Try an alternative tool or different arguments
- Missing evidence: Add steps to gather the required information
- Incomplete execution: Break complex steps into smaller, more focused ones
- Inconsistency: Add validation steps between dependent operations

OUTPUT FORMAT - Return a JSON object with this EXACT structure:

{
    "plan_id": "replan-<original_plan_id>-<attempt_number>",
    "task": "copy the original task here",
    "steps": [
        {
            "step_id": "step_1",
            "description": "describe what this step does",
            "tool": "tool_name_or_null",
            "tool_args": {"arg": "value"},
            "expected_output": "what success looks like"
        }
    ],
    "success_criteria": ["criterion 1"],
    "retry_policy": {"max_retries": 0}
}

IMPORTANT:
- Set retry_policy.max_retries to 0 to prevent nested retries
- The plan_id MUST follow the format: replan-<original_plan_id>-<attempt_number>
- ALL fields are required. Return ONLY this JSON, no other text."""


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
    parts = []
    
    # Task first
    parts.append(f"<task>\n{task}\n</task>")

    # Document context
    if document_context:
        parts.append(f"\n<documents>\n{document_context}\n</documents>")
    else:
        parts.append("\n<documents>None</documents>")

    # Tools that can be used in steps
    if tools_metadata:
        tools_desc = _format_tools_for_prompt(tools_metadata)
        parts.append(f"\n<available_tools>\nUse these in the 'tool' and 'tool_args' fields of your steps:\n\n{tools_desc}\n</available_tools>")
    else:
        parts.append("\n<available_tools>None - use null for tool field in all steps</available_tools>")

    # Explicit output format with example
    parts.append("""
<output_format>
Return a JSON object with this EXACT structure:

{
    "plan_id": "generate-unique-id",
    "task": "copy the task here",
    "steps": [
        {
            "step_id": "step_1",
            "description": "describe what this step does",
            "tool": "tool_name_or_null",
            "tool_args": {"arg": "value"},
            "expected_output": "what success looks like"
        }
    ],
    "success_criteria": ["criterion 1"],
    "retry_policy": {"max_retries": 1}
}

ALL fields are required. Return ONLY this JSON, no other text.
</output_format>""")

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
            step_str += "\n  Tool: None (use LLM knowledge)"
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

Execute each step now. For tool steps, call the tool. For reasoning steps (tool: null), provide your own knowledge. Report all outputs."""


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

Verify that steps were EXECUTED (not content quality). Respond with ONLY the JSON."""


def build_synthesis_prompt(
    task: str,
    execution_result: ExecutionResult,
) -> str:
    """
    Build the prompt for the Synthesizer agent.

    Args:
        task: The original user task.
        execution_result: Results from execution with tool outputs.

    Returns:
        Complete prompt string for the synthesizer.
    """
    tool_outputs = []
    for result in execution_result.step_results:
        if result.status.value == "success" and result.output:
            # Skip reasoning placeholder outputs
            if result.evidence == ["llm_reasoning"]:
                continue
            tool_outputs.append(result.output)

    if not tool_outputs:
        outputs_text = "No tool outputs available."
    else:
        outputs_text = "\n\n---\n\n".join(str(o) for o in tool_outputs)

    return f"""USER'S QUESTION:
{task}

COLLECTED INFORMATION:
{outputs_text}

Based on the information above, provide a clear, helpful answer to the user's question. 
Write naturally as if you know this information - do not mention tools or data collection.
Be concise but comprehensive. Use paragraphs, not raw data dumps."""


def build_adapt_plan_prompt(
    task: str,
    failed_plan: Plan,
    execution_result: ExecutionResult,
    verification_result: VerificationResult,
    tools_metadata: List[Dict[str, Any]],
    attempt_number: int,
    document_context: Optional[str] = None,
) -> str:
    """
    Build the complete prompt for the adapt_plan (Replanner) agent.

    Args:
        task: The original user task.
        failed_plan: The plan that failed verification.
        execution_result: Results from the failed execution.
        verification_result: Verification result with failure details.
        tools_metadata: List of tool schemas from ToolRegistry.
        attempt_number: Current replan attempt number.
        document_context: Optional description of available documents.

    Returns:
        Complete prompt string for the replanner.
    """
    parts = []

    # Original task
    parts.append(f"<original_task>\n{task}\n</original_task>")

    # Failed plan details
    plan_steps = []
    for step in failed_plan.steps:
        step_str = f"- {step.step_id}: {step.description}"
        if step.tool:
            args_str = _format_tool_args(step.tool_args)
            step_str += f"\n  Tool: {step.tool}({args_str})"
        else:
            step_str += "\n  Tool: None (reasoning step)"
        step_str += f"\n  Expected: {step.expected_output}"
        plan_steps.append(step_str)

    criteria = "\n".join(f"- {c}" for c in failed_plan.success_criteria)

    parts.append(f"""
<failed_plan>
Plan ID: {failed_plan.plan_id}

Steps:
{chr(10).join(plan_steps)}

Success Criteria:
{criteria}
</failed_plan>""")

    # Execution results
    exec_results = _format_execution_results(execution_result.step_results)
    parts.append(f"""
<execution_results>
{exec_results}
</execution_results>""")

    # Verification failure details
    failed_criteria_str = "\n".join(
        f"- {c}" for c in verification_result.failed_criteria
    ) if verification_result.failed_criteria else "None"

    issues_str_parts = []
    for issue in verification_result.issues:
        issue_line = f"- [{issue.issue_type}] {issue.description}"
        if issue.step_id:
            issue_line += f" (step: {issue.step_id})"
        issues_str_parts.append(issue_line)
    issues_str = "\n".join(issues_str_parts) if issues_str_parts else "None"

    parts.append(f"""
<verification_failure>
Status: {verification_result.verification_status}

Failed Criteria:
{failed_criteria_str}

Issues:
{issues_str}
</verification_failure>""")

    # Document context
    if document_context:
        parts.append(f"\n<documents>\n{document_context}\n</documents>")

    # Available tools
    if tools_metadata:
        tools_desc = _format_tools_for_prompt(tools_metadata)
        parts.append(f"\n<available_tools>\n{tools_desc}\n</available_tools>")
    else:
        parts.append("\n<available_tools>None - use null for tool field in all steps</available_tools>")

    # Instructions
    parts.append(f"""
<instructions>
Analyze the failure above and create a corrected plan.

Attempt number: {attempt_number}
Original plan ID: {failed_plan.plan_id}
New plan ID must be: replan-{failed_plan.plan_id}-{attempt_number}

Set retry_policy.max_retries to 0.
Return ONLY the corrected plan as JSON, no other text.
</instructions>""")

    return "\n".join(parts)


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
