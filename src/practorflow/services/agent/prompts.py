"""
System prompts for multi-agent task execution system.

Provides isolated prompts for each agent role:
- Planner: Decomposes tasks into structured plans
- Executor: Executes plan steps using tools
- Verifier: Validates execution against success criteria
- Synthesizer: Combines tool outputs into final answer

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
4. Define success criteria based on EXECUTION only, not content quality
5. NEVER execute tools or provide final answers - planning only
6. If the task cannot be completed with available tools, set steps to explain why

CRITICAL - TOOL SELECTION:
- If the user explicitly asks to use a tool's functionality, USE that tool regardless of other conditions.
- knowledge_search: Use when documents are listed in "AVAILABLE DOCUMENTS", or when user asks to search their documents/files.
- web_search: Use for current information, news, facts, or when user asks to "search the web", "look up", "find online".
- web_fetch: Use when user provides a URL and wants to read/fetch/get content from it.
- summarize_text: Use when user explicitly asks to summarize (e.g., "summarize this", "give me a summary", "TLDR").
- json_transform: Use when user asks to extract, transform, or parse JSON data.
- calculator: Use when user asks to calculate, compute, or do math operations.
- For general knowledge questions without explicit tool requests: Use reasoning steps (tool: null) to use LLM's own knowledge.

FOLLOW-UP QUESTION HANDLING:
- Review conversation history to understand what has already been discussed.
- If the user asks a follow-up (e.g., "what else", "anything more", "besides that"), they want NEW information.
- Adjust search queries to target different aspects or use different keywords to find new content.
- Avoid planning searches that will return the same information already discussed.
- Review the conversation history to understand what has already been discussed.
- If the user asks a follow-up question (e.g., "what else", "anything more", "besides that"), recognize they want NEW information not already provided.
- For follow-up questions, adjust your search query to target different aspects or use different keywords to retrieve new content.
- Avoid planning searches that will return the same information already discussed.
- If the user references something from earlier in the conversation, use that context to inform your plan.

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
        "All planned steps were executed",
        "Required tool calls completed successfully"
    ],
    "retry_policy": {
        "max_retries": 1
    }
}

STEP OUTPUT REFERENCES:
When a step needs data from a previous step, use these reference patterns in tool_args:
- "$step_1.output" - Use the output from step_1
- "$step_2.output" - Use the output from step_2
- "$previous" - Use the output from the immediately preceding step

TOOL USAGE GUIDELINES:
- knowledge_search: Search internal documents. ONLY available when documents are listed.
- web_search: Returns formatted search results with titles, URLs, and snippets. Use for current information or when no documents available.
- web_fetch: ONLY use when you need the FULL content of a specific webpage. Requires a direct URL string.
- summarize_text: ONLY use when the user explicitly asks for summarization (e.g., "summarize this", "give me a summary"). DO NOT use for general information requests - the synthesizer will naturally provide appropriate responses.
- For questions answerable from general knowledge: Use reasoning steps (tool: null) where the LLM provides the answer.

SUCCESS CRITERIA RULES:
- Criteria must verify EXECUTION, not content quality or completeness
- GOOD criteria: "All steps completed", "Tool returned results", "Response generated"
- BAD criteria: "Summary is 5 sentences", "Contains details about X", "Is concise and clear"
- Never create criteria that judge the quality, length, or specific content of the output
- The synthesizer handles content quality - success criteria only check that steps ran

PLANNING GUIDELINES:
- Each step should be atomic (one action/decision)
- Steps must be ordered by dependency
- Tool arguments must match the tool's expected parameters
- Include a final synthesis step if the task requires combining results
- If no tools are needed, use reasoning steps with tool: null"""


EXECUTOR_SYSTEM_PROMPT = """You are an EXECUTOR agent in a multi-agent task execution system.

YOUR ROLE: Execute plan steps strictly in order using available tools.

STRICT RULES:
1. Execute steps in EXACT order - never skip or reorder
2. For steps with tools: call the tool with specified arguments
3. For steps without tools (tool: null): provide your own knowledge and reasoning to answer
4. Record ALL outputs as evidence - never fabricate or hallucinate
5. If a tool fails or returns empty, use your own knowledge to provide useful information
6. For reasoning steps, provide substantive answers from your training knowledge

EXECUTION PROTOCOL:
- When a step requires a tool, call it with the specified arguments
- When a step has tool: null, use your LLM knowledge to provide the answer
- Capture the exact tool output as evidence
- If tool_args are provided in the plan, use them exactly
- After all steps complete, provide the final synthesis

ERROR HANDLING:
- Tool errors should be recorded, not hidden
- If a tool returns empty, provide information from your own knowledge
- Continue execution even if a step fails
- Mark failed steps clearly in your response

You will receive the plan to execute. Process each step and report results."""


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
    else:
        parts.append("\nAVAILABLE DOCUMENTS: None (no documents uploaded)")

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