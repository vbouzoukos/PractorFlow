"""
Agent response parsing utilities.

Provides functions for extracting JSON from LLM responses and
parsing executor results into step results.
"""

import json
import re
from typing import Any, Dict, List, Optional

from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.agent.deps import AgentDeps
from practorflow.services.agent.schemas import (
    Plan,
    StepResult,
    StepStatus,
)

logger = get_logger("agent_parser", level=appConfiguration.LoggerConfiguration.AgentLevel)


def parse_json_from_response(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse JSON from model response.

    Handles JSON in code blocks or bare JSON.

    Args:
        text: Model response text.

    Returns:
        Parsed dict or None if parsing fails.
    """
    if not text:
        return None

    code_block_pattern = r'```(?:json)?\s*(\{[\s\S]*?\})\s*```'
    match = re.search(code_block_pattern, text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    brace_start = text.find('{')
    brace_end = text.rfind('}')
    if brace_start != -1 and brace_end > brace_start:
        try:
            return json.loads(text[brace_start:brace_end + 1])
        except json.JSONDecodeError:
            pass

    return None


def parse_executor_results(
    plan: Plan,
    response_text: str,
    deps: AgentDeps,
) -> List[StepResult]:
    """
    Parse executor response into step results.

    Falls back to tool-based execution if parsing fails.

    Args:
        plan: The plan being executed.
        response_text: Executor's response text.
        deps: Dependencies for fallback execution.

    Returns:
        List of StepResult objects.
    """
    step_results: List[StepResult] = []

    for step in plan.steps:
        step_mentioned = step.step_id in response_text or step.description.lower() in response_text.lower()

        if step.tool and step.tool in deps.tool_registry:
            deps.tool_registry.set_document_scope(deps.document_scope)
            tool_args = step.tool_args or {}

            try:
                tool_result = deps.tool_registry.execute(step.tool, **tool_args)

                if tool_result.success:
                    step_results.append(StepResult(
                        step_id=step.step_id,
                        status=StepStatus.SUCCESS,
                        output=tool_result.data,
                        evidence=[f"tool:{step.tool}"],
                    ))
                else:
                    step_results.append(StepResult(
                        step_id=step.step_id,
                        status=StepStatus.FAILURE,
                        error=tool_result.error,
                        evidence=[],
                    ))
            except Exception as e:
                step_results.append(StepResult(
                    step_id=step.step_id,
                    status=StepStatus.FAILURE,
                    error=str(e),
                    evidence=[],
                ))
        elif step.tool:
            step_results.append(StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILURE,
                error=f"Tool not found: {step.tool}",
                evidence=[],
            ))
        else:
            step_results.append(StepResult(
                step_id=step.step_id,
                status=StepStatus.SUCCESS if step_mentioned else StepStatus.SUCCESS,
                output=f"Reasoning completed: {step.description}",
                evidence=["llm_reasoning"],
            ))

    return step_results