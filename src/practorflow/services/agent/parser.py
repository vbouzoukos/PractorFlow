"""
Agent response parsing utilities.

Provides functions for extracting JSON from LLM responses and
parsing executor results into step results.
"""

import json
import re
from typing import Any, Dict, List, Optional, Set

from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

from practorflow.services.agent.deps import AgentDeps
from practorflow.services.agent.schemas import (
    Plan,
    PlanStep,
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


# Mapping of tool names to their primary input parameter that typically
# receives data from previous steps
TOOL_INPUT_PARAMS = {
    "summarize_text": "text",
    "json_transform": "json_data",
}


def _get_last_tool_output(
    step_outputs: Dict[str, Any],
    tool_steps: Set[str],
) -> Optional[Any]:
    """
    Get the most recent non-None output from a tool step (not reasoning step).

    Args:
        step_outputs: Dictionary mapping step_id to output values.
        tool_steps: Set of step_ids that used tools (not reasoning steps).

    Returns:
        The most recent tool output value, or None if none exist.
    """
    if not step_outputs or not tool_steps:
        return None

    for step_id in reversed(list(step_outputs.keys())):
        if step_id in tool_steps:
            output = step_outputs[step_id]
            if output is not None:
                logger.debug(f"[Parser] Found last tool output from {step_id}")
                return output

    return None


def _resolve_tool_args(
    tool_args: Dict[str, Any],
    step_outputs: Dict[str, Any],
    tool_steps: Set[str],
) -> Dict[str, Any]:
    """
    Resolve variable references in tool arguments.

    Supports patterns like:
    - $step_1.output - references output from step_1
    - $step_1 - shorthand for step_1's output
    - $previous - references the immediately previous step's output

    When a referenced step is not found, falls back to the most recent
    successful tool output to handle failed step scenarios.

    Args:
        tool_args: Original tool arguments with potential variable references.
        step_outputs: Dictionary mapping step_id to output values.
        tool_steps: Set of step_ids that used tools.

    Returns:
        Resolved tool arguments with variables substituted.
    """
    if not tool_args:
        return {}

    resolved = {}

    for key, value in tool_args.items():
        if isinstance(value, str):
            # Pattern: $previous
            if value == "$previous":
                last_output = _get_last_tool_output(step_outputs, tool_steps)
                if last_output is not None:
                    resolved[key] = last_output
                    logger.debug(f"[Parser] Resolved $previous -> last tool output")
                else:
                    logger.warning(f"[Parser] Cannot resolve $previous: no tool outputs available")
                    resolved[key] = None
            # Pattern: $step_id.output or $step_id
            elif value.startswith("$"):
                match = re.match(r'^\$(\w+)(?:\.output)?$', value)
                if match:
                    step_id = match.group(1)
                    if step_id in step_outputs and step_outputs[step_id] is not None:
                        resolved[key] = step_outputs[step_id]
                        logger.debug(f"[Parser] Resolved {value} -> (output from {step_id})")
                    else:
                        # Step not found or failed - try fallback to last successful tool output
                        last_output = _get_last_tool_output(step_outputs, tool_steps)
                        if last_output is not None:
                            resolved[key] = last_output
                            logger.warning(
                                f"[Parser] Step {step_id} not found/failed, using fallback from last tool output"
                            )
                        else:
                            logger.warning(f"[Parser] Cannot resolve {value}: no fallback available")
                            resolved[key] = None
                else:
                    resolved[key] = value
            else:
                resolved[key] = value
        elif isinstance(value, dict):
            resolved[key] = _resolve_tool_args(value, step_outputs, tool_steps)
        elif isinstance(value, list):
            resolved[key] = [
                _resolve_tool_args(item, step_outputs, tool_steps) if isinstance(item, dict)
                else item
                for item in value
            ]
        else:
            resolved[key] = value

    return resolved


def _infer_tool_args(
    tool_name: str,
    step_outputs: Dict[str, Any],
    tool_steps: Set[str],
    existing_args: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Infer missing required tool arguments from previous step outputs.

    When a tool like summarize_text has no 'text' argument provided,
    automatically inject the output from the most recent successful tool step.
    Skips reasoning step outputs as they don't contain meaningful data.

    Args:
        tool_name: Name of the tool being executed.
        step_outputs: Dictionary mapping step_id to output values.
        tool_steps: Set of step_ids that used tools.
        existing_args: Arguments already provided in the plan.

    Returns:
        Tool arguments with inferred values for missing required params.
    """
    args = dict(existing_args)

    # Check if this tool has a known input parameter that needs inference
    if tool_name in TOOL_INPUT_PARAMS:
        input_param = TOOL_INPUT_PARAMS[tool_name]

        # Only infer if the parameter is not already provided or is None
        if input_param not in args or args[input_param] is None:
            last_output = _get_last_tool_output(step_outputs, tool_steps)

            if last_output is not None:
                args[input_param] = last_output
                logger.info(
                    f"[Parser] Inferred {tool_name}.{input_param} from last successful tool output"
                )
            else:
                logger.warning(
                    f"[Parser] Cannot infer {tool_name}.{input_param}: no tool outputs available"
                )

    return args


def parse_executor_results(
    plan: Plan,
    response_text: str,
    deps: AgentDeps,
) -> List[StepResult]:
    """
    Parse executor response into step results.

    Falls back to tool-based execution if parsing fails.
    Resolves variable references in tool_args from previous step outputs.
    Infers missing required parameters from previous step outputs when possible.

    Args:
        plan: The plan being executed.
        response_text: Executor's response text.
        deps: Dependencies for fallback execution.

    Returns:
        List of StepResult objects.
    """
    step_results: List[StepResult] = []
    step_outputs: Dict[str, Any] = {}
    tool_steps: Set[str] = set()  # Track which steps used tools

    for step in plan.steps:
        # Check if step has a valid tool (not None, not "null" string)
        has_tool = step.tool and step.tool != "null"
        
        if has_tool and step.tool in deps.tool_registry:
            deps.tool_registry.set_document_scope(deps.document_scope)

            # Start with provided tool_args or empty dict
            raw_tool_args = step.tool_args or {}

            # Resolve any variable references like $step_1.output
            tool_args = _resolve_tool_args(raw_tool_args, step_outputs, tool_steps)

            # Infer missing required parameters from previous tool outputs
            tool_args = _infer_tool_args(step.tool, step_outputs, tool_steps, tool_args)

            try:
                tool_result = deps.tool_registry.execute(step.tool, **tool_args)

                if tool_result.success:
                    # Store output and mark as tool step
                    step_outputs[step.step_id] = tool_result.data
                    tool_steps.add(step.step_id)

                    step_results.append(StepResult(
                        step_id=step.step_id,
                        status=StepStatus.SUCCESS,
                        output=tool_result.data,
                        evidence=[f"tool:{step.tool}"],
                    ))
                else:
                    # Store None but still mark as tool step
                    step_outputs[step.step_id] = None
                    tool_steps.add(step.step_id)

                    step_results.append(StepResult(
                        step_id=step.step_id,
                        status=StepStatus.FAILURE,
                        error=tool_result.error,
                        evidence=[],
                    ))
            except Exception as e:
                step_outputs[step.step_id] = None
                tool_steps.add(step.step_id)

                step_results.append(StepResult(
                    step_id=step.step_id,
                    status=StepStatus.FAILURE,
                    error=str(e),
                    evidence=[],
                ))
        elif has_tool:
            step_outputs[step.step_id] = None

            step_results.append(StepResult(
                step_id=step.step_id,
                status=StepStatus.FAILURE,
                error=f"Tool not found: {step.tool}",
                evidence=[],
            ))
        else:
            # Reasoning step - store output but do NOT add to tool_steps
            # This ensures _get_last_tool_output skips reasoning outputs
            step_outputs[step.step_id] = f"Reasoning completed: {step.description}"

            step_results.append(StepResult(
                step_id=step.step_id,
                status=StepStatus.SUCCESS,
                output=f"Reasoning completed: {step.description}",
                evidence=["llm_reasoning"],
            ))

    return step_results