"""
JSON repair utilities for handling truncated LLM responses.

Local LLMs often truncate JSON output due to max_tokens limits.
This module provides functions to repair and complete truncated JSON,
specifically for Plan objects in the agent pipeline.
"""

import json
import re
import uuid
from typing import Any, Dict, Optional

from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("json_repair", level=appConfiguration.LoggerConfiguration.AgentLevel)


def _extract_json_portion(text: str) -> str:
    """
    Extract JSON portion from text, handling markdown code blocks.

    Args:
        text: Raw text potentially containing JSON.

    Returns:
        Extracted JSON string starting from first '{'.
    """
    # Remove markdown code block markers
    text = re.sub(r'^```(?:json)?\s*', '', text.strip())
    text = re.sub(r'\s*```$', '', text.strip())

    # Find first brace
    brace_start = text.find('{')
    if brace_start == -1:
        return ""

    return text[brace_start:]


def _count_braces(text: str) -> tuple:
    """
    Count open/close braces and brackets, respecting strings.

    Args:
        text: JSON text to analyze.

    Returns:
        Tuple of (open_braces, close_braces, open_brackets, close_brackets, in_string)
    """
    open_braces = 0
    close_braces = 0
    open_brackets = 0
    close_brackets = 0
    in_string = False
    escape_next = False

    for char in text:
        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == '{':
            open_braces += 1
        elif char == '}':
            close_braces += 1
        elif char == '[':
            open_brackets += 1
        elif char == ']':
            close_brackets += 1

    return open_braces, close_braces, open_brackets, close_brackets, in_string


def repair_truncated_json(text: str) -> Optional[str]:
    """
    Attempt to repair truncated JSON by closing open structures.

    Handles common truncation scenarios:
    - Unclosed strings
    - Unclosed arrays
    - Unclosed objects
    - Missing closing braces/brackets

    Args:
        text: Potentially truncated JSON text.

    Returns:
        Repaired JSON string or None if unrepairable.
    """
    json_text = _extract_json_portion(text)
    if not json_text:
        return None

    # First try parsing as-is
    try:
        json.loads(json_text)
        return json_text
    except json.JSONDecodeError:
        pass

    # Analyze structure
    open_b, close_b, open_br, close_br, in_string = _count_braces(json_text)

    # Build repair suffix
    repair = ""

    # Close unclosed string
    if in_string:
        repair += '"'
        # Re-analyze after closing string
        test_text = json_text + repair
        open_b, close_b, open_br, close_br, in_string = _count_braces(test_text)

    # Close unclosed arrays
    missing_brackets = open_br - close_br
    if missing_brackets > 0:
        repair += ']' * missing_brackets

    # Close unclosed objects
    missing_braces = open_b - close_b
    if missing_braces > 0:
        repair += '}' * missing_braces

    repaired = json_text + repair

    # Try parsing repaired JSON
    try:
        json.loads(repaired)
        logger.debug(f"[JsonRepair] Successfully repaired JSON (added: {repr(repair)})")
        return repaired
    except json.JSONDecodeError as e:
        logger.debug(f"[JsonRepair] Simple repair failed: {e}")

    # Try more aggressive repair - truncate at last valid structure
    return _aggressive_repair(json_text)


def _aggressive_repair(text: str) -> Optional[str]:
    """
    Aggressively repair JSON by finding last valid structural point.

    Truncates JSON at the last complete value and closes structures.

    Args:
        text: Truncated JSON text.

    Returns:
        Repaired JSON string or None.
    """
    # Find positions of structural characters outside strings
    positions = []
    in_string = False
    escape_next = False

    for i, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if char == '\\' and in_string:
            escape_next = True
            continue

        if char == '"':
            in_string = not in_string
            positions.append((i, 'string_toggle'))
            continue

        if in_string:
            continue

        if char in '{}[],':
            positions.append((i, char))

    # Try truncating at each comma from end, looking for valid JSON
    comma_positions = [p[0] for p in positions if p[1] == ',']

    for comma_pos in reversed(comma_positions):
        truncated = text[:comma_pos]

        # Count remaining structures
        open_b, close_b, open_br, close_br, _ = _count_braces(truncated)

        repair = ']' * (open_br - close_br) + '}' * (open_b - close_b)

        try:
            result = truncated + repair
            json.loads(result)
            logger.debug(f"[JsonRepair] Aggressive repair succeeded at position {comma_pos}")
            return result
        except json.JSONDecodeError:
            continue

    return None


def repair_plan_json(text: str) -> Optional[Dict[str, Any]]:
    """
    Repair truncated Plan JSON specifically.

    If JSON repair fails, attempts to construct a minimal valid Plan
    from whatever was successfully parsed.

    Args:
        text: Potentially truncated Plan JSON.

    Returns:
        Parsed Plan dict or None.
    """
    # Try standard repair first
    repaired = repair_truncated_json(text)
    if repaired:
        try:
            parsed = json.loads(repaired)
            # Validate it has minimum Plan structure
            if "plan_id" in parsed and "steps" in parsed:
                # Ensure steps is a list
                if isinstance(parsed["steps"], list) and len(parsed["steps"]) > 0:
                    # Ensure required fields exist
                    if "task" not in parsed:
                        parsed["task"] = "Task from truncated response"
                    if "success_criteria" not in parsed:
                        parsed["success_criteria"] = ["All steps completed"]
                    return parsed
        except json.JSONDecodeError: # pragma: no cover
            pass # pragma: no cover

    # Try to extract partial Plan data
    return _extract_partial_plan(text)


def _extract_partial_plan(text: str) -> Optional[Dict[str, Any]]:
    """
    Extract a partial Plan from truncated JSON.

    Attempts to find and parse individual fields to construct
    a minimal valid Plan.

    Args:
        text: Truncated JSON text.

    Returns:
        Minimal Plan dict or None.
    """
    logger.debug("[JsonRepair] Attempting partial Plan extraction")

    # Try to extract plan_id
    plan_id_match = re.search(r'"plan_id"\s*:\s*"([^"]+)"', text)
    plan_id = plan_id_match.group(1) if plan_id_match else str(uuid.uuid4())

    # Try to extract task
    task_match = re.search(r'"task"\s*:\s*"([^"]+)"', text)
    task = task_match.group(1) if task_match else "Task"

    # Try to extract complete steps
    steps = []
    step_pattern = re.compile(
        r'\{\s*"step_id"\s*:\s*"([^"]+)"\s*,\s*'
        r'"description"\s*:\s*"([^"]+)"\s*,\s*'
        r'"tool"\s*:\s*(null|"[^"]*")\s*,\s*'
        r'"tool_args"\s*:\s*(\{[^}]*\}|null)\s*,\s*'
        r'"expected_output"\s*:\s*"([^"]+)"',
        re.DOTALL
    )

    for match in step_pattern.finditer(text):
        step_id = match.group(1)
        description = match.group(2)
        tool_raw = match.group(3)
        tool = None if tool_raw == "null" else tool_raw.strip('"')
        tool_args_raw = match.group(4)
        try:
            tool_args = None if tool_args_raw == "null" else json.loads(tool_args_raw)
        except json.JSONDecodeError: # pragma: no cover
            continue # pragma: no cover
        expected_output = match.group(5)

        steps.append({
            "step_id": step_id,
            "description": description,
            "tool": tool,
            "tool_args": tool_args,
            "expected_output": expected_output,
        })

    if not steps:
        logger.warning("[JsonRepair] Could not extract any complete steps")
        return None

    logger.info(f"[JsonRepair] Extracted {len(steps)} complete steps from truncated JSON")

    return {
        "plan_id": plan_id,
        "task": task,
        "steps": steps,
        "success_criteria": ["All steps completed"],
        "retry_policy": {"max_retries": 1},
    }