import json
import pytest
from unittest.mock import patch

from practorflow.services.agent.json_helpers import (
    _extract_json_portion,
    _count_braces,
    repair_truncated_json,
    repair_plan_json,
    _aggressive_repair,
    _extract_partial_plan,
)


# ------------------------
# _extract_json_portion
# ------------------------


def test_extract_json_portion_plain_json():
    text = '{"a": 1}'
    assert _extract_json_portion(text) == text


def test_extract_json_portion_markdown_block():
    text = """```json
    {
      "a": 1
    }
    ```"""
    extracted = _extract_json_portion(text)
    assert extracted.startswith("{")
    assert '"a": 1' in extracted


def test_extract_json_portion_no_json():
    text = "no json here"
    assert _extract_json_portion(text) == ""


# ------------------------
# _count_braces
# ------------------------


def test_count_braces_balanced():
    text = '{"a": [1, 2]}'
    result = _count_braces(text)
    assert result[:4] == (1, 1, 1, 1)
    assert result[4] is False


def test_count_braces_unbalanced_and_in_string():
    text = '{"a": "value}'
    open_b, close_b, open_br, close_br, in_string = _count_braces(text)
    assert open_b == 1
    assert close_b == 0
    assert in_string is True


def test_count_braces_escaped_quote():
    text = '{"a": "va\\"lue"}'
    open_b, close_b, open_br, close_br, in_string = _count_braces(text)
    assert open_b == close_b == 1
    assert in_string is False


# ------------------------
# repair_truncated_json
# ------------------------


def test_repair_truncated_json_valid_json_returns_as_is():
    text = '{"a": 1}'
    assert repair_truncated_json(text) == text


def test_repair_truncated_json_closes_unterminated_string_and_brace():
    text = '{"a": "value'
    repaired = repair_truncated_json(text)

    # Must close the string AND the object
    assert repaired.endswith('"}')
    json.loads(repaired)


def test_repair_truncated_json_closes_array_and_object():
    text = '{"a": [1, 2'
    repaired = repair_truncated_json(text)
    assert repaired.endswith("]}")
    json.loads(repaired)


def test_repair_truncated_json_unrepairable_returns_none():
    text = "this is not json at all"
    assert repair_truncated_json(text) is None


def test_repair_truncated_json_falls_back_to_aggressive():
    text = '{"a": 1, "b": 2,'  # trailing comma forces aggressive repair
    repaired = repair_truncated_json(text)
    assert repaired is not None
    json.loads(repaired)


# ------------------------
# _aggressive_repair
# ------------------------


def test_aggressive_repair_simple_case():
    # escaped backslash + escaped quote inside string
    # forces:
    #  - char == '\' and in_string
    #  - escape_next branch on next iteration
    #  - JSONDecodeError -> continue
    text = '{"a": "va\\\\\\"l", "b": 2,'

    with patch(
        "practorflow.services.agent.json_helpers.json.loads",
        side_effect=[
            json.JSONDecodeError("fail", doc="", pos=0),  # first comma -> continue
            {"a": 'va\\"l'},  # second comma -> success
        ],
    ):
        repaired = _aggressive_repair(text)

    assert repaired is not None


def test_aggressive_repair_unrepairable():
    text = '{"a": {"b": '
    assert _aggressive_repair(text) is None


# ------------------------
# repair_plan_json
# ------------------------


def test_repair_plan_json_valid_plan():
    plan = {
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
    }
    text = json.dumps(plan)
    repaired = repair_plan_json(text)
    assert repaired["plan_id"] == "p1"


def test_repair_plan_json_adds_missing_fields():
    text = """
    {
      "plan_id": "p1",
      "steps": [{
        "step_id": "s1",
        "description": "d",
        "tool": null,
        "tool_args": null,
        "expected_output": "x"
      }]
    """
    repaired = repair_plan_json(text)
    assert repaired["task"]
    assert repaired["success_criteria"]


def test_repair_plan_json_invalid_returns_partial():
    text = """
    {
      "plan_id": "p1",
      "task": "t",
      "steps": [
        {
          "step_id": "s1",
          "description": "d",
          "tool": null,
          "tool_args": null,
          "expected_output": "x"
        }
    """
    repaired = repair_plan_json(text)
    assert repaired is not None
    assert repaired["plan_id"] == "p1"
    assert len(repaired["steps"]) == 1


def test_repair_plan_json_no_steps_returns_none():
    text = '{"plan_id": "p1", "task": "t"}'
    assert repair_plan_json(text) is None


# ------------------------
# _extract_partial_plan
# ------------------------


def test_extract_partial_plan_success():
    text = """
    {
      "plan_id": "p1",
      "task": "t",
      "steps": [
        {
          "step_id": "s1",
          "description": "d",
          "tool": null,
          "tool_args": null,
          "expected_output": "x"
        }
      ]
    """
    partial = _extract_partial_plan(text)
    assert partial["plan_id"] == "p1"
    assert len(partial["steps"]) == 1


def test_extract_partial_plan_generates_uuid_when_missing_plan_id():
    text = """
    {
      "steps": [
        {
          "step_id": "s1",
          "description": "d",
          "tool": null,
          "tool_args": null,
          "expected_output": "x"
        }
      ]
    """
    partial = _extract_partial_plan(text)
    assert "plan_id" in partial
    assert partial["steps"]


def test_extract_partial_plan_no_steps_returns_none():
    text = '{"plan_id": "p1", "task": "t"}'
    assert _extract_partial_plan(text) is None
