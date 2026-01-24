"""
JSON transformation tool.

Provides JSON parsing, querying with JSONPath, and transformation operations.
Useful for processing structured data from APIs or documents.
"""

import json
import re
from typing import Any, Dict, List, Optional, Union

from practorflow.llm.tools.base import BaseTool, ToolParameter, ToolResult
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("tool", level=appConfiguration.LoggerConfiguration.ToolLevel)


class JsonTransformTool(BaseTool):
    """
    JSON transformation tool.

    Supports:
    - Parsing JSON strings
    - Querying with JSONPath expressions
    - Extracting specific fields
    - Flattening nested structures
    """

    def __init__(self, max_input_length: int = 100000):
        """
        Initialize JSON transform tool.

        Args:
            max_input_length: Maximum input JSON length in characters.
        """
        self._max_input_length = max_input_length

    @property
    def name(self) -> str:
        return "json_transform"

    @property
    def description(self) -> str:
        return (
            "Parse, query, and transform JSON data. "
            "Use this tool to extract specific fields from JSON, "
            "query nested structures with JSONPath, or flatten complex objects."
        )

    @property
    def parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="json_data",
                type="string",
                description="JSON string to process, or already parsed object",
                required=True,
            ),
            ToolParameter(
                name="operation",
                type="string",
                description="Operation: parse, query, extract, flatten, keys, values",
                required=True,
                enum=["parse", "query", "extract", "flatten", "keys", "values"],
            ),
            ToolParameter(
                name="path",
                type="string",
                description="JSONPath expression for 'query' or dot-notation path for 'extract' (e.g., 'data.users[0].name')",
                required=False,
            ),
            ToolParameter(
                name="fields",
                type="array",
                description="List of field names to extract (for 'extract' operation with multiple fields)",
                required=False,
            ),
        ]

    def _parse_json(self, json_data: Union[str, Dict, List]) -> Any:
        """
        Parse JSON string or return already parsed data.

        Args:
            json_data: JSON string or parsed object.

        Returns:
            Parsed JSON data.

        Raises:
            ValueError: If JSON is invalid.
        """
        if isinstance(json_data, (dict, list)):
            return json_data

        if isinstance(json_data, str):
            if len(json_data) > self._max_input_length:
                raise ValueError(f"JSON exceeds maximum length of {self._max_input_length}")
            return json.loads(json_data)

        raise ValueError(f"Invalid json_data type: {type(json_data)}")

    def _query_jsonpath(self, data: Any, path: str) -> Any:
        """
        Query JSON data using simplified JSONPath.

        Supports:
        - Dot notation: data.field.subfield
        - Array indexing: data.items[0]
        - Wildcard: data.items[*].name

        Args:
            data: Parsed JSON data.
            path: JSONPath expression.

        Returns:
            Queried value(s).
        """
        if not path or path == "$":
            return data

        path = path.lstrip("$.")

        parts = re.split(r'\.(?![^\[]*\])', path)
        current = data

        for part in parts:
            if not part:
                continue

            array_match = re.match(r'(\w+)\[(\d+|\*)\]', part)

            if array_match:
                field_name = array_match.group(1)
                index = array_match.group(2)

                if field_name:
                    if isinstance(current, dict):
                        current = current.get(field_name)
                    else:
                        return None

                if current is None:
                    return None

                if not isinstance(current, list):
                    return None

                if index == "*":
                    remaining_path = ".".join(parts[parts.index(part) + 1:])
                    if remaining_path:
                        return [self._query_jsonpath(item, remaining_path) for item in current]
                    return current
                else:
                    idx = int(index)
                    if idx < len(current):
                        current = current[idx]
                    else:
                        return None
            else:
                if isinstance(current, dict):
                    current = current.get(part)
                elif isinstance(current, list):
                    return [self._query_jsonpath(item, part) for item in current if isinstance(item, dict)]
                else:
                    return None

            if current is None:
                return None

        return current

    def _extract_path(self, data: Any, path: str) -> Any:
        """
        Extract value using dot notation path.

        Args:
            data: Parsed JSON data.
            path: Dot notation path (e.g., 'user.address.city').

        Returns:
            Extracted value or None.
        """
        return self._query_jsonpath(data, path)

    def _extract_fields(self, data: Any, fields: List[str]) -> Dict[str, Any]:
        """
        Extract multiple fields from JSON.

        Args:
            data: Parsed JSON data.
            fields: List of field paths to extract.

        Returns:
            Dictionary of field names to values.
        """
        result = {}
        for field in fields:
            key = field.split(".")[-1]
            result[key] = self._extract_path(data, field)
        return result

    def _flatten(self, data: Any, prefix: str = "", separator: str = ".") -> Dict[str, Any]:
        """
        Flatten nested JSON structure.

        Args:
            data: Parsed JSON data.
            prefix: Current key prefix.
            separator: Separator for nested keys.

        Returns:
            Flattened dictionary.
        """
        result = {}

        if isinstance(data, dict):
            for key, value in data.items():
                new_key = f"{prefix}{separator}{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    result.update(self._flatten(value, new_key, separator))
                else:
                    result[new_key] = value
        elif isinstance(data, list):
            for idx, item in enumerate(data):
                new_key = f"{prefix}[{idx}]"
                if isinstance(item, (dict, list)):
                    result.update(self._flatten(item, new_key, separator))
                else:
                    result[new_key] = item
        else:
            result[prefix] = data

        return result

    def execute(self, **kwargs) -> ToolResult:
        """
        Execute JSON transformation.

        Args:
            json_data: JSON string or object to process.
            operation: Operation to perform.
            path: Path for query/extract operations.
            fields: Fields for multi-field extraction.

        Returns:
            ToolResult with transformed data or error.
        """
        json_data = kwargs.get("json_data")
        operation = kwargs.get("operation")
        path = kwargs.get("path")
        fields = kwargs.get("fields")

        try:
            logger.debug(f"[JsonTransform] Operation: {operation}")

            parsed = self._parse_json(json_data)

            if operation == "parse":
                return ToolResult(
                    success=True,
                    data=parsed,
                    metadata={"operation": "parse", "type": type(parsed).__name__},
                )

            elif operation == "query":
                if not path:
                    return ToolResult(success=False, error="'path' is required for query operation")

                result = self._query_jsonpath(parsed, path)
                return ToolResult(
                    success=True,
                    data=result,
                    metadata={"operation": "query", "path": path},
                )

            elif operation == "extract":
                if fields:
                    result = self._extract_fields(parsed, fields)
                elif path:
                    result = self._extract_path(parsed, path)
                else:
                    return ToolResult(success=False, error="'path' or 'fields' required for extract operation")

                return ToolResult(
                    success=True,
                    data=result,
                    metadata={"operation": "extract"},
                )

            elif operation == "flatten":
                result = self._flatten(parsed)
                return ToolResult(
                    success=True,
                    data=result,
                    metadata={"operation": "flatten", "keys_count": len(result)},
                )

            elif operation == "keys":
                if isinstance(parsed, dict):
                    result = list(parsed.keys())
                else:
                    return ToolResult(success=False, error="'keys' operation requires a JSON object")

                return ToolResult(
                    success=True,
                    data=result,
                    metadata={"operation": "keys", "count": len(result)},
                )

            elif operation == "values":
                if isinstance(parsed, dict):
                    result = list(parsed.values())
                else:
                    return ToolResult(success=False, error="'values' operation requires a JSON object")

                return ToolResult(
                    success=True,
                    data=result,
                    metadata={"operation": "values", "count": len(result)},
                )

            else:
                return ToolResult(success=False, error=f"Unknown operation: {operation}")

        except json.JSONDecodeError as e:
            logger.error(f"[JsonTransform] JSON parse error: {e}")
            return ToolResult(success=False, error=f"Invalid JSON: {str(e)}")
        except Exception as e:
            logger.error(f"[JsonTransform] Error: {e}")
            return ToolResult(success=False, error=f"JSON transformation failed: {str(e)}")