"""
HTTP response parser for API Tools.

Parses HTTP responses and extracts data using output/error mappings.
"""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import httpx

from practorflow.llm.tools.api.models import (
    ApiToolConfig,
    ErrorMapping,
    OutputMapping,
    ParseType,
)


@dataclass
class ParsedResponse:
    """Parsed HTTP response data."""
    success: bool
    data: Any
    error: Optional[str]
    status_code: int
    raw: Any


class ResponseParser:
    """
    Parses HTTP responses based on tool configuration.
    
    Handles:
    - Response body parsing (JSON, text, binary)
    - Success code validation
    - Output field extraction using JSONPath
    - Error field extraction using JSONPath
    """
    
    def parse(self, response: httpx.Response, config: ApiToolConfig) -> ParsedResponse:
        """
        Parse HTTP response based on configuration.
        
        Args:
            response: Raw HTTP response.
            config: Tool configuration with parsing rules.
        
        Returns:
            ParsedResponse with extracted data.
        """
        status_code = response.status_code
        success = status_code in config.success_codes
        
        raw = self._parse_content(response, config.parse_as)
        
        if success:
            if config.output_mapping:
                data = self._apply_output_mapping(raw, config.output_mapping)
            else:
                data = raw
            return ParsedResponse(
                success=True,
                data=data,
                error=None,
                status_code=status_code,
                raw=raw,
            )
        else:
            if config.error_mapping:
                error = self._apply_error_mapping(raw, config.error_mapping)
            else:
                error = f"HTTP {status_code}"
            return ParsedResponse(
                success=False,
                data=None,
                error=error,
                status_code=status_code,
                raw=raw,
            )
    
    def _parse_content(self, response: httpx.Response, parse_as: ParseType) -> Any:
        """
        Parse response body based on parse type.
        
        Args:
            response: HTTP response.
            parse_as: How to parse the response.
        
        Returns:
            Parsed content.
        """
        if parse_as == ParseType.JSON:
            try:
                return response.json()
            except Exception:
                return response.text
        
        if parse_as == ParseType.TEXT:
            return response.text
        
        if parse_as == ParseType.BINARY:
            return response.content
        
        return response.text
    
    def _apply_output_mapping(
        self,
        data: Any,
        mappings: List[OutputMapping],
    ) -> Dict[str, Any]:
        """
        Extract fields from response using JSONPath mappings.
        
        Args:
            data: Parsed response data.
            mappings: Output field mappings.
        
        Returns:
            Dictionary of extracted fields.
        """
        result = {}
        for mapping in mappings:
            value = self._query_jsonpath(data, mapping.json_path)
            result[mapping.field_name] = value
        return result
    
    def _apply_error_mapping(
        self,
        data: Any,
        mappings: List[ErrorMapping],
    ) -> Optional[str]:
        """
        Extract error message from response using JSONPath mappings.
        
        Args:
            data: Parsed response data.
            mappings: Error field mappings.
        
        Returns:
            Combined error message or None.
        """
        errors = []
        for mapping in mappings:
            value = self._query_jsonpath(data, mapping.json_path)
            if value is not None:
                errors.append(str(value))
        
        if errors:
            return "; ".join(errors)
        return None
    
    def _query_jsonpath(self, data: Any, path: str) -> Any:
        """
        Query data using simplified JSONPath.
        
        Supports:
        - Root: $
        - Dot notation: $.field.subfield
        - Array index: $.items[0]
        - Wildcard: $.items[*].name
        
        Args:
            data: Data to query.
            path: JSONPath expression.
        
        Returns:
            Extracted value or None.
        """
        if not path or path == "$":
            return data
        
        path = path.lstrip("$").lstrip(".")
        
        if not path:
            return data
        
        parts = re.split(r'\.(?![^\[]*\])', path)
        current = data
        
        for i, part in enumerate(parts):
            if current is None:
                return None
            
            array_match = re.match(r'(\w*)\[(\d+|\*)\]', part)
            
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
                    remaining_parts = parts[i + 1:]
                    if remaining_parts:
                        remaining_path = ".".join(remaining_parts)
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
                    return [
                        self._query_jsonpath(item, part)
                        for item in current
                        if isinstance(item, dict)
                    ]
                else:
                    return None
        
        return current