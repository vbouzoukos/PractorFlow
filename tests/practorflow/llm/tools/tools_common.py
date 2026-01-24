"""
Common helpers for LLM tools tests.

This module provides tool-specific test infrastructure used across
tool_registry and other tool-related test files.
"""

from typing import Any, Dict, List, Optional

from practorflow.llm.tools.base import BaseTool, ToolParameter, ToolResult


class MockTool(BaseTool):
    """
    Mock tool implementation for testing.
    
    Provides configurable behavior for testing ToolRegistry
    and other tool-related functionality.
    """

    def __init__(
        self,
        name: str = "mock_tool",
        description: str = "A mock tool for testing",
        parameters: Optional[List[ToolParameter]] = None,
        result: Optional[ToolResult] = None,
        supports_document_scope: bool = False,
    ):
        """
        Initialize mock tool.
        
        Args:
            name: Tool name identifier.
            description: Tool description.
            parameters: List of tool parameters. Defaults to single query param.
            result: ToolResult to return on execute. Defaults to success with data.
            supports_document_scope: Whether tool has set_document_scope method.
        """
        self._name = name
        self._description = description
        self._parameters = parameters or [
            ToolParameter(
                name="query",
                type="string",
                description="Test query parameter",
                required=True,
            )
        ]
        self._result = result or ToolResult(success=True, data="mock result")
        self._supports_document_scope = supports_document_scope
        self._document_scope: Optional[set] = None
        self._last_call_kwargs: Optional[Dict[str, Any]] = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> List[ToolParameter]:
        return self._parameters

    def execute(self, **kwargs) -> ToolResult:
        """Execute mock tool and record call arguments."""
        self._last_call_kwargs = kwargs
        return self._result

    def set_document_scope(self, document_ids: Optional[set]) -> None:
        """Set document scope if tool supports it."""
        if self._supports_document_scope:
            self._document_scope = document_ids

    def get_last_call_kwargs(self) -> Optional[Dict[str, Any]]:
        """Get kwargs from last execute call for test assertions."""
        return self._last_call_kwargs

    def get_document_scope(self) -> Optional[set]:
        """Get current document scope for test assertions."""
        return self._document_scope


def create_mock_tool(
    name: str = "test_tool",
    description: str = "Test tool",
    success: bool = True,
    data: Any = "test data",
    error: Optional[str] = None,
    supports_document_scope: bool = False,
) -> MockTool:
    """
    Create a MockTool with specified configuration.
    
    Args:
        name: Tool name identifier.
        description: Tool description.
        success: Whether tool execution succeeds.
        data: Data to return in ToolResult.
        error: Error message if success is False.
        supports_document_scope: Whether tool supports document scoping.
    
    Returns:
        Configured MockTool instance.
    """
    result = ToolResult(success=success, data=data, error=error)
    return MockTool(
        name=name,
        description=description,
        result=result,
        supports_document_scope=supports_document_scope,
    )