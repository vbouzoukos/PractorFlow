"""
Async base tool interface for LLM tool calling.

Defines the abstract interface for async tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from practorflow.llm.tools.base import ToolParameter, ToolResult


class AsyncBaseTool(ABC):
    """
    Abstract base class for async tools.
    
    Tools are callable units that can be invoked by the LLM runner
    to perform async actions like HTTP requests.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique tool name identifier."""
        pass  # pragma: no cover
    
    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what the tool does.
        This is provided to the LLM to help it decide when to use the tool.
        """
        pass  # pragma: no cover
    
    @property
    @abstractmethod
    def parameters(self) -> List[ToolParameter]:
        """List of parameters the tool accepts."""
        pass  # pragma: no cover
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """
        Execute the tool with given parameters.
        
        Args:
            **kwargs: Tool-specific parameters
            
        Returns:
            ToolResult with success status and data or error
        """
        pass  # pragma: no cover
    
    def validate_parameters(self, **kwargs) -> Optional[str]:
        """
        Validate provided parameters against tool definition.
        
        Args:
            **kwargs: Parameters to validate
            
        Returns:
            Error message if validation fails, None if valid
        """
        provided = set(kwargs.keys())
        
        for param in self.parameters:
            if param.required and param.name not in provided:
                if param.default is None:
                    return f"Missing required parameter: {param.name}"
        
        defined_names = {p.name for p in self.parameters}
        unknown = provided - defined_names
        if unknown:
            return f"Unknown parameters: {', '.join(unknown)}"
        
        return None
    
    def get_schema(self) -> Dict[str, Any]:
        """
        Get tool schema in OpenAI/Anthropic function calling format.
        
        Returns:
            Dictionary representing the tool schema
        """
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                }
            }
        }
    
    async def __call__(self, **kwargs) -> ToolResult:
        """Allow tool to be called directly."""
        validation_error = self.validate_parameters(**kwargs)
        if validation_error:
            return ToolResult(success=False, error=validation_error)
        
        return await self.execute(**kwargs)
