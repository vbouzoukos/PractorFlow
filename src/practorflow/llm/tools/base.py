"""
Base tool interface for LLM tool calling.

Defines the abstract interface that all tools must implement.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "integer", "number", "boolean", "array"
    description: str
    required: bool = True
    default: Any = None
    enum: Optional[List[Any]] = None  # Allowed values


@dataclass
class ToolResult:
    """Result from tool execution."""
    success: bool
    data: Any = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_context_string(self) -> str:
        """Convert result to string suitable for LLM context."""
        if not self.success:
            return f"Tool execution failed: {self.error}"
        
        if isinstance(self.data, str):
            return self.data
        
        if isinstance(self.data, dict):
            return self._format_dict(self.data)
        
        if isinstance(self.data, list):
            return self._format_list(self.data)
        
        return str(self.data)
    
    def _format_dict(self, d: Dict[str, Any], indent: int = 0) -> str:
        """Format dictionary for readable output."""
        lines = []
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_dict(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_list(value, indent + 1))
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)
    
    def _format_list(self, lst: List[Any], indent: int = 0) -> str:
        """Format list for readable output."""
        lines = []
        prefix = "  " * indent
        for i, item in enumerate(lst, 1):
            if isinstance(item, dict):
                lines.append(f"{prefix}{i}.")
                lines.append(self._format_dict(item, indent + 1))
            else:
                lines.append(f"{prefix}{i}. {item}")
        return "\n".join(lines)
