"""
Abstract base class for MCP server configuration storage.

Defines the interface for MCP server config persistence.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from practorflow.llm.tools.mcp.types import MCPServerConfig


class MCPServerStore(ABC):
    """
    Abstract base class for MCP server configuration storage.

    Handles CRUD operations for globally available MCP server configs.
    """

    @abstractmethod
    def create(self, config: MCPServerConfig) -> MCPServerConfig:
        """Create a new MCP server configuration."""
        pass  # pragma: no cover

    @abstractmethod
    def get(self, server_id: str) -> Optional[MCPServerConfig]:
        """Get an MCP server configuration by ID."""
        pass  # pragma: no cover

    @abstractmethod
    def update(self, config: MCPServerConfig) -> Optional[MCPServerConfig]:
        """Update an existing MCP server configuration."""
        pass  # pragma: no cover

    @abstractmethod
    def delete(self, server_id: str) -> bool:
        """Delete an MCP server configuration."""
        pass  # pragma: no cover

    @abstractmethod
    def list(self) -> List[MCPServerConfig]:
        """List all MCP server configurations."""
        pass  # pragma: no cover

    @abstractmethod
    def close(self) -> None:
        """Close the storage connection."""
        pass  # pragma: no cover