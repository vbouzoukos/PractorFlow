"""
Abstract base class for API Tool storage.

Defines the interface for tool configuration storage.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

from practorflow.llm.tools.api.models.models import ApiToolConfig


class ApiToolStore(ABC):
    """
    Abstract base class for API tool storage.
    
    Handles tool configuration CRUD operations.
    Tool ID (tool_id) is the sole unique identifier.
    Authorization logic belongs in the service/API layer, not here.
    """
    
    @abstractmethod
    def create(self, tool: ApiToolConfig) -> ApiToolConfig:
        """Create a new tool configuration."""
    
    @abstractmethod
    def get(self, tool_id: str) -> Optional[ApiToolConfig]:
        """Get a tool configuration by ID."""
    
    @abstractmethod
    def update(self, tool: ApiToolConfig) -> Optional[ApiToolConfig]:
        """Update an existing tool configuration."""
    
    @abstractmethod
    def delete(self, tool_id: str) -> bool:
        """Delete a tool configuration."""
    
    @abstractmethod
    def list(self, user_id: str) -> List[ApiToolConfig]:
        """List all tool configurations for a user."""
    
    @abstractmethod
    def list_enabled(self, user_id: str) -> List[ApiToolConfig]:
        """List enabled tool configurations for a user."""
    
    @abstractmethod
    def list_filtered(self, query: Any) -> List[ApiToolConfig]:
        """
        List tool configurations matching query.
        
        Args:
            query: Query condition (implementation-specific, e.g., TinyDB Query).
        
        Returns:
            List of matching ApiToolConfig instances.
        """
    
    @abstractmethod
    def close(self) -> None:
        """Close the storage connection."""