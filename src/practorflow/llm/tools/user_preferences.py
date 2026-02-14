"""
Abstract base class for user tool preferences storage.

Defines the interface for per-user tool preference persistence.
Allows adaptation to different storage backends (TinyDB, SQL, etc.)
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional


@dataclass
class EnabledToolEntry:
    """Reference to an enabled tool."""

    type: str  # 'builtin', 'api', or 'mcp'
    id: str


@dataclass
class UserToolPreferences:
    """Per-user tool preferences."""

    user_id: str
    enabled_tools: List[EnabledToolEntry] = field(default_factory=list)
    enabled_mcp_servers: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class UserToolPreferencesStore(ABC):
    """
    Abstract base class for user tool preferences storage.

    Handles per-user enabled tools and MCP servers CRUD operations.
    """

    @abstractmethod
    def get(self, user_id: str) -> Optional[UserToolPreferences]:
        """Get tool preferences for a user."""
        pass  # pragma: no cover

    @abstractmethod
    def update(self, preferences: UserToolPreferences) -> UserToolPreferences:
        """Update tool preferences for a user."""
        pass  # pragma: no cover

    @abstractmethod
    def create_default(self, user_id: str) -> UserToolPreferences:
        """Create default preferences for a user."""
        pass  # pragma: no cover

    @abstractmethod
    def delete(self, user_id: str) -> bool:
        """Delete preferences for a user."""
        pass  # pragma: no cover

    @abstractmethod
    def close(self) -> None:
        """Close the storage connection."""
        pass  # pragma: no cover
