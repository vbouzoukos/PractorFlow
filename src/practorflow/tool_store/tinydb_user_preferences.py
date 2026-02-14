"""
TinyDB implementation of user tool preferences storage.

Provides persistent storage for per-user enabled tools and MCP servers using TinyDB.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from tinydb import TinyDB, Query

from practorflow.llm.tools.user_preferences import (
    EnabledToolEntry,
    UserToolPreferences,
    UserToolPreferencesStore,
)
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("tool", level=appConfiguration.LoggerConfiguration.ToolLevel)


class TinyDBUserToolPreferencesStore(UserToolPreferencesStore):
    """
    TinyDB-based user tool preferences storage.

    Stores one record per user tracking which tools and MCP servers
    are enabled.
    """

    def __init__(self, db_path: str):
        """
        Initialize TinyDB user tool preferences store.

        Args:
            db_path: Path to the TinyDB JSON file.
        """
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._db = TinyDB(db_path)
        self._table = self._db.table("user_tool_preferences")
        self._query = Query()

    def get(self, user_id: str) -> Optional[UserToolPreferences]:
        """Get tool preferences for a user."""
        result = self._table.get(self._query.user_id == user_id)

        if result is None:
            return None

        return self._deserialize(result)

    def update(self, preferences: UserToolPreferences) -> UserToolPreferences:
        """Update tool preferences for a user."""
        preferences.updated_at = datetime.now()

        existing = self._table.get(self._query.user_id == preferences.user_id)

        if existing:
            self._table.update(
                self._serialize(preferences),
                self._query.user_id == preferences.user_id,
            )
            logger.debug(
                f"[UserToolPreferences] Updated preferences for user '{preferences.user_id}'"
            )
        else:
            self._table.insert(self._serialize(preferences))
            logger.info(
                f"[UserToolPreferences] Created preferences for user '{preferences.user_id}'"
            )

        return preferences

    def create_default(self, user_id: str) -> UserToolPreferences:
        """Create default preferences for a user."""
        preferences = UserToolPreferences(user_id=user_id)

        self._table.insert(self._serialize(preferences))

        logger.info(
            f"[UserToolPreferences] Created default preferences for user '{user_id}'"
        )

        return preferences

    def delete(self, user_id: str) -> bool:
        """Delete preferences for a user."""
        removed = self._table.remove(self._query.user_id == user_id)
        if removed:
            logger.info(
                f"[UserToolPreferences] Deleted preferences for user '{user_id}'"
            )
        return len(removed) > 0

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def _serialize(self, preferences: UserToolPreferences) -> Dict[str, Any]:
        """Serialize UserToolPreferences to dict for TinyDB storage."""
        return {
            "user_id": preferences.user_id,
            "enabled_tools": [
                {"type": entry.type, "id": entry.id}
                for entry in preferences.enabled_tools
            ],
            "enabled_mcp_servers": preferences.enabled_mcp_servers,
            "created_at": preferences.created_at.isoformat(),
            "updated_at": preferences.updated_at.isoformat(),
        }

    def _deserialize(self, data: Dict[str, Any]) -> UserToolPreferences:
        """Deserialize dict from TinyDB to UserToolPreferences."""
        return UserToolPreferences(
            user_id=data["user_id"],
            enabled_tools=[
                EnabledToolEntry(type=entry["type"], id=entry["id"])
                for entry in data.get("enabled_tools", [])
            ],
            enabled_mcp_servers=data.get("enabled_mcp_servers", []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )