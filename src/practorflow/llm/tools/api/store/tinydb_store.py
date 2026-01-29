"""
TinyDB implementation of API Tool storage.

Provides persistent storage for tool configurations using TinyDB.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from tinydb import TinyDB, Query

from practorflow.llm.tools.api.models.models import ApiToolConfig
from practorflow.llm.tools.api.store.base_store import ApiToolStore


class TinyDBApiToolStore(ApiToolStore):
    """
    TinyDB-based API tool storage.
    
    Stores tool configurations in a single TinyDB table.
    """
    
    def __init__(self, db_path: str):
        """
        Initialize TinyDB API tool store.
        
        Args:
            db_path: Path to the TinyDB JSON file.
        """
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)
        
        self._db = TinyDB(db_path)
        self._table = self._db.table("api_tools")
        self._query = Query()
    
    def create(self, tool: ApiToolConfig) -> ApiToolConfig:
        """Create a new tool configuration."""
        existing = self._table.get(
            (self._query.name == tool.name) & 
            (self._query.user_id == tool.user_id)
        )
        if existing:
            raise ValueError(f"Tool with name '{tool.name}' already exists")
        
        tool.created_at = datetime.now()
        tool.updated_at = datetime.now()
        
        self._table.insert(self._serialize(tool))
        return tool
    
    def get(self, tool_id: str, user_id: str) -> Optional[ApiToolConfig]:
        """Get a tool configuration by ID."""
        result = self._table.get(
            (self._query.tool_id == tool_id) & 
            (self._query.user_id == user_id)
        )
        if result:
            return self._deserialize(result)
        return None
    
    def update(self, tool: ApiToolConfig) -> Optional[ApiToolConfig]:
        """Update an existing tool configuration."""
        existing = self._table.get(
            (self._query.tool_id == tool.tool_id) & 
            (self._query.user_id == tool.user_id)
        )
        if not existing:
            return None
        
        tool.updated_at = datetime.now()
        
        self._table.update(
            self._serialize(tool),
            (self._query.tool_id == tool.tool_id) & 
            (self._query.user_id == tool.user_id)
        )
        return tool
    
    def delete(self, tool_id: str, user_id: str) -> bool:
        """Delete a tool configuration."""
        removed = self._table.remove(
            (self._query.tool_id == tool_id) & 
            (self._query.user_id == user_id)
        )
        return len(removed) > 0
    
    def list(self, user_id: str) -> List[ApiToolConfig]:
        """List all tool configurations for a user."""
        results = self._table.search(self._query.user_id == user_id)
        return [self._deserialize(r) for r in results]
    
    def list_enabled(self, user_id: str) -> List[ApiToolConfig]:
        """List enabled tool configurations for a user."""
        results = self._table.search(
            (self._query.user_id == user_id) & 
            (self._query.enabled == True)
        )
        return [self._deserialize(r) for r in results]
    
    def list_filtered(self, query: Any) -> List[ApiToolConfig]:
        """
        List tool configurations matching query.
        
        Args:
            query: TinyDB Query condition.
        
        Returns:
            List of matching ApiToolConfig instances.
        """
        results = self._table.search(query)
        return [self._deserialize(r) for r in results]
    
    def close(self) -> None:
        """Close the database connection."""
        self._db.close()
    
    def _serialize(self, tool: ApiToolConfig) -> Dict[str, Any]:
        """Serialize ApiToolConfig to dict for storage."""
        data = tool.model_dump()
        data["created_at"] = tool.created_at.isoformat()
        data["updated_at"] = tool.updated_at.isoformat()
        if tool.auth_secret_expires_at:
            data["auth_secret_expires_at"] = tool.auth_secret_expires_at.isoformat()
        return data
    
    def _deserialize(self, data: Dict[str, Any]) -> ApiToolConfig:
        """Deserialize dict to ApiToolConfig."""
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        if data.get("auth_secret_expires_at"):
            data["auth_secret_expires_at"] = datetime.fromisoformat(data["auth_secret_expires_at"])
        return ApiToolConfig.model_validate(data)