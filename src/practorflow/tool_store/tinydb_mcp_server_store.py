"""
TinyDB implementation of MCP server configuration storage.

Provides persistent storage for MCP server configs using TinyDB.
"""

import os
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel
from tinydb import TinyDB, Query

from practorflow.llm.tools.mcp.types import (
    HttpConfig,
    MCPServerConfig,
    StdioConfig,
    TransportType,
)
from practorflow.llm.tools.mcp.store import MCPServerStore
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("tool", level=appConfiguration.LoggerConfiguration.ToolLevel)


class TinyDBMCPServerStore(MCPServerStore):
    """
    TinyDB-based MCP server configuration storage.

    Stores global MCP server configurations as JSON documents.
    """

    def __init__(self, db_path: str):
        """
        Initialize TinyDB MCP server store.

        Args:
            db_path: Path to the TinyDB JSON file.
        """
        db_dir = os.path.dirname(db_path)
        if db_dir:
            os.makedirs(db_dir, exist_ok=True)

        self._db = TinyDB(db_path)
        self._table = self._db.table("mcp_servers")
        self._query = Query()

    def create(self, config: MCPServerConfig) -> MCPServerConfig:
        """Create a new MCP server configuration."""
        config.created_at = datetime.now()
        config.updated_at = datetime.now()

        self._table.insert(self._serialize(config))

        logger.info(
            f"[MCPServerStore] Created server '{config.name}' ({config.server_id})"
        )

        return config

    def get(self, server_id: str) -> Optional[MCPServerConfig]:
        """Get an MCP server configuration by ID."""
        result = self._table.get(self._query.server_id == server_id)

        if result is None:
            return None

        return self._deserialize(result)

    def update(self, config: MCPServerConfig) -> Optional[MCPServerConfig]:
        """Update an existing MCP server configuration."""
        existing = self._table.get(self._query.server_id == config.server_id)

        if not existing:
            return None

        config.updated_at = datetime.now()

        self._table.update(
            self._serialize(config),
            self._query.server_id == config.server_id,
        )

        logger.debug(
            f"[MCPServerStore] Updated server '{config.name}' ({config.server_id})"
        )

        return config

    def delete(self, server_id: str) -> bool:
        """Delete an MCP server configuration."""
        removed = self._table.remove(self._query.server_id == server_id)
        if removed:
            logger.info(
                f"[MCPServerStore] Deleted server '{server_id}'"
            )
        return len(removed) > 0

    def list(self) -> List[MCPServerConfig]:
        """List all MCP server configurations."""
        results = self._table.all()
        return [self._deserialize(r) for r in results]

    def close(self) -> None:
        """Close the database connection."""
        self._db.close()

    def _serialize(self, config: MCPServerConfig) -> Dict[str, Any]:
        """Serialize MCPServerConfig to dict for TinyDB storage."""
        data = {
            "server_id": config.server_id,
            "created_at": config.created_at.isoformat(),
            "updated_at": config.updated_at.isoformat(),
            "name": config.name,
            "transport": config.transport.value,
            "stdio_config": (
                config.stdio_config.model_dump()
                if isinstance(config.stdio_config, BaseModel)
                else config.stdio_config
            ) if config.stdio_config else None,
            "http_config": (
                config.http_config.model_dump()
                if isinstance(config.http_config, BaseModel)
                else config.http_config
            ) if config.http_config else None,
            "enabled": config.enabled,
            "purpose": config.purpose,
            "keywords": config.keywords,
            "category": config.category,
            "tags": config.tags,
            "use_when": config.use_when,
            "do_not_use_when": config.do_not_use_when,
        }
        return data

    def _deserialize(self, data: Dict[str, Any]) -> MCPServerConfig:
        """Deserialize dict from TinyDB to MCPServerConfig."""
        return MCPServerConfig(
            server_id=data["server_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            name=data["name"],
            transport=TransportType(data["transport"]),
            stdio_config=StdioConfig(**data["stdio_config"]) if data.get("stdio_config") else None,
            http_config=HttpConfig(**data["http_config"]) if data.get("http_config") else None,
            enabled=data.get("enabled", True),
            purpose=data.get("purpose", ""),
            keywords=data.get("keywords", []),
            category=data.get("category", ""),
            tags=data.get("tags", []),
            use_when=data.get("use_when", []),
            do_not_use_when=data.get("do_not_use_when", []),
        )
