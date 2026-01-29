"""
API Tool Factory.

Singleton factory for creating ApiTool instances from configurations
with shared rate limiters.
"""

from typing import Dict, List, Optional

from tinydb import Query

from practorflow.llm.tools.api.models import ApiToolConfig
from practorflow.llm.tools.api.request import RateLimiter
from practorflow.llm.tools.api.store.base_store import ApiToolStore
from practorflow.llm.tools.api.tool import ApiTool
from practorflow.logger.logger import get_logger
from practorflow.settings.app_settings import appConfiguration

logger = get_logger("tool", level=appConfiguration.LoggerConfiguration.ToolLevel)

_instance: Optional["ApiToolFactory"] = None


def initialize_factory(store: ApiToolStore) -> "ApiToolFactory":
    """
    Initialize the singleton ApiToolFactory instance.
    
    Args:
        store: ApiToolStore implementation for loading configurations.
    
    Returns:
        The initialized ApiToolFactory instance.
    
    Raises:
        RuntimeError: If factory is already initialized.
    """
    global _instance
    if _instance is not None:
        raise RuntimeError("ApiToolFactory is already initialized")
    _instance = ApiToolFactory(store)
    logger.info("[ApiToolFactory] Initialized")
    return _instance


def get_factory() -> "ApiToolFactory":
    """
    Get the singleton ApiToolFactory instance.
    
    Returns:
        The ApiToolFactory instance.
    
    Raises:
        RuntimeError: If factory is not initialized.
    """
    if _instance is None:
        raise RuntimeError("ApiToolFactory is not initialized. Call initialize_factory first.")
    return _instance


def is_factory_initialized() -> bool:
    """
    Check if the factory is initialized.
    
    Returns:
        True if initialized, False otherwise.
    """
    return _instance is not None


class ApiToolFactory:
    """
    Factory for creating ApiTool instances.
    
    Maintains shared rate limiters keyed by tool_id to ensure
    rate limits are enforced across multiple tool instances.
    """
    
    def __init__(self, store: ApiToolStore):
        """
        Initialize factory with storage backend.
        
        Args:
            store: ApiToolStore for loading configurations.
        """
        self._store = store
        self._rate_limiters: Dict[str, RateLimiter] = {}
    
    def create_tool(self, config: ApiToolConfig) -> ApiTool:
        """
        Create an ApiTool instance from configuration.
        
        Gets or creates a shared rate limiter for the tool.
        
        Args:
            config: Tool configuration.
        
        Returns:
            ApiTool instance.
        """
        rate_limiter = self._get_or_create_rate_limiter(config)
        tool = ApiTool(config=config, rate_limiter=rate_limiter)
        
        logger.debug(f"[ApiToolFactory] Created tool '{config.name}' (id={config.tool_id})")
        
        return tool
    
    def create_tools_for_user(self, user_id: str) -> List[ApiTool]:
        """
        Create all enabled tools for a user.
        
        Loads enabled configurations from store for the user and system tools.
        Skips tools with expired secrets.
        
        Args:
            user_id: User ID to load tools for.
        
        Returns:
            List of ApiTool instances (excludes expired tools).
        """
        q = Query()
        query = (q.enabled == True) & ((q.user_id == user_id) | (q.system == True))
        configs = self._store.list_filtered(query)
        tools: List[ApiTool] = []
        
        for config in configs:
            tool = self.create_tool(config)
            
            if tool.expired:
                logger.warning(f"[ApiToolFactory] Skipping expired tool '{config.name}' for user '{user_id}'")
                continue
            
            tools.append(tool)
        
        logger.info(f"[ApiToolFactory] Created {len(tools)} tools for user '{user_id}'")
        
        return tools
    
    def invalidate_rate_limiter(self, tool_id: str) -> None:
        """
        Remove rate limiter for a tool.
        
        Should be called when tool configuration is updated or deleted
        to ensure fresh rate limiter on next tool creation.
        
        Args:
            tool_id: Tool ID to invalidate.
        """
        if tool_id in self._rate_limiters:
            del self._rate_limiters[tool_id]
            logger.debug(f"[ApiToolFactory] Invalidated rate limiter for tool_id={tool_id}")
    
    def _get_or_create_rate_limiter(self, config: ApiToolConfig) -> RateLimiter:
        """
        Get existing or create new rate limiter for a tool.
        
        Args:
            config: Tool configuration.
        
        Returns:
            RateLimiter instance.
        """
        if config.tool_id not in self._rate_limiters:
            self._rate_limiters[config.tool_id] = RateLimiter(rpm=config.rate_limit_rpm)
            logger.debug(f"[ApiToolFactory] Created rate limiter for tool_id={config.tool_id} (rpm={config.rate_limit_rpm})")
        
        return self._rate_limiters[config.tool_id]