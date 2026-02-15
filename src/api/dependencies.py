"""
FastAPI dependencies for dependency injection.
"""

from typing import Optional

from fastapi import Depends, HTTPException, status

from practorflow.services.chat import ChatService
from practorflow.services.agent import AgentService
from practorflow.services.history.truncator import DeleteSessionService
from practorflow.session_store.session_history import SessionHistory
from api.auth.service import AuthService
from api.auth.dependencies import get_current_user
from api.auth.schemas import UserContext
from practorflow.llm.tools.api.store.base_store import ApiToolStore
from practorflow.llm.tools.api.encryption import EncryptionService
from practorflow.llm.tools.api.models.models import ApiToolConfig
from practorflow.llm.tools.mcp.store import MCPServerStore
from practorflow.llm.tools.user_preferences import UserToolPreferencesStore


# System tools use empty string as user_id
SYSTEM_USER_ID = ""


class ServiceContainer:
    """
    Container for application services.
    
    Initialized during application lifespan and used for dependency injection.
    """
    
    chat_service: Optional[ChatService] = None
    agent_service: Optional[AgentService] = None
    session_history: Optional[SessionHistory] = None
    auth_service: Optional[AuthService] = None
    delete_session_service: Optional[DeleteSessionService] = None
    api_tool_store: Optional[ApiToolStore] = None
    encryption_service: Optional[EncryptionService] = None
    mcp_server_store: Optional[MCPServerStore] = None
    tool_preferences_store: Optional[UserToolPreferencesStore] = None


# Global service container instance
container = ServiceContainer()


def get_chat_service() -> ChatService:
    """
    Dependency to get ChatService instance.
    
    Returns:
        ChatService instance.
    
    Raises:
        RuntimeError: If ChatService is not initialized.
    """
    if container.chat_service is None:
        raise RuntimeError("ChatService not initialized. Application not started properly.")
    return container.chat_service


def get_agent_service() -> AgentService:
    """
    Dependency to get AgentService instance.
    
    Returns:
        AgentService instance.
    
    Raises:
        RuntimeError: If AgentService is not initialized.
    """
    if container.agent_service is None:
        raise RuntimeError("AgentService not initialized. Application not started properly.")
    return container.agent_service


def get_session_history() -> SessionHistory:
    """
    Dependency to get SessionHistory instance.
    
    Returns:
        SessionHistory instance.
    
    Raises:
        RuntimeError: If SessionHistory is not initialized.
    """
    if container.session_history is None:
        raise RuntimeError("SessionHistory not initialized. Application not started properly.")
    return container.session_history


def get_auth_service() -> AuthService:
    """
    Dependency to get AuthService instance.
    
    Returns:
        AuthService instance.
    
    Raises:
        RuntimeError: If AuthService is not initialized.
    """
    if container.auth_service is None:
        raise RuntimeError("AuthService not initialized. Application not started properly.")
    return container.auth_service

def get_delete_session_service() -> DeleteSessionService:
    """
    Dependency to get DeleteSessionService instance.
    
    Returns:
        DeleteSessionService instance.
    
    Raises:
        RuntimeError: If DeleteSessionService is not initialized.
    """
    if container.delete_session_service is None:
        raise RuntimeError("DeleteSessionService not initialized. Application not started properly.")
    return container.delete_session_service


def get_api_tool_store() -> ApiToolStore:
    """
    Dependency to get ApiToolStore instance.
    
    Returns:
        ApiToolStore instance.
    
    Raises:
        RuntimeError: If ApiToolStore is not initialized.
    """
    if container.api_tool_store is None:
        raise RuntimeError("ApiToolStore not initialized. Application not started properly.")
    return container.api_tool_store


def get_encryption_service() -> EncryptionService:
    """
    Dependency to get EncryptionService instance.
    
    Returns:
        EncryptionService instance.
    
    Raises:
        RuntimeError: If EncryptionService is not initialized.
    """
    if container.encryption_service is None:
        raise RuntimeError("EncryptionService not initialized. Application not started properly.")
    return container.encryption_service


def get_mcp_server_store() -> MCPServerStore:
    """
    Dependency to get MCPServerStore instance.

    Returns:
        MCPServerStore instance.

    Raises:
        RuntimeError: If MCPServerStore is not initialized.
    """
    if container.mcp_server_store is None:
        raise RuntimeError("MCPServerStore not initialized. Application not started properly.")
    return container.mcp_server_store


def get_tool_preferences_store() -> UserToolPreferencesStore:
    """
    Dependency to get UserToolPreferencesStore instance.

    Returns:
        UserToolPreferencesStore instance.

    Raises:
        RuntimeError: If UserToolPreferencesStore is not initialized.
    """
    if container.tool_preferences_store is None:
        raise RuntimeError("UserToolPreferencesStore not initialized. Application not started properly.")
    return container.tool_preferences_store


async def resolve_tool(
    tool_id: str,
    current_user: UserContext = Depends(get_current_user),
    store: ApiToolStore = Depends(get_api_tool_store),
) -> ApiToolConfig:
    """
    Resolve a tool by path parameter with authorization enforcement.
    
    Looks up tool by tool_id, then checks:
    - If tool belongs to current user: allowed.
    - If tool is a system tool: requires llm_admin permission.
    - Otherwise: 403 forbidden.
    
    Args:
        tool_id: Tool identifier from path.
        current_user: Authenticated user context.
        store: API tool store instance.
    
    Returns:
        ApiToolConfig instance.
    
    Raises:
        HTTPException: 404 if not found, 403 if not authorized.
    """
    tool = store.get(tool_id)
    if tool is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Tool '{tool_id}' not found",
        )

    if tool.user_id == current_user.user_id:
        return tool

    if tool.system and "llm_admin" in current_user.permissions:
        return tool

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Not authorized to access this tool",
    )