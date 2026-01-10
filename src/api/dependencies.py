"""
FastAPI dependencies for dependency injection.
"""

from typing import Optional

from practorflow.services.chat import ChatService
from practorflow.services.agent import AgentService
from practorflow.session_store.session_history import SessionHistory
from api.auth.service import AuthService


class ServiceContainer:
    """
    Container for application services.
    
    Initialized during application lifespan and used for dependency injection.
    """
    
    chat_service: Optional[ChatService] = None
    agent_service: Optional[AgentService] = None
    session_history: Optional[SessionHistory] = None
    auth_service: Optional[AuthService] = None


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