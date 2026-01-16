"""
Common data classes for API clients.

Shared dataclasses used by both ChatClient and AgentClient
for session management, authentication, and data transfer.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SessionSummary:
    """Summary information for a session."""
    session_id: str
    user: Optional[str] = None
    message_count: int = 0
    document_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    title: Optional[str] = None


@dataclass
class MessageInfo:
    """Information for a single message."""
    id: str
    role: str
    content: str
    timestamp: str


@dataclass
class SessionHistory:
    """Full session with message history."""
    session_id: str
    user: Optional[str] = None
    instructions: Optional[str] = None
    messages: List[MessageInfo] = None
    document_count: int = 0
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if self.messages is None:
            self.messages = []


@dataclass
class DocumentInfo:
    """Document information for a session document."""
    id: str
    filename: str
    file_type: str = "unknown"


@dataclass
class AuthStatus:
    """Authentication status information."""
    provider: str
    requires_credentials: bool
    is_open_mode: bool