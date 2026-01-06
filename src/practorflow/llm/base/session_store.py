from abc import ABC, abstractmethod
from practorflow.llm.base.session import Session

class SessionStore(ABC):
    """Base class for session storage. Extend this for persistent storage (DB, file, etc)."""
    
    @abstractmethod
    def get(self, session_id: str) -> Session:
        """Get a session by ID. Implementation decides behavior if not found."""
        pass  # pragma: no cover
    
    @abstractmethod
    def save(self, session: Session):
        """Save a session. Implementation handles persistence."""
        pass  # pragma: no cover
    
    @abstractmethod
    def delete(self, session_id: str):
        """Delete a session. Implementation handles cleanup."""
        pass  # pragma: no cover
    
    @abstractmethod
    def exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        pass  # pragma: no cover