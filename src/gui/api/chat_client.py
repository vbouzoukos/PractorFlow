"""
Chat API client.

HTTP client for communicating with the PractorFlow FastAPI backend.
Supports session management, message sending with file uploads,
SSE streaming responses, and JWT authentication.
"""

import json
from typing import Iterator, List, Optional
from dataclasses import dataclass

import httpx
from httpx_sse import connect_sse


@dataclass
class StreamChunk:
    """Represents a chunk from the SSE stream."""
    text: str
    finished: bool = False
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    error: Optional[str] = None


@dataclass
class SessionSummary:
    """Summary information for a session."""
    session_id: str
    user: Optional[str] = None
    message_count: int = 0
    document_count: int = 0
    created_at: str = ""
    updated_at: str = ""


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
class AuthStatus:
    """Authentication status information."""
    provider: str
    requires_credentials: bool
    is_open_mode: bool


class ChatClient:
    """
    HTTP client for the PractorFlow Chat API.
    
    Provides synchronous methods for:
    - Authentication (obtaining JWT tokens)
    - Starting chat sessions
    - Sending messages with optional file uploads
    - Streaming responses via SSE
    - Deleting sessions
    - Listing all sessions
    - Retrieving session history
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        app_secret: Optional[str] = None,
        username: Optional[str] = None,
    ):
        """
        Initialize the chat client.
        
        Args:
            base_url: API base URL (e.g., "http://localhost:8000").
            timeout: Request timeout in seconds.
            app_secret: Optional app secret for local authentication.
            username: Optional username for token subject.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._app_secret = app_secret
        self._username = username
        self._access_token: Optional[str] = None
        self._auth_status: Optional[AuthStatus] = None
    
    def _get_auth_headers(self) -> dict:
        """
        Get authorization headers for API requests.
        
        Returns:
            Dictionary with Authorization header if token is available.
        """
        if self._access_token:
            return {"Authorization": f"Bearer {self._access_token}"}
        return {}
    
    def get_auth_status(self) -> AuthStatus:
        """
        Get authentication status from the server.
        
        Returns:
            AuthStatus with provider and mode information.
        
        Raises:
            httpx.HTTPError: If the request fails.
        """
        url = f"{self._base_url}/auth/status"
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url)
            response.raise_for_status()
            
            data = response.json()
            self._auth_status = AuthStatus(
                provider=data.get("provider", "unknown"),
                requires_credentials=data.get("requires_credentials", False),
                is_open_mode=data.get("is_open_mode", True),
            )
            return self._auth_status
    
    def authenticate(
        self,
        app_secret: Optional[str] = None,
        username: Optional[str] = None,
        identity_token: Optional[str] = None,
    ) -> str:
        """
        Authenticate and obtain a JWT token.
        
        Args:
            app_secret: App secret for local authentication.
            username: Optional username for token subject.
            identity_token: Identity token for OIDC authentication.
        
        Returns:
            JWT access token.
        
        Raises:
            httpx.HTTPError: If authentication fails.
        """
        url = f"{self._base_url}/auth/token"
        
        # Use provided credentials or fall back to instance defaults
        secret = app_secret if app_secret is not None else self._app_secret
        user = username if username is not None else self._username
        
        payload = {}
        if secret:
            payload["app_secret"] = secret
        if user:
            payload["username"] = user
        if identity_token:
            payload["identity_token"] = identity_token
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            self._access_token = data.get("access_token")
            return self._access_token
    
    def ensure_authenticated(self) -> None:
        """
        Ensure the client has a valid authentication token.
        
        In open mode, obtains a token without credentials.
        In secure mode, uses configured app_secret.
        
        Raises:
            httpx.HTTPError: If authentication fails.
        """
        if self._access_token:
            return
        
        # Get auth status to determine mode
        if self._auth_status is None:
            self.get_auth_status()
        
        # Authenticate (works for both open mode and secure mode)
        self.authenticate()
    
    def clear_token(self) -> None:
        """Clear the current authentication token."""
        self._access_token = None
    
    def start_session(self) -> str:
        """
        Start a new chat session.
        
        Returns:
            Session ID string.
        
        Raises:
            httpx.HTTPError: If the request fails.
        """
        self.ensure_authenticated()
        
        url = f"{self._base_url}/chat"
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url, headers=self._get_auth_headers())
            response.raise_for_status()
            
            data = response.json()
            return data["session_id"]
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a chat session.
        
        Args:
            session_id: Session ID to delete.
        
        Returns:
            True if deleted successfully.
        
        Raises:
            httpx.HTTPError: If the request fails.
        """
        self.ensure_authenticated()
        
        url = f"{self._base_url}/chat/{session_id}"
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.delete(url, headers=self._get_auth_headers())
            response.raise_for_status()
            
            data = response.json()
            return data.get("deleted", False)
    
    def list_sessions(self, user: Optional[str] = None) -> List[SessionSummary]:
        """
        List all chat sessions.
        
        Args:
            user: Optional user identifier to filter sessions.
        
        Returns:
            List of SessionSummary objects sorted by updated_at descending.
        
        Raises:
            httpx.HTTPError: If the request fails.
        """
        self.ensure_authenticated()
        
        url = f"{self._base_url}/chat/sessions"
        params = {}
        if user is not None:
            params["user"] = user
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(
                url,
                params=params,
                headers=self._get_auth_headers(),
            )
            response.raise_for_status()
            
            data = response.json()
            return [
                SessionSummary(
                    session_id=item.get("session_id", ""),
                    user=item.get("user"),
                    message_count=item.get("message_count", 0),
                    document_count=item.get("document_count", 0),
                    created_at=item.get("created_at", ""),
                    updated_at=item.get("updated_at", ""),
                )
                for item in data
            ]
    
    def get_history(self, session_id: str) -> Optional[SessionHistory]:
        """
        Get full session with message history.
        
        Args:
            session_id: Session ID to retrieve.
        
        Returns:
            SessionHistory object with full message history,
            or None if session not found.
        
        Raises:
            httpx.HTTPError: If the request fails (except 404).
        """
        self.ensure_authenticated()
        
        url = f"{self._base_url}/chat/{session_id}/history"
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url, headers=self._get_auth_headers())
            
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            
            data = response.json()
            messages = [
                MessageInfo(
                    id=msg.get("id", ""),
                    role=msg.get("role", ""),
                    content=msg.get("content", ""),
                    timestamp=msg.get("timestamp", ""),
                )
                for msg in data.get("messages", [])
            ]
            
            return SessionHistory(
                session_id=data.get("session_id", ""),
                user=data.get("user"),
                instructions=data.get("instructions"),
                messages=messages,
                document_count=data.get("document_count", 0),
                created_at=data.get("created_at", ""),
                updated_at=data.get("updated_at", ""),
            )
    
    def send_message_stream(
        self,
        session_id: str,
        message: str,
        file_paths: Optional[list] = None
    ) -> Iterator[StreamChunk]:
        """
        Send a message and stream the response.
        
        Args:
            session_id: Session ID.
            message: Message text.
            file_paths: Optional list of file paths to upload.
        
        Yields:
            StreamChunk objects with response data.
        
        Raises:
            httpx.HTTPError: If the request fails.
        """
        self.ensure_authenticated()
        
        url = f"{self._base_url}/chat/{session_id}"
        
        data = {"message": message}
        files = []
        file_handles = []
        
        try:
            if file_paths:
                for path in file_paths:
                    f = open(path, "rb")
                    file_handles.append(f)
                    filename = path.split("/")[-1].split("\\")[-1]
                    files.append(("files", (filename, f)))
            
            with httpx.Client(timeout=None) as client:
                with connect_sse(
                    client,
                    "POST",
                    url,
                    data=data,
                    files=files if files else None,
                    headers=self._get_auth_headers(),
                ) as event_source:
                    for sse in event_source.iter_sse():
                        if sse.data == "[DONE]":
                            break
                        
                        try:
                            chunk_data = json.loads(sse.data)
                            
                            if "error" in chunk_data:
                                yield StreamChunk(
                                    text="",
                                    finished=True,
                                    error=chunk_data["error"]
                                )
                                break
                            
                            yield StreamChunk(
                                text=chunk_data.get("text", ""),
                                finished=chunk_data.get("finished", False),
                                finish_reason=chunk_data.get("finish_reason"),
                                usage=chunk_data.get("usage")
                            )
                        except json.JSONDecodeError:
                            continue
        
        finally:
            for f in file_handles:
                f.close()
    
    def health_check(self) -> bool:
        """
        Check if the API is healthy.
        
        Returns:
            True if API is reachable and healthy.
        """
        url = f"{self._base_url}/health"
        
        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(url)
                return response.status_code == 200
        except httpx.HTTPError:
            return False