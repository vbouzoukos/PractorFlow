"""
Chat API client.

HTTP client for communicating with the PractorFlow FastAPI backend.
Supports session creation, message sending with file uploads,
SSE streaming responses, and JWT authentication.
"""

import json
from typing import Iterator, List, Optional
from dataclasses import dataclass

import httpx
from httpx_sse import connect_sse

from gui.api.client_data import AuthStatus


@dataclass
class StreamChunk:
    """Represents a chunk from the SSE stream."""
    text: str
    finished: bool = False
    finish_reason: Optional[str] = None
    usage: Optional[dict] = None
    error: Optional[str] = None


class ChatClient:
    """
    HTTP client for the PractorFlow Chat API.
    
    Provides synchronous methods for:
    - Authentication (obtaining JWT tokens)
    - Starting chat sessions
    - Sending messages with optional file uploads
    - Streaming responses via SSE
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
            AuthStatus object with provider info.
        
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
    
    def authenticate(self, app_secret: Optional[str] = None) -> str:
        """
        Authenticate and obtain a JWT token.
        
        Args:
            app_secret: Optional app secret (uses instance default if not provided).
        
        Returns:
            Access token string.
        
        Raises:
            httpx.HTTPError: If authentication fails.
        """
        url = f"{self._base_url}/auth/token"
        
        secret = app_secret or self._app_secret
        payload = {}
        
        if secret:
            payload["app_secret"] = secret
        if self._username:
            payload["username"] = self._username
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.post(url, json=payload)
            response.raise_for_status()
            
            data = response.json()
            self._access_token = data["access_token"]
            return self._access_token
    
    def ensure_authenticated(self) -> None:
        """
        Ensure we have a valid authentication token.
        
        In open mode, obtains a token without credentials.
        In secure mode, uses configured app_secret.
        
        Raises:
            httpx.HTTPError: If authentication fails.
        """
        if self._access_token:
            return
        
        if self._auth_status is None:
            self.get_auth_status()
        
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
        
        files = []
        if file_paths:
            for path in file_paths:
                files.append(("files", open(path, "rb")))
        
        try:
            with httpx.Client(timeout=None) as client:
                with connect_sse(
                    client,
                    "POST",
                    url,
                    data={"message": message},
                    files=files if files else None,
                    headers=self._get_auth_headers(),
                ) as event_source:
                    for event in event_source.iter_sse():
                        if event.data == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(event.data)
                            
                            if "error" in data:
                                yield StreamChunk(
                                    text="",
                                    finished=True,
                                    error=data.get("error"),
                                )
                                break
                            
                            yield StreamChunk(
                                text=data.get("text", ""),
                                finished=data.get("finished", False),
                                finish_reason=data.get("finish_reason"),
                                usage=data.get("usage"),
                            )
                        except json.JSONDecodeError:
                            continue
        finally:
            for _, f in files:
                f.close()