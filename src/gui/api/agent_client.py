"""
Agent API client.

HTTP client for communicating with the PractorFlow Agent API.
Supports session management, task execution with file uploads,
and JWT authentication.
"""

from typing import List, Optional
from dataclasses import dataclass

import httpx

from gui.api.client_data import (
    SessionSummary,
    MessageInfo,
    SessionHistory,
    AuthStatus,
)


@dataclass
class AgentTaskResult:
    """Response from agent task execution."""
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None


@dataclass
class AgentJobStatus:
    job_id: str
    status: str
    result: Optional[dict] = None
    error: Optional[str] = None


class AgentClient:
    """
    HTTP client for the PractorFlow Agent API.
    
    Provides synchronous methods for:
    - Authentication (obtaining JWT tokens)
    - Starting agent sessions
    - Executing tasks with optional file uploads
    - Deleting sessions
    - Listing all agent sessions
    - Retrieving session history
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 120.0,
        app_secret: Optional[str] = None,
        username: Optional[str] = None,
    ):
        """
        Initialize the agent client.
        
        Args:
            base_url: API base URL (e.g., "http://localhost:8000").
            timeout: Request timeout in seconds (longer default for agent tasks).
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
        Start a new agent session.
        
        Returns:
            Session ID string.
        
        Raises:
            httpx.HTTPError: If the request fails.
        """
        self.ensure_authenticated()
        
        url = f"{self._base_url}/agent"
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url, headers=self._get_auth_headers())
            response.raise_for_status()
            
            data = response.json()
            return data["session_id"]
    
    def execute_task(
        self,
        session_id: str,
        task: str,
        file_paths: Optional[List[str]] = None,
    ) -> str:
        """
        Execute an agent task.
        
        Args:
            session_id: Session ID.
            task: Task description to execute.
            file_paths: Optional list of file paths to upload.
        
        Returns:
            Job ID with task outcome.
        
        Raises:
            httpx.HTTPError: If the request fails.
        """
        self.ensure_authenticated()

        url = f"{self._base_url}/agent/{session_id}/execute"

        files = []
        if file_paths:
            for path in file_paths:
                files.append(("files", open(path, "rb")))

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    url,
                    data={"task": task},
                    files=files if files else None,
                    headers=self._get_auth_headers(),
                )
                response.raise_for_status()
                return response.json()["job_id"]
        finally:
            for _, f in files:
                f.close()

    def get_job(self, job_id: str) -> AgentJobStatus:
        self.ensure_authenticated()

        url = f"{self._base_url}/agent/jobs/{job_id}"

        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url, headers=self._get_auth_headers())
            response.raise_for_status()

            data = response.json()
            return AgentJobStatus(
                job_id=data.get("job_id"),
                status=data.get("status"),
                result=data.get("result"),
                error=data.get("error"),
            )

    def delete_session(self, session_id: str) -> bool:
        """
        Delete an agent session.
        
        Args:
            session_id: Session ID to delete.
        
        Returns:
            True if deleted successfully.
        
        Raises:
            httpx.HTTPError: If the request fails.
        """
        self.ensure_authenticated()
        
        url = f"{self._base_url}/agent/{session_id}"
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.delete(url, headers=self._get_auth_headers())
            response.raise_for_status()
            
            data = response.json()
            return data.get("deleted", False)
    
    def list_sessions(self, user: Optional[str] = None) -> List[SessionSummary]:
        """
        List all agent sessions.
        
        Args:
            user: Optional user identifier to filter sessions.
        
        Returns:
            List of SessionSummary objects sorted by updated_at descending.
        
        Raises:
            httpx.HTTPError: If the request fails.
        """
        self.ensure_authenticated()
        
        url = f"{self._base_url}/agent/sessions"
        params = {}
        if user:
            params["user"] = user
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url, params=params, headers=self._get_auth_headers())
            response.raise_for_status()
            
            data = response.json()
            return [
                SessionSummary(
                    session_id=s.get("session_id", ""),
                    user=s.get("user"),
                    message_count=s.get("message_count", 0),
                    document_count=s.get("document_count", 0),
                    created_at=s.get("created_at", ""),
                    updated_at=s.get("updated_at", ""),
                    title=s.get("title"),
                )
                for s in data
            ]
    
    def get_history(self, session_id: str) -> Optional[SessionHistory]:
        """
        Get full session with complete message history.
        
        Args:
            session_id: Session ID to retrieve.
        
        Returns:
            SessionHistory object with full message history,
            or None if session not found.
        
        Raises:
            httpx.HTTPError: If the request fails (except 404).
        """
        self.ensure_authenticated()
        
        url = f"{self._base_url}/agent/{session_id}/history"
        
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

    def truncate_messages(self, session_id: str, from_index: int) -> Optional[dict]:
        """
        Truncate messages from a given index onwards.
        
        Args:
            session_id: Session ID to truncate messages from.
            from_index: Index from which to truncate (inclusive).
        
        Returns:
            Dict with truncated_count and remaining_count,
            or None if session not found.
        
        Raises:
            httpx.HTTPError: If the request fails (except 404).
        """
        self.ensure_authenticated()
        
        url = f"{self._base_url}/agent/{session_id}/truncate"
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.put(
                url,
                json={"from_index": from_index},
                headers=self._get_auth_headers(),
            )
            
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            
            data = response.json()
            return {
                "truncated_count": data.get("truncated_count", 0),
                "remaining_count": data.get("remaining_count", 0),
            }