"""
Session API client.

HTTP client for communicating with the PractorFlow Session API.
Provides unified session management operations including listing,
history retrieval, deletion, truncation, and document management.
"""

from typing import List, Optional

import httpx

from gui.api.client_data import (
    SessionSummary,
    MessageInfo,
    SessionHistory,
    DocumentInfo,
    AuthStatus,
)


class SessionClient:
    """
    HTTP client for the PractorFlow Session API.
    
    Provides synchronous methods for:
    - Listing all sessions
    - Retrieving session history
    - Deleting sessions
    - Truncating session messages
    - Listing session documents
    - Deleting session documents
    """
    
    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        app_secret: Optional[str] = None,
        username: Optional[str] = None,
    ):
        """
        Initialize the session client.
        
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
    
    def list_sessions(
        self,
        user: Optional[str] = None,
        session_type: Optional[str] = None,
    ) -> List[SessionSummary]:
        """
        List all sessions.
        
        Args:
            user: Optional user identifier to filter sessions.
            session_type: Optional session type to filter (e.g., 'agent').
        
        Returns:
            List of SessionSummary objects sorted by updated_at descending.
        
        Raises:
            httpx.HTTPError: If the request fails.
        """
        self.ensure_authenticated()
        
        url = f"{self._base_url}/sessions"
        params = {}
        if user is not None:
            params["user"] = user
        if session_type is not None:
            params["session_type"] = session_type
        
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
                    title=item.get("title"),
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
        
        url = f"{self._base_url}/sessions/{session_id}/history"
        
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
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session.
        
        Args:
            session_id: Session ID to delete.
        
        Returns:
            True if deleted successfully.
        
        Raises:
            httpx.HTTPError: If the request fails.
        """
        self.ensure_authenticated()
        
        url = f"{self._base_url}/sessions/{session_id}"
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.delete(url, headers=self._get_auth_headers())
            response.raise_for_status()
            
            data = response.json()
            return data.get("deleted", False)
    
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
        
        url = f"{self._base_url}/sessions/{session_id}/truncate"
        
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
    
    def list_session_documents(self, session_id: str) -> Optional[List[DocumentInfo]]:
        """
        List all documents in a session.
        
        Args:
            session_id: Session ID to retrieve documents for.
        
        Returns:
            List of DocumentInfo objects, or None if session not found.
        
        Raises:
            httpx.HTTPError: If the request fails (except 404).
        """
        self.ensure_authenticated()
        
        url = f"{self._base_url}/sessions/{session_id}/documents"
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url, headers=self._get_auth_headers())
            
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            
            data = response.json()
            return [
                DocumentInfo(
                    id=doc.get("id", ""),
                    filename=doc.get("filename", ""),
                    file_type=doc.get("file_type", "unknown"),
                )
                for doc in data.get("documents", [])
            ]
    
    def delete_session_document(
        self,
        session_id: str,
        document_id: str,
    ) -> Optional[bool]:
        """
        Delete a document from a session.
        
        Args:
            session_id: Session ID containing the document.
            document_id: Document ID to delete.
        
        Returns:
            True if deleted successfully.
            None if session or document not found.
        
        Raises:
            httpx.HTTPError: If the request fails (except 404).
        """
        self.ensure_authenticated()
        
        url = f"{self._base_url}/sessions/{session_id}/documents/{document_id}"
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.delete(url, headers=self._get_auth_headers())
            
            if response.status_code == 404:
                return None
            
            response.raise_for_status()
            
            data = response.json()
            return data.get("deleted", False)