"""
Chat API client.

HTTP client for communicating with the PractorFlow FastAPI backend.
Supports session management, message sending with file uploads,
and SSE streaming responses.
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


class ChatClient:
    """
    HTTP client for the PractorFlow Chat API.
    
    Provides synchronous methods for:
    - Starting chat sessions
    - Sending messages with optional file uploads
    - Streaming responses via SSE
    - Deleting sessions
    - Listing all sessions
    - Retrieving session history
    """
    
    def __init__(self, base_url: str, timeout: float = 30.0):
        """
        Initialize the chat client.
        
        Args:
            base_url: API base URL (e.g., "http://localhost:8000").
            timeout: Request timeout in seconds.
        """
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
    
    def start_session(self) -> str:
        """
        Start a new chat session.
        
        Returns:
            Session ID string.
        
        Raises:
            httpx.HTTPError: If the request fails.
        """
        url = f"{self._base_url}/chat"
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url)
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
        url = f"{self._base_url}/chat/{session_id}"
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.delete(url)
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
        url = f"{self._base_url}/chat/sessions"
        params = {}
        if user is not None:
            params["user"] = user
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url, params=params)
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
        url = f"{self._base_url}/chat/{session_id}/history"
        
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url)
            
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
                    files=files if files else None
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