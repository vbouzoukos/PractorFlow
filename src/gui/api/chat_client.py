"""
Chat API client.

HTTP client for communicating with the PractorFlow FastAPI backend.
Supports session management, message sending with file uploads,
and SSE streaming responses.
"""

import json
from typing import Iterator, Optional
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


class ChatClient:
    """
    HTTP client for the PractorFlow Chat API.
    
    Provides synchronous methods for:
    - Starting chat sessions
    - Sending messages with optional file uploads
    - Streaming responses via SSE
    - Deleting sessions
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