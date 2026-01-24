"""
Chat file protocol for file upload compatibility.
"""

from typing import BinaryIO, Optional, Protocol, runtime_checkable


@runtime_checkable
class ChatFile(Protocol):
    """
    Protocol for file upload compatibility.
    
    Compatible with FastAPI's UploadFile and similar interfaces.
    """
    
    file: BinaryIO
    """File-like object with read() method."""
    
    filename: str
    """Original filename."""
    
    content_type: Optional[str]
    """MIME type of the file."""