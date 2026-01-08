"""
Pydantic schemas for API request/response models.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    """Response model for session creation."""
    
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(default="Session created successfully")


class SessionSummary(BaseModel):
    """Summary information for a session."""
    
    session_id: str = Field(..., description="Unique session identifier")
    user: Optional[str] = Field(default=None, description="User identifier")
    message_count: int = Field(..., description="Number of messages in session")
    document_count: int = Field(..., description="Number of documents in session")
    created_at: str = Field(..., description="Session creation timestamp (ISO format)")
    updated_at: str = Field(..., description="Session last update timestamp (ISO format)")


class MessageResponse(BaseModel):
    """Response model for a single message."""
    
    id: str = Field(..., description="Message identifier")
    role: str = Field(..., description="Message role (user, assistant, system)")
    content: str = Field(..., description="Message content text")
    timestamp: str = Field(..., description="Message timestamp (ISO format)")


class SessionHistoryResponse(BaseModel):
    """Response model for full session history."""
    
    session_id: str = Field(..., description="Unique session identifier")
    user: Optional[str] = Field(default=None, description="User identifier")
    instructions: Optional[str] = Field(default=None, description="System instructions")
    messages: List[MessageResponse] = Field(..., description="List of messages in session")
    document_count: int = Field(..., description="Number of documents in session")
    created_at: str = Field(..., description="Session creation timestamp (ISO format)")
    updated_at: str = Field(..., description="Session last update timestamp (ISO format)")


class DocumentInfo(BaseModel):
    """Document information model."""
    
    id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type/extension")


class DocumentListResponse(BaseModel):
    """Response model for listing session documents."""
    
    session_id: str = Field(..., description="Session identifier")
    documents: List[DocumentInfo] = Field(..., description="List of documents in session")
    count: int = Field(..., description="Total number of documents")


class DocumentDeleteResponse(BaseModel):
    """Response model for document deletion."""
    
    session_id: str = Field(..., description="Session identifier")
    document_id: str = Field(..., description="Deleted document identifier")
    deleted: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Status message")


class DeleteResponse(BaseModel):
    """Response model for session deletion."""
    
    session_id: str = Field(..., description="Deleted session identifier")
    deleted: bool = Field(..., description="Whether deletion was successful")
    message: str = Field(..., description="Status message")


class StreamChunkData(BaseModel):
    """Model for SSE stream chunk data."""
    
    text: str = Field(..., description="Response text chunk")
    finished: bool = Field(default=False, description="Whether streaming is complete")
    finish_reason: Optional[str] = Field(default=None, description="Reason for completion")
    usage: Optional[Dict[str, int]] = Field(default=None, description="Token usage statistics")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(default=None, description="Detailed error information")