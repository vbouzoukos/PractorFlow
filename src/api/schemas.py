"""
Pydantic schemas for API request/response models.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field


class SessionResponse(BaseModel):
    """Response model for session creation."""
    
    session_id: str = Field(..., description="Unique session identifier")
    message: str = Field(default="Session created successfully")


class DocumentInfo(BaseModel):
    """Document information model."""
    
    id: str = Field(..., description="Document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File type/extension")


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