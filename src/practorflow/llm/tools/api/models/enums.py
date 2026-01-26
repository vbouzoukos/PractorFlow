"""
Enums for API Tool configuration.
"""

from enum import Enum


class AuthType(str, Enum):
    """Authentication types for API tools."""
    NONE = "none"
    API_KEY = "api_key"
    BEARER = "bearer"
    BASIC = "basic"


class AuthKeyLocation(str, Enum):
    """Location for API key authentication."""
    HEADER = "header"
    QUERY = "query"
    BODY = "body"


class RetryBackoff(str, Enum):
    """Backoff strategies for retry logic."""
    NONE = "none"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class HttpMethod(str, Enum):
    """Supported HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    PATCH = "PATCH"
    DELETE = "DELETE"


class BodyContentType(str, Enum):
    """Request body content types."""
    NONE = "none"
    JSON = "json"
    FORM = "form"
    MULTIPART = "multipart"


class ParamLocation(str, Enum):
    """Parameter locations in HTTP request."""
    PATH = "path"
    QUERY = "query"
    HEADER = "header"
    BODY = "body"


class ParamType(str, Enum):
    """Parameter data types."""
    STRING = "string"
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"


class ParseType(str, Enum):
    """Response parsing types."""
    JSON = "json"
    TEXT = "text"
    BINARY = "binary"