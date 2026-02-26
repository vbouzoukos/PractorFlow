"""
API Tools client.

HTTP client for communicating with the PractorFlow API Tools API.
Provides CRUD operations for managing API tool configurations including
listing, creating, updating, deleting, and retrieving secrets.
"""

from typing import Optional

import httpx

from gui.api.client_data import AuthStatus
from gui.logger import get_logger

logger = get_logger("practorflow-client", level="INFO", log_file="logs/practorflow-client.log")


class ApiToolsClient:
    """
    HTTP client for the PractorFlow API Tools API.

    Provides synchronous methods for:
    - Listing all API tools
    - Creating API tools
    - Getting a single API tool
    - Updating API tools
    - Deleting API tools
    - Retrieving decrypted secrets
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        app_secret: Optional[str] = None,
        username: Optional[str] = None,
    ):
        """
        Initialize the API tools client.

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

        try:
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
        except httpx.HTTPError as e:
            logger.error(f"SessionClient.get_auth_status failed: {e}")
            raise

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

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(url, json=payload)
                response.raise_for_status()

                data = response.json()
                self._access_token = data["access_token"]
                return self._access_token
        except httpx.HTTPError as e:
            logger.error(f"SessionClient.authenticate failed: {e}")
            raise

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

    def list_tools(self) -> dict:
        """List all API tool configurations."""
        self.ensure_authenticated()

        url = f"{self._base_url}/api-tools"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(
                    url,
                    headers=self._get_auth_headers(),
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"ApiToolsClient.list_tools failed: {e}")
            raise

    def create_tool(self, data: dict) -> dict:
        """Create a new API tool configuration."""
        self.ensure_authenticated()

        url = f"{self._base_url}/api-tools"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    url,
                    json=data,
                    headers=self._get_auth_headers(),
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"ApiToolsClient.create_tool failed: {e}")
            raise

    def get_tool(self, tool_id: str) -> dict:
        """Get a single API tool configuration by ID."""
        self.ensure_authenticated()

        url = f"{self._base_url}/api-tools/{tool_id}"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(
                    url,
                    headers=self._get_auth_headers(),
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"ApiToolsClient.get_tool failed: {e}")
            raise

    def update_tool(self, tool_id: str, data: dict) -> dict:
        """Update an existing API tool configuration."""
        self.ensure_authenticated()

        url = f"{self._base_url}/api-tools/{tool_id}"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.put(
                    url,
                    json=data,
                    headers=self._get_auth_headers(),
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"ApiToolsClient.update_tool failed: {e}")
            raise

    def delete_tool(self, tool_id: str) -> bool:
        """Delete an API tool configuration."""
        self.ensure_authenticated()

        url = f"{self._base_url}/api-tools/{tool_id}"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.delete(
                    url,
                    headers=self._get_auth_headers(),
                )
                response.raise_for_status()
                return True
        except httpx.HTTPError as e:
            logger.error(f"ApiToolsClient.delete_tool failed: {e}")
            raise

    def get_secrets(self, tool_id: str) -> dict:
        """Get decrypted secrets for an API tool."""
        self.ensure_authenticated()

        url = f"{self._base_url}/api-tools/{tool_id}/secrets"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(
                    url,
                    headers=self._get_auth_headers(),
                )
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"ApiToolsClient.get_secrets failed: {e}")
            raise
