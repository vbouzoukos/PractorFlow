"""
Tools client.

HTTP client for communicating with the PractorFlow Tools API.
Provides operations for listing tools and updating preferences.
"""

from typing import Optional

import httpx

from gui.api.client_data import AuthStatus
from gui.logger import get_logger

logger = get_logger("practorflow-client", level="INFO", log_file="logs/practorflow-client.log")


class ToolsClient:
    """
    HTTP client for the PractorFlow Tools API.

    Provides synchronous methods for:
    - Listing all tools with enabled state
    - Updating a single tool preference
    """

    def __init__(
        self,
        base_url: str,
        timeout: float = 30.0,
        app_secret: Optional[str] = None,
        username: Optional[str] = None,
    ):
        """
        Initialize the tools client.

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
            logger.error(f"ToolsClient.get_auth_status failed: {e}")
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
            logger.error(f"ToolsClient.authenticate failed: {e}")
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
        """List all tools with enabled state. GET /tools"""
        self.ensure_authenticated()

        url = f"{self._base_url}/tools"

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.get(url, headers=self._get_auth_headers())
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"ToolsClient.list_tools failed: {e}")
            raise

    def update_preference(
        self,
        type: str,
        id: str,
        enabled: bool,
        server_id: Optional[str] = None,
    ) -> dict:
        """Update a single tool preference. PUT /tools/preferences"""
        self.ensure_authenticated()

        url = f"{self._base_url}/tools/preferences"

        payload: dict = {"type": type, "id": id, "enabled": enabled}
        if server_id is not None:
            payload["server_id"] = server_id

        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.put(url, json=payload, headers=self._get_auth_headers())
                response.raise_for_status()
                return response.json()
        except httpx.HTTPError as e:
            logger.error(f"ToolsClient.update_preference failed: {e}")
            raise
