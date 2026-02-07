"""
API Configuration Manager.

Provides configuration management for the API layer, following the same
dataclass-based singleton pattern used in the core library.

Configuration is loaded from:
- config/api/options/config.env (non-sensitive settings)
- config/api/secrets/config.env (sensitive settings)

Authentication Mode Detection:
- OIDC_ISSUER_URL empty → Local mode
- OIDC_ISSUER_URL set → OIDC mode
- APP_SECRET empty (in local mode) → Open mode (no credential validation)

Environment Variables:
    JWT_ALGORITHM: JWT signing algorithm (default: HS256)
    JWT_TOKEN_EXPIRY_MINUTES: Token expiration in minutes (default: 60)
    JWT_SECRET_KEY: Secret key for signing JWTs
    APP_SECRET: Local authentication secret (empty = open mode)
    ADMIN_ENABLED: Enable admin functionality (default: false)
    ADMIN_SECRET: Secret for obtaining llm_admin permission (local mode only)
    OIDC_ISSUER_URL: OIDC provider issuer URL
    OIDC_AUDIENCE: Expected audience claim for OIDC tokens
    OIDC_CLIENT_ID: OIDC client identifier
    OIDC_CLIENT_SECRET: OIDC client secret
    CLEANUP_INTERVAL_MINUTES: Interval for orphan document cleanup (default: 60, 0 to disable)
"""

import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv


@dataclass
class JWTConfig:
    """JWT configuration settings."""

    algorithm: str = "HS256"
    secret_key: str = ""
    token_expiry_minutes: int = 60

    @property
    def is_configured(self) -> bool:
        """Check if JWT is properly configured with a secret key."""
        return bool(self.secret_key)


@dataclass
class OIDCConfig:
    """OIDC provider configuration settings."""

    issuer_url: str = ""
    audience: str = ""
    client_id: str = ""
    client_secret: str = ""

    @property
    def is_configured(self) -> bool:
        """Check if OIDC is properly configured."""
        return bool(self.issuer_url and self.client_id)


@dataclass
class AuthConfig:
    """Authentication configuration settings."""

    app_secret: str = ""
    admin_enabled: bool = False
    admin_secret: str = ""
    jwt: JWTConfig = field(default_factory=JWTConfig)
    oidc: OIDCConfig = field(default_factory=OIDCConfig)

    @property
    def is_oidc_mode(self) -> bool:
        """
        Check if authentication uses OIDC mode.

        OIDC mode is active when OIDC_ISSUER_URL is configured.
        """
        return bool(self.oidc.issuer_url)

    @property
    def is_local_mode(self) -> bool:
        """
        Check if authentication uses local mode.

        Local mode is active when OIDC_ISSUER_URL is empty.
        """
        return not self.is_oidc_mode

    @property
    def is_open_mode(self) -> bool:
        """
        Check if authentication is in open mode.

        Open mode is active when in local mode and APP_SECRET is empty.
        In this mode, JWT tokens are issued without credential validation.
        """
        return self.is_local_mode and not self.app_secret

    @property
    def requires_credentials(self) -> bool:
        """Check if authentication requires credential validation."""
        return not self.is_open_mode


@dataclass
class CleanupConfig:
    """Cleanup job configuration settings."""

    interval_minutes: int = 60

    @property
    def is_enabled(self) -> bool:
        """Check if cleanup job is enabled."""
        return self.interval_minutes > 0


@dataclass
class APIConfig:
    """
    API configuration container.

    Aggregates all API-specific configuration settings.
    """

    auth: AuthConfig = field(default_factory=AuthConfig)
    cleanup: CleanupConfig = field(default_factory=CleanupConfig)


def _load_auth_config() -> AuthConfig:
    """
    Load authentication configuration from environment variables.

    Returns:
        AuthConfig instance populated from environment.
    """
    jwt_config = JWTConfig(
        algorithm=os.getenv("JWT_ALGORITHM", "HS256"),
        secret_key=os.getenv("JWT_SECRET_KEY", ""),
        token_expiry_minutes=int(os.getenv("JWT_TOKEN_EXPIRY_MINUTES", "60")),
    )

    oidc_config = OIDCConfig(
        issuer_url=os.getenv("OIDC_ISSUER_URL", ""),
        audience=os.getenv("OIDC_AUDIENCE", ""),
        client_id=os.getenv("OIDC_CLIENT_ID", ""),
        client_secret=os.getenv("OIDC_CLIENT_SECRET", ""),
    )

    return AuthConfig(
        app_secret=os.getenv("APP_SECRET", "") or "",
        admin_enabled=str(os.getenv("ADMIN_ENABLED", "false")).lower() in ("true", "1", "yes"),
        admin_secret=os.getenv("ADMIN_SECRET", "") or "",
        jwt=jwt_config,
        oidc=oidc_config,
    )


def _load_cleanup_config() -> CleanupConfig:
    """
    Load cleanup configuration from environment variables.

    Returns:
        CleanupConfig instance populated from environment.
    """
    return CleanupConfig(
        interval_minutes=int(os.getenv("CLEANUP_INTERVAL_MINUTES", 0)),
    )


# Global singleton instance
apiConfiguration: Optional[APIConfig] = None


def load_api_configuration(config_path: str = "../config/api") -> None:
    """
    Load API configuration from .env files and initialize the global singleton.

    Args:
        config_path: Path to the API config folder containing options/ and secrets/.
                    Default is "../config/api" (for running from src/).
    """
    global apiConfiguration

    # Load environment files
    options_env = os.path.join(config_path, "options", "config.env")
    secrets_env = os.path.join(config_path, "secrets", "config.env")

    # Load options first, then secrets (secrets override options)
    if os.path.exists(options_env):
        load_dotenv(dotenv_path=options_env, override=True)

    if os.path.exists(secrets_env):
        load_dotenv(dotenv_path=secrets_env, override=True)

    # Create configuration instance
    apiConfiguration = APIConfig(
        auth=_load_auth_config(),
        cleanup=_load_cleanup_config(),
    )


def get_api_configuration() -> APIConfig:
    """
    Get the API configuration singleton.

    Returns:
        APIConfig instance.

    Raises:
        RuntimeError: If configuration has not been loaded.
    """
    if apiConfiguration is None:
        raise RuntimeError(
            "API configuration not loaded. Call load_api_configuration() first."
        )
    return apiConfiguration