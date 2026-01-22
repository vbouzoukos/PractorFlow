import os
import pytest
from unittest.mock import patch

from api.config.api_settings import (
    JWTConfig,
    OIDCConfig,
    AuthConfig,
    CleanupConfig,
    APIConfig,
    _load_auth_config,
    _load_cleanup_config,
    load_api_configuration,
    get_api_configuration,
)


# ------------------------------------------------------------------
# JWTConfig
# ------------------------------------------------------------------

def test_jwt_config_is_configured():
    cfg = JWTConfig(secret_key="secret")
    assert cfg.is_configured is True

    cfg.secret_key = ""
    assert cfg.is_configured is False


# ------------------------------------------------------------------
# OIDCConfig
# ------------------------------------------------------------------

def test_oidc_config_is_configured():
    cfg = OIDCConfig(issuer_url="x", client_id="y")
    assert cfg.is_configured is True

    cfg.client_id = ""
    assert cfg.is_configured is False


# ------------------------------------------------------------------
# AuthConfig
# ------------------------------------------------------------------

def test_auth_config_modes():
    cfg = AuthConfig(app_secret="secret")
    cfg.oidc.issuer_url = ""

    assert cfg.is_local_mode is True
    assert cfg.is_oidc_mode is False
    assert cfg.is_open_mode is False
    assert cfg.requires_credentials is True


def test_auth_config_open_mode():
    cfg = AuthConfig(app_secret="")
    cfg.oidc.issuer_url = ""

    assert cfg.is_open_mode is True
    assert cfg.requires_credentials is False


def test_auth_config_oidc_mode():
    cfg = AuthConfig(app_secret="")
    cfg.oidc.issuer_url = "https://issuer"

    assert cfg.is_oidc_mode is True
    assert cfg.is_local_mode is False
    assert cfg.is_open_mode is False
    assert cfg.requires_credentials is True


# ------------------------------------------------------------------
# CleanupConfig
# ------------------------------------------------------------------

def test_cleanup_config():
    cfg = CleanupConfig(interval_minutes=10)
    assert cfg.is_enabled is True

    cfg.interval_minutes = 0
    assert cfg.is_enabled is False


# ------------------------------------------------------------------
# Load helpers
# ------------------------------------------------------------------

@patch.dict(os.environ, {
    "JWT_ALGORITHM": "HS512",
    "JWT_SECRET_KEY": "jwt-secret",
    "JWT_TOKEN_EXPIRY_MINUTES": "15",
    "OIDC_ISSUER_URL": "issuer",
    "OIDC_AUDIENCE": "aud",
    "OIDC_CLIENT_ID": "cid",
    "OIDC_CLIENT_SECRET": "csecret",
    "APP_SECRET": "app-secret",
})
def test_load_auth_config_from_env():
    cfg = _load_auth_config()

    assert cfg.app_secret == "app-secret"
    assert cfg.jwt.algorithm == "HS512"
    assert cfg.jwt.secret_key == "jwt-secret"
    assert cfg.jwt.token_expiry_minutes == 15
    assert cfg.oidc.issuer_url == "issuer"
    assert cfg.oidc.client_id == "cid"


@patch.dict(os.environ, {"CLEANUP_INTERVAL_MINUTES": "5"})
def test_load_cleanup_config_from_env():
    cfg = _load_cleanup_config()
    assert cfg.interval_minutes == 5


# ------------------------------------------------------------------
# load_api_configuration
# ------------------------------------------------------------------

def test_load_api_configuration_with_env_files(tmp_path):
    options = tmp_path / "options"
    secrets = tmp_path / "secrets"
    options.mkdir()
    secrets.mkdir()

    (options / "config.env").write_text("APP_SECRET=opt\n")
    (secrets / "config.env").write_text("APP_SECRET=sec\n")

    with patch("api.config.api_settings.load_dotenv") as load_dotenv:
        load_api_configuration(str(tmp_path))

    # both dotenv files attempted
    assert load_dotenv.call_count == 2

    cfg = get_api_configuration()
    assert isinstance(cfg, APIConfig)


def test_load_api_configuration_without_files(tmp_path):
    with patch("api.config.api_settings.load_dotenv") as load_dotenv:
        load_api_configuration(str(tmp_path))

    load_dotenv.assert_not_called()
    cfg = get_api_configuration()
    assert isinstance(cfg, APIConfig)


# ------------------------------------------------------------------
# get_api_configuration
# ------------------------------------------------------------------

def test_get_api_configuration_not_loaded(monkeypatch):
    monkeypatch.setattr(
        "api.config.api_settings.apiConfiguration",
        None,
    )

    with pytest.raises(RuntimeError):
        get_api_configuration()
