import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timezone

from api.auth.jwt_handler import (
    JWTHandler,
    JWTError,
    JWTExpiredError,
    JWTInvalidError,
)
from api.auth.schemas import UserContext
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError


@pytest.fixture()
def jwt_config():
    cfg = Mock()
    cfg.algorithm = "HS256"
    cfg.secret_key = "secret"
    cfg.token_expiry_minutes = 5
    return cfg


@pytest.fixture()
def handler(jwt_config):
    return JWTHandler(jwt_config)


# ------------------------------------------------------------------
# Constructors (MUST be executed for coverage)
# ------------------------------------------------------------------

def test_jwt_error_init():
    err = JWTError("e", "d")
    assert err.error == "e"
    assert err.description == "d"


def test_jwt_expired_error_init():
    err = JWTExpiredError()
    assert err.error == "token_expired"
    assert "expired" in err.description


def test_jwt_invalid_error_init():
    err = JWTInvalidError("bad")
    assert err.error == "invalid_token"
    assert err.description == "bad"


# ------------------------------------------------------------------
# expiry_seconds
# ------------------------------------------------------------------

def test_expiry_seconds(handler):
    assert handler.expiry_seconds == 300


# ------------------------------------------------------------------
# create_token
# ------------------------------------------------------------------

def test_create_token_success(handler):
    with patch("api.auth.jwt_handler.jwt.encode", return_value="jwt") as enc:
        token = handler.create_token("user-1")

    assert token == "jwt"
    enc.assert_called_once()
    payload = enc.call_args[0][0]

    # Force payload fields to be executed
    assert payload["sub"] == "user-1"
    assert payload["iss"] == "practorflow-api"
    assert "iat" in payload
    assert "exp" in payload


def test_create_token_no_secret():
    cfg = Mock()
    cfg.algorithm = "HS256"
    cfg.secret_key = None
    cfg.token_expiry_minutes = 1

    handler = JWTHandler(cfg)

    with pytest.raises(JWTError) as exc:
        handler.create_token("x")

    assert exc.value.error == "configuration_error"


def test_create_token_encode_exception(handler):
    with patch("api.auth.jwt_handler.jwt.encode", side_effect=Exception("boom")):
        with pytest.raises(JWTError) as exc:
            handler.create_token("user")

    assert exc.value.error == "token_creation_failed"
    assert "boom" in exc.value.description


# ------------------------------------------------------------------
# validate_token
# ------------------------------------------------------------------

def test_validate_token_success(handler):
    payload = {
        "sub": "user-1",
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc),
        "iss": "practorflow-api",
    }

    with patch("api.auth.jwt_handler.jwt.decode", return_value=payload) as dec:
        ctx = handler.validate_token("token")

    assert isinstance(ctx, UserContext)
    assert ctx.user_id == "user-1"
    assert ctx.is_authenticated is True

    # Force execution of decode options branch
    assert "require" in dec.call_args.kwargs["options"]


def test_validate_token_missing_sub(handler):
    payload = {
        "iat": datetime.now(timezone.utc),
        "exp": datetime.now(timezone.utc),
    }

    with patch("api.auth.jwt_handler.jwt.decode", return_value=payload):
        with pytest.raises(JWTInvalidError) as exc:
            handler.validate_token("token")

    assert "subject" in exc.value.description.lower()


def test_validate_token_expired(handler):
    with patch(
        "api.auth.jwt_handler.jwt.decode",
        side_effect=ExpiredSignatureError(),
    ):
        with pytest.raises(JWTExpiredError):
            handler.validate_token("token")


def test_validate_token_invalid(handler):
    with patch(
        "api.auth.jwt_handler.jwt.decode",
        side_effect=InvalidTokenError("bad"),
    ):
        with pytest.raises(JWTInvalidError) as exc:
            handler.validate_token("token")

    assert "bad" in exc.value.description


def test_validate_token_generic_exception(handler):
    with patch(
        "api.auth.jwt_handler.jwt.decode",
        side_effect=Exception("boom"),
    ):
        with pytest.raises(JWTInvalidError) as exc:
            handler.validate_token("token")

    assert "boom" in exc.value.description


def test_validate_token_no_secret():
    cfg = Mock()
    cfg.algorithm = "HS256"
    cfg.secret_key = None
    cfg.token_expiry_minutes = 1

    handler = JWTHandler(cfg)

    with pytest.raises(JWTError) as exc:
        handler.validate_token("token")

    assert exc.value.error == "configuration_error"
