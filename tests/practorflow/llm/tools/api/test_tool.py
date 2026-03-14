"""Tests for ApiTool."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from practorflow.llm.tools.api.models.models import ApiToolConfig
from practorflow.llm.tools.api.models.enums import AuthType, ParamLocation, ParamType
from practorflow.llm.tools.api.tool import ApiTool
from practorflow.llm.tools.api.request.rate_limiter import RateLimiter
from practorflow.llm.tools import ToolResult


def _make_config(**kwargs):
    defaults = {
        "user_id": "user1",
        "name": "my_tool",
        "base_url": "https://example.com",
        "path": "/api/data",
        "description": "A test tool",
        "keywords": ["test"],
        "retry_max_attempts": 1,
    }
    defaults.update(kwargs)
    return ApiToolConfig(**defaults)


def _make_param(name="q", location=ParamLocation.QUERY):
    from practorflow.llm.tools.api.models.models import ToolParameter as ConfigToolParameter
    return ConfigToolParameter(
        name=name,
        location=location,
        type=ParamType.STRING,
        description="param",
    )


# ---------------------------------------------------------------------------
# __init__ — construction
# ---------------------------------------------------------------------------


def test_init_no_auth_sets_expired_false():
    config = _make_config()
    tool = ApiTool(config=config)
    assert tool.expired is False
    assert tool._decrypted_secret is None
    assert tool._decrypted_username is None
    assert tool._decrypted_password is None


def test_init_uses_provided_rate_limiter():
    config = _make_config()
    rl = RateLimiter(rpm=0)
    tool = ApiTool(config=config, rate_limiter=rl)
    assert tool._rate_limiter is rl


def test_init_creates_rate_limiter_when_not_provided():
    config = _make_config()
    tool = ApiTool(config=config)
    assert tool._rate_limiter is not None


def test_init_expired_secret_sets_expired_flag():
    past = datetime.now() - timedelta(hours=1)
    config = _make_config(auth_secret_expires_at=past, auth_secret="enc:tok", auth_type=AuthType.BEARER)
    tool = ApiTool(config=config)
    assert tool.expired is True


def test_init_auth_bearer_decrypts_secret():
    config = _make_config(auth_type=AuthType.BEARER, auth_secret="enc:tok")
    mock_enc = MagicMock()
    mock_enc.decrypt.return_value = "plain-token"

    with patch("practorflow.llm.tools.api.tool.get_encryption_service", return_value=mock_enc):
        tool = ApiTool(config=config)

    assert tool._decrypted_secret == "plain-token"
    assert tool.expired is False


def test_init_auth_basic_decrypts_username_and_password():
    config = _make_config(
        auth_type=AuthType.BASIC,
        auth_username="enc:user",
        auth_password="enc:pass",
    )
    mock_enc = MagicMock()
    mock_enc.decrypt.side_effect = lambda v: v.replace("enc:", "")

    with patch("practorflow.llm.tools.api.tool.get_encryption_service", return_value=mock_enc):
        tool = ApiTool(config=config)

    assert tool._decrypted_username == "user"
    assert tool._decrypted_password == "pass"


def test_init_decrypt_failure_sets_expired():
    config = _make_config(auth_type=AuthType.BEARER, auth_secret="bad")
    mock_enc = MagicMock()
    mock_enc.decrypt.side_effect = Exception("bad key")

    with patch("practorflow.llm.tools.api.tool.get_encryption_service", return_value=mock_enc):
        tool = ApiTool(config=config)

    assert tool.expired is True


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


def test_name_property():
    config = _make_config(name="awesome_tool")
    tool = ApiTool(config=config)
    assert tool.name == "awesome_tool"


def test_description_property():
    config = _make_config(description="Does something useful")
    tool = ApiTool(config=config)
    assert tool.description == "Does something useful"


def test_parameters_property_empty():
    config = _make_config()
    tool = ApiTool(config=config)
    assert tool.parameters == []


def test_parameters_property_converts_params():
    param = _make_param("search", ParamLocation.QUERY)
    config = _make_config(parameters=[param])
    tool = ApiTool(config=config)
    params = tool.parameters
    assert len(params) == 1
    assert params[0].name == "search"
    assert params[0].type == "string"


def test_parameters_integer_type():
    from practorflow.llm.tools.api.models.models import ToolParameter as ConfigToolParameter
    param = ConfigToolParameter(name="count", location=ParamLocation.QUERY, type=ParamType.INTEGER, description="")
    config = _make_config(parameters=[param])
    tool = ApiTool(config=config)
    assert tool.parameters[0].type == "integer"


def test_parameters_boolean_type():
    from practorflow.llm.tools.api.models.models import ToolParameter as ConfigToolParameter
    param = ConfigToolParameter(name="flag", location=ParamLocation.QUERY, type=ParamType.BOOLEAN, description="")
    config = _make_config(parameters=[param])
    tool = ApiTool(config=config)
    assert tool.parameters[0].type == "boolean"


# ---------------------------------------------------------------------------
# execute
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_success():
    config = _make_config()
    tool = ApiTool(config=config)

    success_result = ToolResult(success=True, data={"key": "val"})
    tool._execute_request = AsyncMock(return_value=success_result)
    tool._rate_limiter.acquire = AsyncMock()

    mock_request_data = MagicMock()
    tool._builder.build = MagicMock(return_value=mock_request_data)

    result = await tool.execute(q="hello")

    assert result.success is True
    assert result.data == {"key": "val"}


@pytest.mark.asyncio
async def test_execute_returns_last_result_when_all_fail():
    config = _make_config(retry_max_attempts=2)
    tool = ApiTool(config=config)

    fail_result = ToolResult(
        success=False,
        error="server error",
        metadata={"status_code": 500, "category": "http_error"},
    )
    tool._execute_request = AsyncMock(return_value=fail_result)
    tool._rate_limiter.acquire = AsyncMock()
    tool._retry_handler.wait = AsyncMock()
    tool._builder.build = MagicMock(return_value=MagicMock())

    result = await tool.execute()

    assert result.success is False
    assert result.error == "server error"


@pytest.mark.asyncio
async def test_execute_retries_on_retryable_status():
    config = _make_config(retry_max_attempts=2)
    tool = ApiTool(config=config)

    fail_result = ToolResult(
        success=False,
        error="retry me",
        metadata={"status_code": 500, "category": "http_error"},
    )
    success_result = ToolResult(success=True, data="ok")
    tool._execute_request = AsyncMock(side_effect=[fail_result, success_result])
    tool._rate_limiter.acquire = AsyncMock()
    tool._retry_handler.wait = AsyncMock()
    tool._retry_handler.is_retryable = MagicMock(return_value=True)
    tool._builder.build = MagicMock(return_value=MagicMock())

    result = await tool.execute()

    assert result.success is True
    assert tool._execute_request.call_count == 2


@pytest.mark.asyncio
async def test_execute_no_retry_when_not_retryable():
    config = _make_config(retry_max_attempts=3)
    tool = ApiTool(config=config)

    fail_result = ToolResult(
        success=False,
        error="not retryable",
        metadata={"status_code": 400, "category": "http_error"},
    )
    tool._execute_request = AsyncMock(return_value=fail_result)
    tool._rate_limiter.acquire = AsyncMock()
    tool._retry_handler.wait = AsyncMock()
    tool._retry_handler.is_retryable = MagicMock(return_value=False)
    tool._builder.build = MagicMock(return_value=MagicMock())

    result = await tool.execute()

    assert result.success is False
    # Should break after first attempt
    assert tool._execute_request.call_count == 1


# ---------------------------------------------------------------------------
# _execute_request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_execute_request_success():
    config = _make_config()
    tool = ApiTool(config=config)

    mock_response = MagicMock()
    mock_parsed = MagicMock(success=True, data={"result": 1}, status_code=200, error=None)
    tool._parser.parse = MagicMock(return_value=mock_parsed)

    mock_request_data = MagicMock()
    mock_request_data.method = "GET"
    mock_request_data.url = "https://example.com/api/data"
    mock_request_data.headers = {}
    mock_request_data.query_params = {}
    mock_request_data.body = None
    mock_request_data.content_type = None

    with patch("practorflow.llm.tools.api.tool.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await tool._execute_request(mock_request_data, attempt=1)

    assert result.success is True
    assert result.data == {"result": 1}
    assert result.metadata["status_code"] == 200
    assert result.metadata["attempt"] == 1


@pytest.mark.asyncio
async def test_execute_request_failure_response():
    config = _make_config()
    tool = ApiTool(config=config)

    mock_response = MagicMock()
    mock_parsed = MagicMock(success=False, data=None, status_code=404, error="not found")
    tool._parser.parse = MagicMock(return_value=mock_parsed)

    mock_request_data = MagicMock()
    mock_request_data.method = "GET"
    mock_request_data.url = "https://example.com/api/data"
    mock_request_data.headers = {}
    mock_request_data.query_params = {}
    mock_request_data.body = None
    mock_request_data.content_type = None

    with patch("practorflow.llm.tools.api.tool.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False
        mock_client.request = AsyncMock(return_value=mock_response)
        mock_client_cls.return_value = mock_client

        result = await tool._execute_request(mock_request_data, attempt=1)

    assert result.success is False
    assert result.error == "not found"
    assert result.metadata["category"] == "http_error"


@pytest.mark.asyncio
async def test_execute_request_timeout_exception():
    import httpx
    config = _make_config()
    tool = ApiTool(config=config)

    mock_request_data = MagicMock()
    mock_request_data.method = "GET"
    mock_request_data.url = "https://example.com/api/data"
    mock_request_data.headers = {}
    mock_request_data.query_params = {}
    mock_request_data.body = None
    mock_request_data.content_type = None

    with patch("practorflow.llm.tools.api.tool.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False
        mock_client.request = AsyncMock(side_effect=httpx.TimeoutException("timed out"))
        mock_client_cls.return_value = mock_client

        result = await tool._execute_request(mock_request_data, attempt=1)

    assert result.success is False
    assert result.metadata["category"] == "timeout_error"
    assert "timeout" in result.error.lower()


@pytest.mark.asyncio
async def test_execute_request_connect_error():
    import httpx
    config = _make_config()
    tool = ApiTool(config=config)

    mock_request_data = MagicMock()
    mock_request_data.method = "GET"
    mock_request_data.url = "https://example.com/api/data"
    mock_request_data.headers = {}
    mock_request_data.query_params = {}
    mock_request_data.body = None
    mock_request_data.content_type = None

    with patch("practorflow.llm.tools.api.tool.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False
        mock_client.request = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_client_cls.return_value = mock_client

        result = await tool._execute_request(mock_request_data, attempt=1)

    assert result.success is False
    assert result.metadata["category"] == "connection_error"
    assert "Connection failed" in result.error


@pytest.mark.asyncio
async def test_execute_request_unexpected_exception():
    config = _make_config()
    tool = ApiTool(config=config)

    mock_request_data = MagicMock()
    mock_request_data.method = "GET"
    mock_request_data.url = "https://example.com/api/data"
    mock_request_data.headers = {}
    mock_request_data.query_params = {}
    mock_request_data.body = None
    mock_request_data.content_type = None

    with patch("practorflow.llm.tools.api.tool.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = False
        mock_client.request = AsyncMock(side_effect=RuntimeError("boom"))
        mock_client_cls.return_value = mock_client

        result = await tool._execute_request(mock_request_data, attempt=2)

    assert result.success is False
    assert result.metadata["category"] == "unexpected_error"
    assert result.metadata["attempt"] == 2
    assert "boom" in result.error
