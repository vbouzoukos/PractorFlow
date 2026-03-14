"""Tests for ApiToolFactory and singleton functions."""

import pytest
from unittest.mock import MagicMock, patch

import practorflow.llm.tools.api.factory as _factory_module
from practorflow.llm.tools.api.factory import (
    ApiToolFactory,
    initialize_factory,
    get_factory,
    is_factory_initialized,
)
from practorflow.llm.tools.api.models import ApiToolConfig


@pytest.fixture(autouse=True)
def reset_singleton():
    original = _factory_module._instance
    _factory_module._instance = None
    yield
    _factory_module._instance = original


def _make_config(**kwargs):
    defaults = {
        "user_id": "user1",
        "name": "my_tool",
        "base_url": "https://example.com",
        "path": "/api/data",
        "description": "A test tool",
        "keywords": ["test"],
    }
    defaults.update(kwargs)
    return ApiToolConfig(**defaults)


# -----------------------------------
# singleton helpers
# -----------------------------------


def test_is_factory_initialized_false_when_not_initialized():
    assert is_factory_initialized() is False


def test_is_factory_initialized_true_after_init():
    initialize_factory(MagicMock())
    assert is_factory_initialized() is True


def test_initialize_factory_returns_instance():
    store = MagicMock()
    factory = initialize_factory(store)
    assert isinstance(factory, ApiToolFactory)


def test_initialize_factory_raises_if_already_initialized():
    initialize_factory(MagicMock())
    with pytest.raises(RuntimeError, match="already initialized"):
        initialize_factory(MagicMock())


def test_get_factory_raises_when_not_initialized():
    with pytest.raises(RuntimeError, match="not initialized"):
        get_factory()


def test_get_factory_returns_instance_when_initialized():
    store = MagicMock()
    factory = initialize_factory(store)
    assert get_factory() is factory


# -----------------------------------
# ApiToolFactory.__init__
# -----------------------------------


def test_factory_init_stores_store():
    store = MagicMock()
    factory = ApiToolFactory(store)
    assert factory._store is store
    assert factory._rate_limiters == {}


# -----------------------------------
# create_tool
# -----------------------------------


def test_create_tool_returns_api_tool():
    store = MagicMock()
    factory = ApiToolFactory(store)
    config = _make_config()

    mock_tool = MagicMock()
    mock_tool.expired = False

    with patch("practorflow.llm.tools.api.factory.ApiTool", return_value=mock_tool) as mock_cls:
        tool = factory.create_tool(config)

    mock_cls.assert_called_once()
    assert tool is mock_tool


def test_create_tool_creates_rate_limiter_for_new_tool():
    store = MagicMock()
    factory = ApiToolFactory(store)
    config = _make_config()

    mock_tool = MagicMock()
    mock_tool.expired = False

    with patch("practorflow.llm.tools.api.factory.ApiTool", return_value=mock_tool):
        factory.create_tool(config)

    assert config.tool_id in factory._rate_limiters


def test_create_tool_reuses_existing_rate_limiter():
    store = MagicMock()
    factory = ApiToolFactory(store)
    config = _make_config()

    mock_tool = MagicMock()
    mock_tool.expired = False

    with patch("practorflow.llm.tools.api.factory.ApiTool", return_value=mock_tool):
        factory.create_tool(config)
        rl_first = factory._rate_limiters[config.tool_id]
        factory.create_tool(config)
        rl_second = factory._rate_limiters[config.tool_id]

    assert rl_first is rl_second


# -----------------------------------
# create_tools_for_user
# -----------------------------------


def test_create_tools_for_user_returns_tools():
    store = MagicMock()
    config = _make_config()
    store.list_filtered.return_value = [config]

    factory = ApiToolFactory(store)

    mock_tool = MagicMock()
    mock_tool.expired = False

    with patch("practorflow.llm.tools.api.factory.ApiTool", return_value=mock_tool):
        tools = factory.create_tools_for_user("user1")

    assert len(tools) == 1
    assert tools[0] is mock_tool


def test_create_tools_for_user_skips_expired_tools():
    store = MagicMock()
    config = _make_config()
    store.list_filtered.return_value = [config]

    factory = ApiToolFactory(store)

    mock_tool = MagicMock()
    mock_tool.expired = True

    with patch("practorflow.llm.tools.api.factory.ApiTool", return_value=mock_tool):
        tools = factory.create_tools_for_user("user1")

    assert tools == []


def test_create_tools_for_user_empty_store():
    store = MagicMock()
    store.list_filtered.return_value = []

    factory = ApiToolFactory(store)

    with patch("practorflow.llm.tools.api.factory.ApiTool") as mock_cls:
        tools = factory.create_tools_for_user("user1")

    mock_cls.assert_not_called()
    assert tools == []


# -----------------------------------
# invalidate_rate_limiter
# -----------------------------------


def test_invalidate_rate_limiter_removes_existing():
    store = MagicMock()
    factory = ApiToolFactory(store)
    config = _make_config()

    mock_tool = MagicMock()
    mock_tool.expired = False

    with patch("practorflow.llm.tools.api.factory.ApiTool", return_value=mock_tool):
        factory.create_tool(config)

    assert config.tool_id in factory._rate_limiters
    factory.invalidate_rate_limiter(config.tool_id)
    assert config.tool_id not in factory._rate_limiters


def test_invalidate_rate_limiter_nonexistent_is_noop():
    store = MagicMock()
    factory = ApiToolFactory(store)
    # Should not raise
    factory.invalidate_rate_limiter("nonexistent-id")
