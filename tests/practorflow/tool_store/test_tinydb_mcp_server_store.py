"""Tests for TinyDBMCPServerStore."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

_mock_db_instance = MagicMock()
_mock_table_instance = MagicMock()
_mock_query_instance = MagicMock()

_mock_db_instance.table.return_value = _mock_table_instance

_tinydb_patcher = patch("practorflow.tool_store.tinydb_mcp_server_store.TinyDB", return_value=_mock_db_instance)
_query_patcher = patch("practorflow.tool_store.tinydb_mcp_server_store.Query", return_value=_mock_query_instance)

_tinydb_patcher.start()
_query_patcher.start()

from practorflow.tool_store.tinydb_mcp_server_store import TinyDBMCPServerStore
from practorflow.llm.tools.mcp.types import (
    HttpConfig,
    MCPServerConfig,
    MCPToolConfig,
    StdioConfig,
    TransportType,
)


@pytest.fixture(autouse=True)
def reset_mocks():
    _mock_db_instance.reset_mock()
    _mock_table_instance.reset_mock()
    _mock_query_instance.reset_mock()
    _mock_db_instance.table.return_value = _mock_table_instance
    yield


@pytest.fixture
def store():
    return TinyDBMCPServerStore(db_path="test.json")


def _make_config(**kwargs):
    defaults = {
        "name": "test_server",
        "transport": TransportType.STDIO,
        "stdio_config": StdioConfig(command="python"),
    }
    defaults.update(kwargs)
    return MCPServerConfig(**defaults)


def test_init_creates_db_and_table():
    TinyDBMCPServerStore(db_path="test.json")
    _mock_db_instance.table.assert_called_with("mcp_servers")


def test_create_inserts_config(store):
    config = _make_config()

    result = store.create(config)

    _mock_table_instance.insert.assert_called_once()
    assert result is config


def test_get_returns_config_when_found(store):
    config = _make_config()
    data = store._serialize(config)
    _mock_table_instance.get.return_value = data

    result = store.get("some-id")

    assert isinstance(result, MCPServerConfig)


def test_get_returns_none_when_not_found(store):
    _mock_table_instance.get.return_value = None

    result = store.get("missing-id")

    assert result is None


def test_update_returns_none_when_not_found(store):
    _mock_table_instance.get.return_value = None
    config = _make_config()

    result = store.update(config)

    assert result is None
    _mock_table_instance.update.assert_not_called()


def test_update_updates_existing_config(store):
    _mock_table_instance.get.return_value = {"server_id": "s1"}
    config = _make_config()

    result = store.update(config)

    _mock_table_instance.update.assert_called_once()
    assert result is config


def test_delete_returns_true_when_removed(store):
    _mock_table_instance.remove.return_value = ["s1"]

    result = store.delete("s1")

    assert result is True


def test_delete_returns_false_when_not_found(store):
    _mock_table_instance.remove.return_value = []

    result = store.delete("missing")

    assert result is False


def test_list_returns_all_configs(store):
    config = _make_config()
    data = store._serialize(config)
    _mock_table_instance.all.return_value = [data]

    result = store.list()

    assert len(result) == 1
    assert isinstance(result[0], MCPServerConfig)


def test_close_closes_db(store):
    store.close()
    _mock_db_instance.close.assert_called_once()


def test_serialize_with_stdio_config(store):
    config = _make_config(stdio_config=StdioConfig(command="python", args=["-m", "mymod"]))

    data = store._serialize(config)

    assert data["stdio_config"] is not None
    assert data["stdio_config"]["command"] == "python"
    assert data["http_config"] is None


def test_serialize_with_http_config(store):
    config = _make_config(
        transport=TransportType.STREAMABLE_HTTP,
        stdio_config=None,
        http_config=HttpConfig(url="https://example.com/mcp"),
    )

    data = store._serialize(config)

    assert data["http_config"] is not None
    assert data["http_config"]["url"] == "https://example.com/mcp"
    assert data["stdio_config"] is None


def test_serialize_with_tools(store):
    tool = MCPToolConfig(name="my_tool")
    config = _make_config(tools=[tool])

    data = store._serialize(config)

    assert len(data["tools"]) == 1
    assert data["tools"][0]["name"] == "my_tool"


def test_deserialize_with_stdio_config(store):
    config = _make_config(stdio_config=StdioConfig(command="python"))
    data = store._serialize(config)

    result = store._deserialize(data)

    assert isinstance(result.stdio_config, StdioConfig)
    assert result.http_config is None


def test_deserialize_with_http_config(store):
    config = _make_config(
        transport=TransportType.STREAMABLE_HTTP,
        stdio_config=None,
        http_config=HttpConfig(url="https://example.com/mcp"),
    )
    data = store._serialize(config)

    result = store._deserialize(data)

    assert isinstance(result.http_config, HttpConfig)
    assert result.stdio_config is None


def test_deserialize_with_tools(store):
    tool = MCPToolConfig(name="my_tool")
    config = _make_config(tools=[tool])
    data = store._serialize(config)

    result = store._deserialize(data)

    assert len(result.tools) == 1
    assert result.tools[0].name == "my_tool"
