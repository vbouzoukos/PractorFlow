"""Tests for TinyDBApiToolStore."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

_mock_db_instance = MagicMock()
_mock_table_instance = MagicMock()
_mock_query_instance = MagicMock()

_mock_db_instance.table.return_value = _mock_table_instance

_tinydb_patcher = patch("practorflow.tool_store.tinydb_api_store.TinyDB", return_value=_mock_db_instance)
_query_patcher = patch("practorflow.tool_store.tinydb_api_store.Query", return_value=_mock_query_instance)

_tinydb_patcher.start()
_query_patcher.start()

from practorflow.tool_store.tinydb_api_store import TinyDBApiToolStore
from practorflow.llm.tools.api.models.models import ApiToolConfig


@pytest.fixture(autouse=True)
def reset_mocks():
    _mock_db_instance.reset_mock()
    _mock_table_instance.reset_mock()
    _mock_query_instance.reset_mock()
    _mock_db_instance.table.return_value = _mock_table_instance
    yield


@pytest.fixture
def store():
    return TinyDBApiToolStore(db_path="test.json")


def _make_tool(**kwargs):
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


def test_init_creates_db_and_table():
    TinyDBApiToolStore(db_path="test.json")
    _mock_db_instance.table.assert_called_with("api_tools")


def test_create_inserts_tool(store):
    tool = _make_tool()

    result = store.create(tool)

    _mock_table_instance.insert.assert_called_once()
    assert result is tool


def test_get_returns_tool_when_found(store):
    tool = _make_tool()
    data = store._serialize(tool)
    _mock_table_instance.get.return_value = data

    result = store.get("some-id")

    assert isinstance(result, ApiToolConfig)


def test_get_returns_none_when_not_found(store):
    _mock_table_instance.get.return_value = None

    result = store.get("missing-id")

    assert result is None


def test_update_returns_none_when_not_found(store):
    _mock_table_instance.get.return_value = None
    tool = _make_tool()

    result = store.update(tool)

    assert result is None
    _mock_table_instance.update.assert_not_called()


def test_update_updates_existing_tool(store):
    _mock_table_instance.get.return_value = {"tool_id": "t1"}
    tool = _make_tool()

    result = store.update(tool)

    _mock_table_instance.update.assert_called_once()
    assert result is tool


def test_delete_returns_true_when_removed(store):
    _mock_table_instance.remove.return_value = [1]

    result = store.delete("t1")

    assert result is True


def test_delete_returns_false_when_not_found(store):
    _mock_table_instance.remove.return_value = []

    result = store.delete("missing")

    assert result is False


def test_list_returns_tools_for_user(store):
    tool = _make_tool()
    data = store._serialize(tool)
    _mock_table_instance.search.return_value = [data]

    result = store.list("user1")

    assert len(result) == 1
    assert isinstance(result[0], ApiToolConfig)


def test_list_enabled_returns_enabled_tools(store):
    tool = _make_tool()
    data = store._serialize(tool)
    _mock_table_instance.search.return_value = [data]

    result = store.list_enabled("user1")

    assert len(result) == 1
    assert isinstance(result[0], ApiToolConfig)


def test_list_filtered_uses_query(store):
    tool = _make_tool()
    data = store._serialize(tool)
    _mock_table_instance.search.return_value = [data]
    query_condition = MagicMock()

    result = store.list_filtered(query_condition)

    _mock_table_instance.search.assert_called_once_with(query_condition)
    assert len(result) == 1


def test_close_closes_db(store):
    store.close()
    _mock_db_instance.close.assert_called_once()


def test_serialize_sets_expires_at_isoformat(store):
    tool = _make_tool()
    tool.auth_secret_expires_at = datetime(2025, 1, 1)

    data = store._serialize(tool)

    assert isinstance(data["auth_secret_expires_at"], str)
    assert "2025-01-01" in data["auth_secret_expires_at"]


def test_deserialize_with_auth_secret_expires_at(store):
    tool = _make_tool()
    tool.auth_secret_expires_at = datetime(2025, 6, 15)
    data = store._serialize(tool)

    result = store._deserialize(data)

    assert isinstance(result.auth_secret_expires_at, datetime)


def test_deserialize_without_auth_secret_expires_at(store):
    tool = _make_tool()
    data = store._serialize(tool)

    result = store._deserialize(data)

    assert result.auth_secret_expires_at is None
