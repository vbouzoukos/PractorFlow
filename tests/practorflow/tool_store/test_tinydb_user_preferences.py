"""Tests for TinyDBUserToolPreferencesStore."""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

_mock_db_instance = MagicMock()
_mock_table_instance = MagicMock()
_mock_query_instance = MagicMock()

_mock_db_instance.table.return_value = _mock_table_instance

_tinydb_patcher = patch("practorflow.tool_store.tinydb_user_preferences.TinyDB", return_value=_mock_db_instance)
_query_patcher = patch("practorflow.tool_store.tinydb_user_preferences.Query", return_value=_mock_query_instance)

_tinydb_patcher.start()
_query_patcher.start()

from practorflow.tool_store.tinydb_user_preferences import TinyDBUserToolPreferencesStore
from practorflow.llm.tools.user_preferences import EnabledToolEntry, UserToolPreferences


@pytest.fixture(autouse=True)
def reset_mocks():
    _mock_db_instance.reset_mock()
    _mock_table_instance.reset_mock()
    _mock_query_instance.reset_mock()
    _mock_db_instance.table.return_value = _mock_table_instance
    yield


@pytest.fixture
def store():
    return TinyDBUserToolPreferencesStore(db_path="test.json")


def _make_preferences(**kwargs):
    defaults = {"user_id": "user1"}
    defaults.update(kwargs)
    return UserToolPreferences(**defaults)


def test_init_creates_db_and_table():
    TinyDBUserToolPreferencesStore(db_path="test.json")
    _mock_db_instance.table.assert_called_with("user_tool_preferences")


def test_get_returns_preferences_when_found(store):
    prefs = _make_preferences()
    data = store._serialize(prefs)
    _mock_table_instance.get.return_value = data

    result = store.get("user1")

    assert isinstance(result, UserToolPreferences)
    assert result.user_id == "user1"


def test_get_returns_none_when_not_found(store):
    _mock_table_instance.get.return_value = None

    result = store.get("missing-user")

    assert result is None


def test_update_updates_when_existing(store):
    prefs = _make_preferences()
    _mock_table_instance.get.return_value = {"user_id": "user1"}

    result = store.update(prefs)

    _mock_table_instance.update.assert_called_once()
    _mock_table_instance.insert.assert_not_called()
    assert result is prefs


def test_update_inserts_when_new(store):
    prefs = _make_preferences()
    _mock_table_instance.get.return_value = None

    result = store.update(prefs)

    _mock_table_instance.insert.assert_called_once()
    _mock_table_instance.update.assert_not_called()
    assert result is prefs


def test_create_default_inserts(store):
    result = store.create_default("user2")

    _mock_table_instance.insert.assert_called_once()
    assert isinstance(result, UserToolPreferences)
    assert result.user_id == "user2"


def test_delete_returns_true_when_removed(store):
    _mock_table_instance.remove.return_value = ["user1"]

    result = store.delete("user1")

    assert result is True


def test_delete_returns_false_when_not_found(store):
    _mock_table_instance.remove.return_value = []

    result = store.delete("missing")

    assert result is False


def test_close_closes_db(store):
    store.close()
    _mock_db_instance.close.assert_called_once()


def test_serialize_with_enabled_tools(store):
    prefs = _make_preferences(
        enabled_tools=[EnabledToolEntry(type="api", id="t1")],
        enabled_mcp_servers=["s1"],
    )

    data = store._serialize(prefs)

    assert len(data["enabled_tools"]) == 1
    assert data["enabled_tools"][0]["type"] == "api"
    assert data["enabled_tools"][0]["id"] == "t1"
    assert data["enabled_mcp_servers"] == ["s1"]


def test_deserialize_with_enabled_tools(store):
    prefs = _make_preferences(
        enabled_tools=[EnabledToolEntry(type="builtin", id="b1")],
        enabled_mcp_servers=["s2"],
    )
    data = store._serialize(prefs)

    result = store._deserialize(data)

    assert len(result.enabled_tools) == 1
    assert result.enabled_tools[0].type == "builtin"
    assert result.enabled_tools[0].id == "b1"
    assert result.enabled_mcp_servers == ["s2"]


def test_deserialize_without_enabled_tools(store):
    prefs = _make_preferences()
    data = store._serialize(prefs)

    result = store._deserialize(data)

    assert result.enabled_tools == []
    assert result.enabled_mcp_servers == []
