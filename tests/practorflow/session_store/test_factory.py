import pytest
from unittest.mock import patch

from practorflow.session_store.factory import (
    create_session_store,
    create_session_history,
    validate_store_path,
    DEFAULT_DB_PATH,
)
from practorflow.session_store.memory_session_store import InMemorySessionStore
from practorflow.session_store.persist_session_store import TinyDBSessionStore
from practorflow.session_store.memory_session_history import InMemorySessionHistory
from practorflow.session_store.persist_session_history import PersistSessionHistory


def test_validate_store_path_existing(tmp_path):
    path = tmp_path / "db.json"
    path.write_text("{}")

    with patch("practorflow.session_store.factory.logger") as mock_logger:
        validate_store_path(str(path))
        mock_logger.warning.assert_not_called()


def test_validate_store_path_missing(tmp_path):
    missing_path = tmp_path / "missing.json"

    with patch("practorflow.session_store.factory.logger") as mock_logger:
        validate_store_path(str(missing_path))
        mock_logger.warning.assert_called_once()


def test_create_session_store_memory(clean_env):
    clean_env["STORE_SESSION"] = "MEMORY"

    store = create_session_store()

    assert isinstance(store, InMemorySessionStore)


def test_create_session_store_memory_case_and_whitespace(clean_env):
    clean_env["STORE_SESSION"] = "  MeMoRy  "

    store = create_session_store()

    assert isinstance(store, InMemorySessionStore)


def test_create_session_store_local(clean_env, tmp_path):
    db_path = tmp_path / "sessions.json"
    clean_env["STORE_SESSION"] = "LOCAL"
    clean_env["STORE_SESSION_DB_PATH"] = str(db_path)

    store = create_session_store()

    assert isinstance(store, TinyDBSessionStore)


def test_create_session_store_local_default_path(clean_env):
    clean_env["STORE_SESSION"] = "local"

    with patch(
        "practorflow.session_store.factory.validate_store_path"
    ) as mock_validate:
        store = create_session_store()

        mock_validate.assert_called_once_with(DEFAULT_DB_PATH)
        assert isinstance(store, TinyDBSessionStore)


def test_create_session_store_invalid_type(clean_env):
    clean_env["STORE_SESSION"] = "INVALID"

    with pytest.raises(ValueError) as exc:
        create_session_store()

    assert "Unsupported session store type" in str(exc.value)


def test_create_session_history_memory(clean_env):
    clean_env["STORE_SESSION"] = "memory"

    history = create_session_history()

    assert isinstance(history, InMemorySessionHistory)


def test_create_session_history_local(clean_env, tmp_path):
    db_path = tmp_path / "sessions.json"
    clean_env["STORE_SESSION"] = "local"
    clean_env["STORE_SESSION_DB_PATH"] = str(db_path)

    history = create_session_history()

    assert isinstance(history, PersistSessionHistory)


def test_create_session_history_invalid_type(clean_env):
    clean_env["STORE_SESSION"] = "INVALID"

    with pytest.raises(ValueError) as exc:
        create_session_history()

    assert "Unsupported session history type" in str(exc.value)
