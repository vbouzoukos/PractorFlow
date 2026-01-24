from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

# Create module-level mocks before any imports
_mock_db_instance = MagicMock()
_mock_table_instance = MagicMock()
_mock_query_instance = MagicMock()

_mock_db_instance.table.return_value = _mock_table_instance

_tinydb_patcher = patch("practorflow.session_store.persist_session_store.TinyDB", return_value=_mock_db_instance)
_query_patcher = patch("practorflow.session_store.persist_session_store.Query", return_value=_mock_query_instance)

_tinydb_patcher.start()
_query_patcher.start()

# Now safe to import after mocks are in place
from practorflow.session_store.persist_session_store import TinyDBSessionStore
from practorflow.llm.base.session import Session, Message


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset mocks between tests."""
    _mock_db_instance.reset_mock()
    _mock_table_instance.reset_mock()
    _mock_query_instance.reset_mock()
    _mock_db_instance.table.return_value = _mock_table_instance
    yield


@pytest.fixture
def mock_tinydb():
    """Provide mock instances for tests."""
    return _mock_db_instance, _mock_table_instance, _mock_query_instance


def _make_session(session_id="s1"):
    msg = Message(role="user", content="hello")
    return Session(session_id=session_id, messages=[msg])


def test_init_creates_db_and_table(mock_tinydb):
    db, table, _ = mock_tinydb

    store = TinyDBSessionStore(db_path="test.json")

    db.table.assert_called_once_with("sessions")
    assert store._db is db
    assert store._sessions is table


def test_get_creates_new_session_if_missing(mock_tinydb):
    _, table, query = mock_tinydb
    table.get.return_value = None

    store = TinyDBSessionStore()
    session = store.get("s1")

    assert isinstance(session, Session)
    assert session.session_id == "s1"
    table.insert.assert_called_once()


def test_get_returns_existing_session(mock_tinydb):
    _, table, query = mock_tinydb

    table.get.return_value = {
        "session_id": "s1",
        "messages": [],
        "instructions": None,
        "documents": [],
        "metadata": {},
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
        "user": None,
        "title": None,
    }

    store = TinyDBSessionStore()
    session = store.get("s1")

    assert isinstance(session, Session)
    assert session.session_id == "s1"


def test_save_inserts_new_session(mock_tinydb):
    _, table, query = mock_tinydb
    table.get.return_value = None

    store = TinyDBSessionStore()
    session = _make_session("s1")

    store.save(session)

    table.insert.assert_called_once()


def test_save_updates_existing_session(mock_tinydb):
    _, table, query = mock_tinydb
    table.get.return_value = {"session_id": "s1"}

    store = TinyDBSessionStore()
    session = _make_session("s1")

    store.save(session)

    table.update.assert_called_once()


def test_delete_session(mock_tinydb):
    _, table, query = mock_tinydb

    store = TinyDBSessionStore()
    store.delete("s1")

    table.remove.assert_called_once()


def test_exists_true(mock_tinydb):
    _, table, query = mock_tinydb
    table.contains.return_value = True

    store = TinyDBSessionStore()

    assert store.exists("s1") is True


def test_exists_false(mock_tinydb):
    _, table, query = mock_tinydb
    table.contains.return_value = False

    store = TinyDBSessionStore()

    assert store.exists("missing") is False


def test_list_sessions(mock_tinydb):
    _, table, _ = mock_tinydb
    table.all.return_value = [
        {
            "session_id": "s1",
            "messages": [],
            "documents": [],
            "created_at": "now",
            "updated_at": "now",
        }
    ]

    store = TinyDBSessionStore()

    result = store.list_sessions()

    assert result[0]["session_id"] == "s1"
    assert "message_count" in result[0]


def test_clear_all(mock_tinydb):
    _, table, _ = mock_tinydb
    table.__len__.return_value = 2

    store = TinyDBSessionStore()

    count = store.clear_all()

    assert count == 2
    table.truncate.assert_called_once()


def test_close_closes_db(mock_tinydb):
    db, _, _ = mock_tinydb

    store = TinyDBSessionStore()
    store.close()

    db.close.assert_called_once()


def test_repr_includes_session_count(mock_tinydb):
    _, table, _ = mock_tinydb
    table.__len__.return_value = 5

    store = TinyDBSessionStore()

    text = repr(store)

    assert "TinyDBSessionStore" in text
    assert "5" in text


def test_deserialize_message_all_fields(mock_tinydb):
    store = TinyDBSessionStore()

    data = {
        "id": "m1",
        "role": "assistant",
        "content": "hello",
        "type": "message",
        "status": "completed",
        "timestamp": datetime.now().isoformat(),
    }

    msg = store._deserialize_message(data)

    assert msg.id == "m1"
    assert msg.role == "assistant"
    assert msg.content == "hello"
    assert msg.type == "message"
    assert msg.status == "completed"
    assert isinstance(msg.timestamp, datetime)


def test_serialize_datetime_none(mock_tinydb):
    store = TinyDBSessionStore()

    result = store._serialize_datetime(None)

    assert result is None


def test_deserialize_datetime_none(mock_tinydb):
    store = TinyDBSessionStore()

    result = store._deserialize_datetime(None)

    assert isinstance(result, datetime)


def test_deserialize_datetime_invalid_string(mock_tinydb):
    store = TinyDBSessionStore()

    result = store._deserialize_datetime("not-a-datetime")

    assert isinstance(result, datetime)


def test_deserialize_datetime_type_error(mock_tinydb):
    store = TinyDBSessionStore()

    result = store._deserialize_datetime(12345)

    assert isinstance(result, datetime)