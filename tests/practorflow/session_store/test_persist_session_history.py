from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

# Create module-level mocks before any imports
_mock_db_instance = MagicMock()
_mock_table_instance = MagicMock()
_mock_db_instance.table.return_value = _mock_table_instance

_tinydb_patcher = patch("practorflow.session_store.persist_session_history.TinyDB", return_value=_mock_db_instance)
_tinydb_patcher.start()

# Now safe to import after mocks are in place
from practorflow.session_store.persist_session_history import PersistSessionHistory
from practorflow.llm.base.session import Session, Message


@pytest.fixture(autouse=True)
def reset_mocks():
    """Reset mocks between tests."""
    _mock_db_instance.reset_mock()
    _mock_table_instance.reset_mock()
    _mock_db_instance.table.return_value = _mock_table_instance
    yield


@pytest.fixture
def mock_tinydb():
    """Provide mock instances for tests."""
    return _mock_db_instance, _mock_table_instance


def _session_dict(session_id="s1", user=None, updated_at=None):
    return {
        "session_id": session_id,
        "messages": [
            {
                "id": "m1",
                "role": "user",
                "content": "hi",
                "type": "message",
                "status": "completed",
                "timestamp": datetime.now().isoformat(),
            }
        ],
        "instructions": None,
        "documents": [],
        "metadata": {},
        "created_at": datetime.now().isoformat(),
        "updated_at": (updated_at or datetime.now()).isoformat(),
        "user": user,
        "title": None,
    }


def test_init_with_external_db(mock_tinydb):
    db, _ = mock_tinydb

    history = PersistSessionHistory(db=db)

    assert history._owns_db is False
    db.table.assert_called_once_with("sessions", cache_size=0)


def test_list_sessions_empty(mock_tinydb):
    db, table = mock_tinydb
    table.all.return_value = []

    history = PersistSessionHistory(db=db)

    result = history.list_sessions()

    assert result == []


def test_list_sessions_filtered_by_user_and_sorted(mock_tinydb):
    db, table = mock_tinydb

    older = _session_dict("s1", user="alice", updated_at=datetime(2020, 1, 1))
    newer = _session_dict("s2", user="alice", updated_at=datetime(2023, 1, 1))

    table.search.return_value = [older, newer]

    history = PersistSessionHistory(db=db)

    result = history.list_sessions(user="alice")

    assert isinstance(result[0], Session)
    assert result[0].session_id == "s2"
    assert result[1].session_id == "s1"


def test_get_history_existing_session(mock_tinydb):
    db, table = mock_tinydb
    table.get.return_value = _session_dict("s1")

    history = PersistSessionHistory(db=db)

    session = history.get_history("s1")

    assert isinstance(session, Session)
    assert session.session_id == "s1"


def test_get_history_missing_session(mock_tinydb):
    db, table = mock_tinydb
    table.get.return_value = None

    history = PersistSessionHistory(db=db)

    assert history.get_history("missing") is None


def test_deserialize_invalid_datetime_fallback(mock_tinydb):
    db, table = mock_tinydb

    data = _session_dict("s1")
    data["created_at"] = "invalid"
    data["updated_at"] = None
    data["messages"][0]["timestamp"] = "invalid"

    table.get.return_value = data

    history = PersistSessionHistory(db=db)

    session = history.get_history("s1")

    assert isinstance(session.created_at, datetime)
    assert isinstance(session.updated_at, datetime)
    assert isinstance(session.messages[0], Message)


def test_close_owned_db_closes_connection(mock_tinydb):
    db, _ = mock_tinydb
    
    history = PersistSessionHistory(db_path="test.json")
    history.close()
    
    db.close.assert_called_once()


def test_sessions_by_title_relevance_and_user_filtering(mock_tinydb):
    db, table = mock_tinydb

    now = datetime.now()

    s1 = {
        "session_id": "s1",
        "title": "Chat with assistant",
        "user": "alice",
        "messages": [],
        "instructions": None,
        "documents": [],
        "metadata": {},
        "created_at": now.isoformat(),
        "updated_at": (now - timedelta(minutes=5)).isoformat(),
    }

    s2 = {
        "session_id": "s2",
        "title": "Previous chat history",
        "user": "alice",
        "messages": [],
        "instructions": None,
        "documents": [],
        "metadata": {},
        "created_at": now.isoformat(),
        "updated_at": now.isoformat(),
    }

    table.search.return_value = [s1, s2]

    history = PersistSessionHistory(db=db)

    result = history.sessions_by_title("chat", user="alice")

    assert [s.session_id for s in result] == ["s1", "s2"]