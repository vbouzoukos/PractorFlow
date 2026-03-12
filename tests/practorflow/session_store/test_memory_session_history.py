from datetime import datetime, timedelta

from practorflow.session_store.memory_session_history import InMemorySessionHistory
from practorflow.llm.base.session import Session


def _make_session(session_id: str, user: str, updated_at: datetime) -> Session:
    s = Session(session_id=session_id, user=user)
    s.updated_at = updated_at
    return s


def test_init_with_no_sessions():
    history = InMemorySessionHistory()
    assert history.list_sessions() == []


def test_init_with_existing_sessions():
    sessions = {"s1": Session(session_id="s1")}
    history = InMemorySessionHistory(sessions=sessions)

    assert history.get_history("s1") is sessions["s1"]


def test_list_sessions_returns_all_sorted_by_updated_at_desc():
    now = datetime.now()
    older = now - timedelta(minutes=10)

    s1 = _make_session("s1", "alice", older)
    s2 = _make_session("s2", "bob", now)

    history = InMemorySessionHistory(
        sessions={
            "s1": s1,
            "s2": s2,
        }
    )

    result = history.list_sessions()

    assert result == [s2, s1]


def test_list_sessions_filtered_by_user():
    now = datetime.now()

    s1 = _make_session("s1", "alice", now)
    s2 = _make_session("s2", "bob", now)

    history = InMemorySessionHistory(
        sessions={
            "s1": s1,
            "s2": s2,
        }
    )

    result = history.list_sessions(user="alice")

    assert result == [s1]


def test_get_history_existing_session():
    session = Session(session_id="s1")
    history = InMemorySessionHistory(sessions={"s1": session})

    assert history.get_history("s1") is session


def test_get_history_missing_session():
    history = InMemorySessionHistory()

    assert history.get_history("missing") is None

def test_sessions_by_title_relevance_and_user_filtering():

    now = datetime.now()

    # Session where title STARTS with search term (most relevant)
    s1 = Session(session_id="s1", user="alice")
    s1.title = "Chat with assistant"
    s1.updated_at = now - timedelta(minutes=5)

    # Session where title CONTAINS search term (less relevant, newer)
    s2 = Session(session_id="s2", user="alice")
    s2.title = "My previous chat history"
    s2.updated_at = now

    # Different user (should be filtered out)
    s3 = Session(session_id="s3", user="bob")
    s3.title = "Chat about testing"
    s3.updated_at = now

    # No title (should be ignored)
    s4 = Session(session_id="s4", user="alice")
    s4.title = None
    s4.updated_at = now

    history = InMemorySessionHistory(
        sessions={
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "s4": s4,
        }
    )

    result = history.sessions_by_title("chat", user="alice")

    # s1 comes before s2 because it starts with "chat"
    assert result == [s1, s2]
