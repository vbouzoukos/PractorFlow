from practorflow.session_store.memory_session_store import InMemorySessionStore
from practorflow.llm.base.session import Session


def test_get_creates_new_session_if_missing():
    store = InMemorySessionStore()

    session = store.get("s1")

    assert isinstance(session, Session)
    assert session.session_id == "s1"


def test_get_returns_existing_session():
    store = InMemorySessionStore()
    session = Session(session_id="s1")
    store.save(session)

    result = store.get("s1")

    assert result is session


def test_save_persists_session():
    store = InMemorySessionStore()
    session = Session(session_id="s1")

    store.save(session)

    assert store.exists("s1") is True
    assert store.get("s1") is session


def test_delete_existing_session():
    store = InMemorySessionStore()
    session = Session(session_id="s1")
    store.save(session)

    store.delete("s1")

    assert store.exists("s1") is False


def test_delete_missing_session_is_noop():
    store = InMemorySessionStore()

    store.delete("missing")

    assert store.exists("missing") is False


def test_exists_returns_true_for_existing_session():
    store = InMemorySessionStore()
    store.save(Session(session_id="s1"))

    assert store.exists("s1") is True


def test_exists_returns_false_for_missing_session():
    store = InMemorySessionStore()

    assert store.exists("missing") is False
