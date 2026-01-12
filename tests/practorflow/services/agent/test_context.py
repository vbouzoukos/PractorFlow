from unittest.mock import MagicMock

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)

from practorflow.services.agent.context import (
    ExecutionContext,
    build_message_history,
    get_document_scope,
    get_document_context,
    build_execution_context,
)
from practorflow.llm.base.session import Session


def _make_msg(role: str, text: str):
    msg = MagicMock()
    msg.role = role
    msg.get_text_content.return_value = text
    return msg


# ------------------------
# ExecutionContext properties
# ------------------------

def test_execution_context_properties():
    session = MagicMock()
    session.session_id = "s1"

    ctx = ExecutionContext(
        session=session,
        message_history=[MagicMock(), MagicMock()],
        document_scope={"doc1", "doc2"},
        document_context="docs",
    )

    assert ctx.session_id == "s1"
    assert ctx.has_history is True
    assert ctx.history_length == 2
    assert ctx.has_documents is True


def test_execution_context_no_history_no_documents():
    session = MagicMock()
    session.session_id = "s1"

    ctx = ExecutionContext(
        session=session,
        message_history=[],
        document_scope=None,
        document_context=None,
    )

    assert ctx.has_history is False
    assert ctx.history_length == 0
    assert ctx.has_documents is False


# ------------------------
# build_message_history
# ------------------------

def test_build_message_history_empty_messages():
    session = Session(session_id="s1", user="u1", messages=[])

    history = build_message_history(session)

    assert history == []


def test_build_message_history_single_message():
    session = Session(
        session_id="s1",
        user="u1",
        messages=[_make_msg("user", "hello")],
    )

    history = build_message_history(session)

    assert history == []


def test_build_message_history_user_and_assistant():
    user_msg = _make_msg("user", "hello")
    assistant_msg = _make_msg("assistant", "hi")
    final_user_msg = _make_msg("user", "question")

    session = Session(
        session_id="s1",
        user="u1",
        messages=[user_msg, assistant_msg, final_user_msg],
    )

    history = build_message_history(session)

    assert len(history) == 2

    assert isinstance(history[0], ModelRequest)
    assert isinstance(history[0].parts[0], UserPromptPart)
    assert history[0].parts[0].content == "hello"

    assert isinstance(history[1], ModelResponse)
    assert isinstance(history[1].parts[0], TextPart)
    assert history[1].parts[0].content == "hi"


def test_build_message_history_ignores_unknown_roles():
    unknown_msg = _make_msg("system", "system text")
    final_user_msg = _make_msg("user", "question")

    session = Session(
        session_id="s1",
        user="u1",
        messages=[unknown_msg, final_user_msg],
    )

    history = build_message_history(session)

    assert history == []


# ------------------------
# get_document_scope
# ------------------------

def test_get_document_scope_no_documents():
    session = MagicMock()
    session.documents = []

    scope = get_document_scope(session)

    assert scope is None


def test_get_document_scope_with_documents():
    session = MagicMock()
    session.documents = [
        {"id": "doc1", "filename": "a.txt"},
        {"id": "doc2", "filename": "b.txt"},
    ]

    scope = get_document_scope(session)

    assert scope == {"doc1", "doc2"}


def test_get_document_scope_ignores_docs_without_id():
    session = MagicMock()
    session.documents = [
        {"filename": "a.txt"},
        {"id": "doc1"},
    ]

    scope = get_document_scope(session)

    assert scope == {"doc1"}


# ------------------------
# get_document_context
# ------------------------

def test_get_document_context_no_documents():
    session = MagicMock()
    session.documents = []

    context = get_document_context(session)

    assert context is None


def test_get_document_context_with_filenames():
    session = MagicMock()
    session.documents = [
        {"id": "doc1", "filename": "a.txt"},
        {"id": "doc2", "filename": "b.txt"},
    ]

    context = get_document_context(session)

    assert "Available documents:" in context
    assert "- a.txt" in context
    assert "- b.txt" in context


def test_get_document_context_missing_filenames():
    session = MagicMock()
    session.documents = [{"id": "doc1"}]

    context = get_document_context(session)

    assert "- unknown" in context


# ------------------------
# build_execution_context
# ------------------------

def test_build_execution_context_basic():
    user_msg = _make_msg("user", "hello")
    final_user_msg = _make_msg("user", "question")

    session = Session(
        session_id="s1",
        user="u1",
        messages=[user_msg, final_user_msg],
    )
    session.documents = []

    ctx = build_execution_context(session)

    assert isinstance(ctx, ExecutionContext)
    assert ctx.session is session
    assert ctx.has_history is True
    assert ctx.history_length == 1
    assert ctx.document_scope is None
    assert ctx.document_context is None


def test_build_execution_context_with_documents_and_extra_ids():
    session = Session(
        session_id="s1",
        user="u1",
        messages=[],
    )
    session.documents = [
        {"id": "doc1", "filename": "a.txt"},
    ]

    ctx = build_execution_context(session, document_ids={"doc2"})

    assert ctx.document_scope == {"doc1", "doc2"}
    assert "a.txt" in ctx.document_context


def test_build_execution_context_only_extra_document_ids():
    session = Session(
        session_id="s1",
        user="u1",
        messages=[],
    )
    session.documents = []

    ctx = build_execution_context(session, document_ids={"docX"})

    assert ctx.document_scope == {"docX"}
    assert ctx.document_context is None

def test_build_message_history_all_messages_empty_via_custom_sequence():
    from practorflow.services.agent.context import build_message_history

    class WeirdMessages(list):
        def __len__(self):
            return 2  # bypass first guard

        def __getitem__(self, item):
            if isinstance(item, slice):
                return []  # force empty all_messages
            return super().__getitem__(item)

    session = MagicMock()
    session.messages = WeirdMessages()

    history = build_message_history(session)

    assert history == []

def test_get_document_context_doc_names_empty_via_truthy_iterable():
    from practorflow.services.agent.context import get_document_context

    class TruthyEmptyIterable:
        def __bool__(self):
            return True  # bypass `if not session.documents`

        def __iter__(self):
            return iter([])  # produces no filenames

    session = MagicMock()
    session.documents = TruthyEmptyIterable()

    context = get_document_context(session)

    assert context is None
