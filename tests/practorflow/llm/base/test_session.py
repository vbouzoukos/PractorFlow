# test_session.py

import pytest
from datetime import datetime

from practorflow.llm.base.session import Session, Message


def test_message_to_dict_and_text_content_string():
    msg = Message(role="user", content="hello")

    d = msg.to_dict()

    assert d["role"] == "user"
    assert d["content"] == "hello"
    assert msg.get_text_content() == "hello"


def test_message_get_text_content_structured():
    msg = Message(
        role="assistant",
        content=[
            {"text": "hello"},
            {"text": "world"},
            {"other": "ignored"},
        ],
    )

    assert msg.get_text_content() == "hello\nworld"

    # fallback branch: non-str, non-list
    msg.content = 12345
    assert msg.get_text_content() == "12345"


def test_session_add_document_new_and_update():
    session = Session(session_id="s1")

    doc = {"id": "d1", "content": "a", "filename": "f.txt"}
    session.add_document(doc)

    assert session.get_document_count() == 1
    assert session.get_document("d1") == doc

    updated = {"id": "d1", "content": "b", "filename": "f.txt"}
    session.add_document(updated)

    assert session.get_document_count() == 1
    assert session.get_document("d1")["content"] == "b"

    # cover missing-document branch
    assert session.get_document("missing") is None


def test_remove_document_found_and_not_found():
    session = Session(session_id="s1")
    session.add_document({"id": "d1", "content": "a", "filename": "f.txt"})

    assert session.remove_document("d1") is True
    assert session.get_document_count() == 0

    assert session.remove_document("missing") is False


def test_clear_documents():
    session = Session(session_id="s1")
    session.add_document({"id": "d1", "content": "a", "filename": "f.txt"})

    session.clear_documents()

    assert session.get_document_count() == 0


def test_list_documents_summary():
    session = Session(session_id="s1")
    session.add_document({"id": "d1", "filename": "file.txt", "file_type": "txt"})

    docs = session.list_documents()

    assert docs == [{"id": "d1", "filename": "file.txt", "file_type": "txt"}]


def test_truncate_messages_normal():
    m1 = Message(role="user", content="a")
    m2 = Message(role="assistant", content="b")
    m3 = Message(role="user", content="c")

    session = Session(session_id="s1", messages=[m1, m2, m3])

    removed = session.truncate_messages(1)

    assert removed == 2
    assert session.get_message_count() == 1


def test_truncate_messages_index_out_of_range():
    session = Session(session_id="s1")

    removed = session.truncate_messages(5)

    assert removed == 0


def test_truncate_messages_negative_index():
    session = Session(session_id="s1")

    with pytest.raises(ValueError):
        session.truncate_messages(-1)


def test_get_message_by_index():
    msg = Message(role="user", content="hi")
    session = Session(session_id="s1", messages=[msg])

    assert session.get_message_by_index(0) is msg
    assert session.get_message_by_index(1) is None


def test_get_message_by_id_and_index_by_id():
    msg1 = Message(role="user", content="a")
    msg2 = Message(role="assistant", content="b")

    session = Session(session_id="s1", messages=[msg1, msg2])

    assert session.get_message_by_id(msg2.id) is msg2
    assert session.get_message_by_id("missing") is None

    assert session.get_message_index_by_id(msg1.id) == 0
    assert session.get_message_index_by_id("missing") is None
