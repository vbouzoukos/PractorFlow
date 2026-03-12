from unittest.mock import MagicMock

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    UserPromptPart,
    TextPart,
)

from practorflow.services.history.builder import messages_to_text


def test_messages_to_text_empty_list():
    result = messages_to_text([])

    assert result == ""


def test_messages_to_text_user_message():
    msg = ModelRequest(parts=[UserPromptPart(content="hello")])

    result = messages_to_text([msg])

    assert result == "User: hello"


def test_messages_to_text_assistant_message():
    msg = ModelResponse(parts=[TextPart(content="world")])

    result = messages_to_text([msg])

    assert result == "Assistant: world"


def test_messages_to_text_multiple_messages():
    msgs = [
        ModelRequest(parts=[UserPromptPart(content="hi")]),
        ModelResponse(parts=[TextPart(content="there")]),
    ]

    result = messages_to_text(msgs)

    assert result == "User: hi\n\nAssistant: there"


def test_messages_to_text_skips_message_without_parts():
    msg = MagicMock(spec=[])

    result = messages_to_text([msg])

    assert result == ""


def test_messages_to_text_skips_part_without_content():
    part = MagicMock(spec=[])
    msg = MagicMock()
    msg.parts = [part]

    result = messages_to_text([msg])

    assert result == ""
