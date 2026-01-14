from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    RetryPromptPart,
)

from practorflow.llm.pyai.message_converter import MessageConverter, ConversionResult


def test_extract_text_content_all_types():
    # str
    assert MessageConverter.extract_text_content("x") == "x"

    # TextPart
    assert MessageConverter.extract_text_content(TextPart(content="t")) == "t"

    # UserPromptPart (str content)
    assert MessageConverter.extract_text_content(UserPromptPart(content="u")) == "u"

    # UserPromptPart (non-str content)
    assert MessageConverter.extract_text_content(UserPromptPart(content=123)) == "123"

    # SystemPromptPart
    assert MessageConverter.extract_text_content(SystemPromptPart(content="s")) == "s"

    # ToolReturnPart (str content)
    tr = ToolReturnPart(tool_name="tool", content="result")
    assert MessageConverter.extract_text_content(tr) == "[Tool Result: tool]\nresult"

    # ToolReturnPart (non-str content)
    tr2 = ToolReturnPart(tool_name="tool", content={"x": 1})
    assert MessageConverter.extract_text_content(tr2) == "[Tool Result: tool]\n{'x': 1}"

    # RetryPromptPart (str content)
    assert (
        MessageConverter.extract_text_content(RetryPromptPart(content="retry"))
        == "retry"
    )

    # RetryPromptPart (non-str content)
    assert MessageConverter.extract_text_content(RetryPromptPart(content=456)) == "456"

    # ToolCallPart with args
    tc = ToolCallPart(tool_name="calc", args={"x": 1}, tool_call_id="1")
    assert MessageConverter.extract_text_content(tc) == "[Calling tool: calc({'x': 1})]"

    # ToolCallPart without args
    tc2 = ToolCallPart(tool_name="noop", args=None, tool_call_id="2")
    assert MessageConverter.extract_text_content(tc2) == "[Calling tool: noop({})]"

    # Fallback branch (unknown object)
    class Unknown:
        def __str__(self):
            return "unknown"

    assert MessageConverter.extract_text_content(Unknown()) == "unknown"


def test_convert_model_request_system_and_user_parts():
    req = ModelRequest(
        parts=[
            SystemPromptPart(content="sys"),
            UserPromptPart(content="user"),
            RetryPromptPart(content="retry"),
        ],
        instructions=None,
    )

    msgs = MessageConverter.convert_model_request(req)

    assert msgs[0]["role"] == "system"
    assert "sys" in msgs[0]["content"]

    assert msgs[1]["role"] == "user"
    assert "user" in msgs[1]["content"]
    assert "retry" in msgs[1]["content"]


def test_convert_model_request_with_tool_return_and_fallback():
    req = ModelRequest(
        parts=[
            ToolReturnPart(tool_name="search", content="data"),
            TextPart(content="extra"),
        ],
        instructions=None,
    )

    msgs = MessageConverter.convert_model_request(req)

    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
    assert "Tool Result: search" in msgs[0]["content"]
    assert "extra" in msgs[0]["content"]


def test_convert_model_response_combines_parts():
    resp = ModelResponse(
        parts=[
            TextPart(content="a"),
            ToolCallPart(tool_name="calc", args={}, tool_call_id="1"),
        ],
        model_name="m",
        timestamp=None,
        usage=None,
    )

    msg = MessageConverter.convert_model_response(resp)

    assert msg["role"] == "assistant"
    assert "a" in msg["content"]
    assert "Calling tool: calc" in msg["content"]


def test_convert_messages_collects_system_and_messages():
    req = ModelRequest(
        parts=[SystemPromptPart(content="sys"), UserPromptPart(content="u")],
        instructions=None,
    )
    resp = ModelResponse(
        parts=[TextPart(content="a")],
        model_name="m",
        timestamp=None,
        usage=None,
    )

    result = MessageConverter.convert_messages(
        messages=[req, resp],
        system_prompt="base",
    )

    assert isinstance(result, ConversionResult)
    assert "base" in result.system_prompt
    assert "sys" in result.system_prompt
    assert len(result.messages) == 2
    assert result.messages[0]["role"] == "user"
    assert result.messages[1]["role"] == "assistant"


def test_convert_messages_without_system_prompt():
    req = ModelRequest(
        parts=[UserPromptPart(content="u")],
        instructions=None,
    )

    result = MessageConverter.convert_messages(messages=[req])

    assert result.system_prompt is None
    assert result.messages[0]["role"] == "user"


def test_create_simple_messages_all_paths():
    msgs = MessageConverter.create_simple_messages(
        prompt="p",
        system_prompt="s",
        history=[{"role": "assistant", "content": "h"}],
    )

    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "assistant"
    assert msgs[2]["role"] == "user"
    assert msgs[2]["content"] == "p"
