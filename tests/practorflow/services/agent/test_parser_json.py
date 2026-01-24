from practorflow.services.agent.parser import parse_json_from_response


def test_parse_json_from_code_block():
    text = """```json
    {"a": 1, "b": "x"}
    ```"""
    assert parse_json_from_response(text) == {"a": 1, "b": "x"}


def test_parse_json_from_bare_json():
    text = 'prefix {"a": 2} suffix'
    assert parse_json_from_response(text) == {"a": 2}


def test_parse_json_invalid_returns_none():
    assert parse_json_from_response("{invalid}") is None
    assert parse_json_from_response("") is None

def test_parse_json_code_block_jsondecodeerror_returns_none():
    text = """```json
    {"a": 1,}
    ```"""
    assert parse_json_from_response(text) is None
