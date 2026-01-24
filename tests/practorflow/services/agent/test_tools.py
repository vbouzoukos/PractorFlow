import json
import pytest
from unittest.mock import MagicMock

from practorflow.services.agent.tools import (
    register_default_tools,
    register_executor_tools,
)
from tests.practorflow.services.agent.common_agent_deps import make_agent_deps


def test_register_default_tools_registers_missing_tools():
    tool_registry = MagicMock()
    tool_registry.__contains__.side_effect = lambda name: False

    knowledge_store = MagicMock()

    register_default_tools(tool_registry, knowledge_store)

    assert tool_registry.register.call_count == 6


def test_register_default_tools_skips_existing():
    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True

    knowledge_store = MagicMock()

    register_default_tools(tool_registry, knowledge_store)

    tool_registry.register.assert_not_called()


@pytest.mark.asyncio
async def test_execute_tool_success_and_sets_scope():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute.return_value = MagicMock(success=True, data="ok", error=None)

    deps = make_agent_deps(MagicMock(), tool_registry, document_scope={"d1"})

    register_executor_tools(agent, deps)
    execute_tool = agent.tool.call_args_list[0][0][0]

    result = await execute_tool(
        ctx=MagicMock(deps=deps),
        tool_name="t",
        tool_args={"a": 1},
    )

    tool_registry.set_document_scope.assert_called_once_with({"d1"})
    assert result == "ok"


@pytest.mark.asyncio
async def test_execute_tool_failure_result():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute.return_value = MagicMock(
        success=False,
        data=None,
        error="bad",
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    execute_tool = agent.tool.call_args_list[0][0][0]

    result = await execute_tool(
        ctx=MagicMock(deps=deps),
        tool_name="t",
        tool_args=None,
    )

    assert "tool error" in result.lower()


@pytest.mark.asyncio
async def test_execute_tool_not_found():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = False

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    execute_tool = agent.tool.call_args_list[0][0][0]

    result = await execute_tool(
        ctx=MagicMock(deps=deps),
        tool_name="missing",
        tool_args=None,
    )

    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_search_knowledge_with_results():
    agent = MagicMock()
    knowledge_store = MagicMock()
    knowledge_store.search_scoped.return_value = [
        {"text": "abc", "filename": "f.txt"}
    ]

    deps = make_agent_deps(knowledge_store, MagicMock())

    register_executor_tools(agent, deps)
    search_knowledge = agent.tool.call_args_list[1][0][0]

    result = await search_knowledge(
        ctx=MagicMock(deps=deps),
        query="q",
    )

    assert "found 1 result" in result.lower()
    assert "f.txt" in result


@pytest.mark.asyncio
async def test_search_web_success():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(success=True, data="web", error=None)

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    search_web = agent.tool.call_args_list[2][0][0]

    result = await search_web(
        ctx=MagicMock(deps=deps),
        query="q",
        max_results=1,
    )

    assert result == "web"


@pytest.mark.asyncio
async def test_fetch_webpage_error():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(
        success=False,
        data=None,
        error="404",
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    fetch_webpage = agent.tool.call_args_list[3][0][0]

    result = await fetch_webpage(
        ctx=MagicMock(deps=deps),
        url="x",
    )

    assert "fetch error" in result.lower()


@pytest.mark.asyncio
async def test_summarize_text_success():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(
        success=True,
        data="summary",
        error=None,
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    summarize = agent.tool.call_args_list[4][0][0]

    result = await summarize(
        ctx=MagicMock(deps=deps),
        text="long",
        num_sentences=2,
    )

    assert result == "summary"


@pytest.mark.asyncio
async def test_transform_json_dict_result():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(
        success=True,
        data={"a": 1},
        error=None,
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    transform = agent.tool.call_args_list[5][0][0]

    result = await transform(
        ctx=MagicMock(deps=deps),
        json_data="{}",
        operation="parse",
    )

    assert json.loads(result) == {"a": 1}


@pytest.mark.asyncio
async def test_transform_json_failure():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(
        success=False,
        data=None,
        error="bad",
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    transform = agent.tool.call_args_list[5][0][0]

    result = await transform(
        ctx=MagicMock(deps=deps),
        json_data="{}",
        operation="parse",
    )

    assert "error" in result.lower()


@pytest.mark.asyncio
async def test_calculate_success():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(
        success=True,
        data=42,
        error=None,
    )

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    calculate = agent.tool.call_args_list[6][0][0]

    result = await calculate(
        ctx=MagicMock(deps=deps),
        expression="6*7",
    )

    assert result == "42"

def test_register_default_tools_registers_missing_tools():
    tool_registry = MagicMock()
    tool_registry.__contains__.side_effect = lambda name: False

    knowledge_store = MagicMock()

    register_default_tools(tool_registry, knowledge_store)

    assert tool_registry.register.call_count == 6


def test_register_default_tools_skips_existing():
    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True

    knowledge_store = MagicMock()

    register_default_tools(tool_registry, knowledge_store)

    tool_registry.register.assert_not_called()


@pytest.mark.asyncio
async def test_execute_tool_success_and_sets_scope():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute.return_value = MagicMock(success=True, data="ok", error=None)

    deps = make_agent_deps(MagicMock(), tool_registry, document_scope={"d1"})

    register_executor_tools(agent, deps)
    execute_tool = agent.tool.call_args_list[0][0][0]

    result = await execute_tool(
        ctx=MagicMock(deps=deps),
        tool_name="t",
        tool_args={"a": 1},
    )

    tool_registry.set_document_scope.assert_called_once_with({"d1"})
    assert result == "ok"


@pytest.mark.asyncio
async def test_execute_tool_failure_result():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute.return_value = MagicMock(success=False, data=None, error="bad")

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    execute_tool = agent.tool.call_args_list[0][0][0]

    result = await execute_tool(
        ctx=MagicMock(deps=deps),
        tool_name="t",
        tool_args=None,
    )

    assert result == "Tool error: bad"


@pytest.mark.asyncio
async def test_execute_tool_not_found():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = False

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    execute_tool = agent.tool.call_args_list[0][0][0]

    result = await execute_tool(
        ctx=MagicMock(deps=deps),
        tool_name="missing",
        tool_args=None,
    )

    assert "not found" in result.lower()


@pytest.mark.asyncio
async def test_execute_tool_exception_returns_tool_execution_failed():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.__contains__.return_value = True
    tool_registry.execute.side_effect = RuntimeError("boom")

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    execute_tool = agent.tool.call_args_list[0][0][0]

    result = await execute_tool(
        ctx=MagicMock(deps=deps),
        tool_name="t",
        tool_args=None,
    )

    assert "tool execution failed" in result.lower()
    assert "boom" in result.lower()


@pytest.mark.asyncio
async def test_search_knowledge_no_results():
    agent = MagicMock()
    knowledge_store = MagicMock()
    knowledge_store.search_scoped.return_value = []

    deps = make_agent_deps(knowledge_store, MagicMock())

    register_executor_tools(agent, deps)
    search_knowledge = agent.tool.call_args_list[1][0][0]

    result = await search_knowledge(
        ctx=MagicMock(deps=deps),
        query="q",
    )

    assert result == ""


@pytest.mark.asyncio
async def test_search_knowledge_with_results():
    agent = MagicMock()
    knowledge_store = MagicMock()
    knowledge_store.search_scoped.return_value = [{"text": "abc", "filename": "f.txt"}]
    document_scope = []
    deps = make_agent_deps(knowledge_store, MagicMock(), document_scope)

    register_executor_tools(agent, deps)
    search_knowledge = agent.tool.call_args_list[1][0][0]

    result = await search_knowledge(
        ctx=MagicMock(deps=deps),
        query="q",
    )

    assert "found 1 result" in result.lower()
    assert "f.txt" in result


@pytest.mark.asyncio
async def test_search_web_success():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(success=True, data="web", error=None)

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    search_web = agent.tool.call_args_list[2][0][0]

    result = await search_web(
        ctx=MagicMock(deps=deps),
        query="q",
        max_results=1,
    )

    assert result == "web"


@pytest.mark.asyncio
async def test_search_web_error():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(success=False, data=None, error="oops")

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    search_web = agent.tool.call_args_list[2][0][0]

    result = await search_web(
        ctx=MagicMock(deps=deps),
        query="q",
        max_results=1,
    )

    assert result == ""


@pytest.mark.asyncio
async def test_fetch_webpage_success_no_content():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(success=True, data=None, error=None)

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    fetch_webpage = agent.tool.call_args_list[3][0][0]

    result = await fetch_webpage(
        ctx=MagicMock(deps=deps),
        url="x",
        extract_mode="text",
    )

    assert result == ""


@pytest.mark.asyncio
async def test_fetch_webpage_error():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(success=False, data=None, error="404")

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    fetch_webpage = agent.tool.call_args_list[3][0][0]

    result = await fetch_webpage(
        ctx=MagicMock(deps=deps),
        url="x",
    )

    assert result == ""


@pytest.mark.asyncio
async def test_summarize_text_success():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(success=True, data="summary", error=None)

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    summarize = agent.tool.call_args_list[4][0][0]

    result = await summarize(
        ctx=MagicMock(deps=deps),
        text="long",
        num_sentences=2,
    )

    assert result == "summary"


@pytest.mark.asyncio
async def test_summarize_text_error():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(success=False, data=None, error="bad")

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    summarize = agent.tool.call_args_list[4][0][0]

    result = await summarize(
        ctx=MagicMock(deps=deps),
        text="long",
        num_sentences=2,
    )

    assert result == ""


@pytest.mark.asyncio
async def test_transform_json_dict_result_is_pretty_json():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(success=True, data={"a": 1}, error=None)

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    transform = agent.tool.call_args_list[5][0][0]

    result = await transform(
        ctx=MagicMock(deps=deps),
        json_data="{}",
        operation="parse",
        path=None,
    )

    assert json.loads(result) == {"a": 1}


@pytest.mark.asyncio
async def test_transform_json_success_no_result_when_data_falsy():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(success=True, data=None, error=None)

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    transform = agent.tool.call_args_list[5][0][0]

    result = await transform(
        ctx=MagicMock(deps=deps),
        json_data="{}",
        operation="keys",
        path=None,
    )

    assert result == "No result."


@pytest.mark.asyncio
async def test_transform_json_failure():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(success=False, data=None, error="bad")

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    transform = agent.tool.call_args_list[5][0][0]

    result = await transform(
        ctx=MagicMock(deps=deps),
        json_data="{}",
        operation="parse",
        path=None,
    )

    assert result == "JSON transform error: bad"


@pytest.mark.asyncio
async def test_calculate_success():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(success=True, data=42, error=None)

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    calculate = agent.tool.call_args_list[6][0][0]

    result = await calculate(
        ctx=MagicMock(deps=deps),
        expression="6*7",
        convert=None,
    )

    assert result == "42"


@pytest.mark.asyncio
async def test_calculate_error():
    agent = MagicMock()
    tool_registry = MagicMock()
    tool_registry.execute.return_value = MagicMock(success=False, data=None, error="nope")

    deps = make_agent_deps(MagicMock(), tool_registry)

    register_executor_tools(agent, deps)
    calculate = agent.tool.call_args_list[6][0][0]

    result = await calculate(
        ctx=MagicMock(deps=deps),
        expression="6*7",
        convert=None,
    )

    assert result == "Calculation error: nope"

@pytest.mark.asyncio
async def test_search_knowledge_no_results_with_document_scope():
    agent = MagicMock()
    knowledge_store = MagicMock()
    knowledge_store.search_scoped.return_value = []

    # IMPORTANT: document_scope must be non-None
    deps = make_agent_deps(
        knowledge_store=knowledge_store,
        tool_registry=MagicMock(),
        document_scope={"doc-1"},
    )

    register_executor_tools(agent, deps)
    search_knowledge = agent.tool.call_args_list[1][0][0]

    result = await search_knowledge(
        ctx=MagicMock(deps=deps),
        query="q",
    )

    assert result == ""
    knowledge_store.search_scoped.assert_called_once()
