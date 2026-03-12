"""
Tests for api_tools builder.

Tests _build_tool_description (pure function) and _create_api_tool_wrapper
(returns an async closure) from the api_tools builder module.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from practorflow.services.tools.builders.api_tools import (
    _build_tool_description,
    _create_api_tool_wrapper,
    build_api_tools,
)


def _make_tool_config(
    description="A tool",
    purpose=None,
    keywords=None,
    category=None,
    tags=None,
    use_when=None,
    do_not_use_when=None,
    requires=None,
    returns=None,
):
    config = MagicMock()
    config.description = description
    config.purpose = purpose
    config.keywords = keywords or []
    config.category = category
    config.tags = tags or []
    config.use_when = use_when or []
    config.do_not_use_when = do_not_use_when or []
    config.requires = requires or []
    config.returns = returns
    return config


def _make_api_tool(config):
    tool = MagicMock()
    tool._config = config
    return tool


class TestBuildToolDescription:
    """Tests for _build_tool_description()."""

    def test_description_only(self):
        config = _make_tool_config(description="Basic tool")
        tool = _make_api_tool(config)

        result = _build_tool_description(tool)

        assert result == "Basic tool"

    def test_with_purpose(self):
        config = _make_tool_config(description="Tool", purpose="Fetch data")
        tool = _make_api_tool(config)

        result = _build_tool_description(tool)

        assert "Purpose: Fetch data" in result

    def test_with_keywords(self):
        config = _make_tool_config(description="Tool", keywords=["search", "query"])
        tool = _make_api_tool(config)

        result = _build_tool_description(tool)

        assert "Keywords: search, query" in result

    def test_with_category(self):
        config = _make_tool_config(description="Tool", category="data")
        tool = _make_api_tool(config)

        result = _build_tool_description(tool)

        assert "Category: data" in result

    def test_with_tags(self):
        config = _make_tool_config(description="Tool", tags=["web", "api"])
        tool = _make_api_tool(config)

        result = _build_tool_description(tool)

        assert "Tags: web, api" in result

    def test_with_use_when(self):
        config = _make_tool_config(description="Tool", use_when=["when A", "when B"])
        tool = _make_api_tool(config)

        result = _build_tool_description(tool)

        assert "Use when:" in result
        assert "when A" in result
        assert "when B" in result

    def test_with_do_not_use_when(self):
        config = _make_tool_config(
            description="Tool", do_not_use_when=["never X", "never Y"]
        )
        tool = _make_api_tool(config)

        result = _build_tool_description(tool)

        assert "Do NOT use when:" in result
        assert "never X" in result

    def test_with_requires(self):
        config = _make_tool_config(description="Tool", requires=["auth token", "user id"])
        tool = _make_api_tool(config)

        result = _build_tool_description(tool)

        assert "Requires:" in result
        assert "auth token" in result

    def test_with_returns(self):
        config = _make_tool_config(description="Tool", returns="JSON response")
        tool = _make_api_tool(config)

        result = _build_tool_description(tool)

        assert "Returns: JSON response" in result

    def test_all_fields(self):
        config = _make_tool_config(
            description="Full tool",
            purpose="Do everything",
            keywords=["k1"],
            category="cat",
            tags=["t1"],
            use_when=["always"],
            do_not_use_when=["never"],
            requires=["req1"],
            returns="data",
        )
        tool = _make_api_tool(config)

        result = _build_tool_description(tool)

        assert "Full tool" in result
        assert "Do everything" in result
        assert "k1" in result
        assert "cat" in result
        assert "t1" in result
        assert "always" in result
        assert "never" in result
        assert "req1" in result
        assert "data" in result


class TestCreateApiToolWrapper:
    """Tests for _create_api_tool_wrapper()."""

    @pytest.mark.asyncio
    async def test_success_returns_data_string(self):
        wrapper = _create_api_tool_wrapper("my_tool")

        registry = MagicMock()
        registry.execute = AsyncMock(return_value=MagicMock(success=True, data="result data"))

        ctx = MagicMock()
        ctx.deps.tool_registry = registry
        ctx.deps.document_scope = None

        result = await wrapper(ctx, param="val")

        assert result == "result data"

    @pytest.mark.asyncio
    async def test_success_with_none_data_returns_default(self):
        wrapper = _create_api_tool_wrapper("my_tool")

        registry = MagicMock()
        registry.execute = AsyncMock(return_value=MagicMock(success=True, data=None))

        ctx = MagicMock()
        ctx.deps.tool_registry = registry
        ctx.deps.document_scope = None

        result = await wrapper(ctx)

        assert result == "Tool executed successfully"

    @pytest.mark.asyncio
    async def test_failure_returns_error_string(self):
        wrapper = _create_api_tool_wrapper("my_tool")

        registry = MagicMock()
        registry.execute = AsyncMock(
            return_value=MagicMock(success=False, error="something went wrong")
        )

        ctx = MagicMock()
        ctx.deps.tool_registry = registry
        ctx.deps.document_scope = None

        result = await wrapper(ctx)

        assert "something went wrong" in result

    @pytest.mark.asyncio
    async def test_exception_returns_error_string(self):
        wrapper = _create_api_tool_wrapper("my_tool")

        registry = MagicMock()
        registry.execute = AsyncMock(side_effect=RuntimeError("network error"))

        ctx = MagicMock()
        ctx.deps.tool_registry = registry
        ctx.deps.document_scope = None

        result = await wrapper(ctx)

        assert "failed" in result

    @pytest.mark.asyncio
    async def test_sets_document_scope_before_execute(self):
        wrapper = _create_api_tool_wrapper("my_tool")

        registry = MagicMock()
        registry.execute = AsyncMock(return_value=MagicMock(success=True, data="ok"))

        ctx = MagicMock()
        ctx.deps.tool_registry = registry
        ctx.deps.document_scope = {"doc1", "doc2"}

        await wrapper(ctx)

        registry.set_document_scope.assert_called_once_with({"doc1", "doc2"})


class TestBuildApiTools:
    """Tests for build_api_tools()."""

    def test_empty_registry_returns_empty_list(self):
        registry = MagicMock()
        registry.get_all_tools.return_value = []

        result = build_api_tools(registry)

        assert result == []

    def test_non_api_tool_without_user_id_skipped(self):
        """Tools without user_id attribute are skipped."""
        registry = MagicMock()
        plain_tool = MagicMock(spec=[])  # no user_id attr
        registry.get_all_tools.return_value = [plain_tool]

        result = build_api_tools(registry)

        assert result == []

    def test_non_api_tool_type_with_user_id_skipped(self):
        """Tools with user_id but not ApiTool type are skipped."""
        from practorflow.llm.tools.api.tool import ApiTool

        registry = MagicMock()
        tool = MagicMock()
        tool.user_id = "u1"
        # not an instance of ApiTool
        registry.get_all_tools.return_value = [tool]

        result = build_api_tools(registry)

        assert result == []

    def test_exception_during_build_skips_tool(self):
        """Exceptions during tool building are caught and the tool is skipped."""
        from practorflow.llm.tools.api.tool import ApiTool

        registry = MagicMock()
        tool = MagicMock(spec=ApiTool)
        tool.user_id = "u1"
        tool.name = "bad_tool"
        tool.get_schema.side_effect = RuntimeError("schema error")
        registry.get_all_tools.return_value = [tool]

        result = build_api_tools(registry)

        assert result == []

    def test_success_path_returns_tool(self):
        """build_api_tools() builds and returns a Tool for a valid ApiTool."""
        from practorflow.llm.tools.api.tool import ApiTool
        from pydantic_ai import Tool
        from unittest.mock import patch

        registry = MagicMock()
        tool = MagicMock(spec=ApiTool)
        tool.user_id = "u1"
        tool.name = "my_tool"
        tool.get_schema.return_value = {
            "function": {
                "name": "my_tool",
                "parameters": {
                    "type": "object",
                    "properties": {"q": {"type": "string"}},
                    "required": ["q"],
                },
            }
        }
        config = _make_tool_config(description="A tool")
        tool._config = config
        registry.get_all_tools.return_value = [tool]

        mock_pydantic_tool = MagicMock(spec=Tool)
        with patch(
            "practorflow.services.tools.builders.api_tools.Tool.from_schema",
            return_value=mock_pydantic_tool,
        ):
            result = build_api_tools(registry)

        assert len(result) == 1
        assert result[0] is mock_pydantic_tool

    def test_success_path_uses_name_from_schema(self):
        """build_api_tools() uses function name from schema, not tool.name."""
        from practorflow.llm.tools.api.tool import ApiTool
        from pydantic_ai import Tool
        from unittest.mock import patch, call

        registry = MagicMock()
        tool = MagicMock(spec=ApiTool)
        tool.user_id = "u1"
        tool.name = "original_name"
        tool.get_schema.return_value = {
            "function": {
                "name": "schema_name",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        }
        config = _make_tool_config(description="Tool")
        tool._config = config
        registry.get_all_tools.return_value = [tool]

        mock_pydantic_tool = MagicMock(spec=Tool)
        with patch(
            "practorflow.services.tools.builders.api_tools.Tool.from_schema",
            return_value=mock_pydantic_tool,
        ) as mock_from_schema:
            build_api_tools(registry)

        assert mock_from_schema.call_args[1]["name"] == "schema_name"

    def test_success_path_uses_default_parameters_when_missing(self):
        """build_api_tools() uses default empty parameters when schema has none."""
        from practorflow.llm.tools.api.tool import ApiTool
        from pydantic_ai import Tool
        from unittest.mock import patch

        registry = MagicMock()
        tool = MagicMock(spec=ApiTool)
        tool.user_id = "u1"
        tool.name = "tool_no_params"
        tool.get_schema.return_value = {"function": {"name": "tool_no_params"}}
        config = _make_tool_config(description="Tool")
        tool._config = config
        registry.get_all_tools.return_value = [tool]

        mock_pydantic_tool = MagicMock(spec=Tool)
        with patch(
            "practorflow.services.tools.builders.api_tools.Tool.from_schema",
            return_value=mock_pydantic_tool,
        ) as mock_from_schema:
            build_api_tools(registry)

        called_schema = mock_from_schema.call_args[1]["json_schema"]
        assert called_schema == {"type": "object", "properties": {}, "required": []}

    def test_success_path_multiple_tools(self):
        """build_api_tools() returns all successfully built tools."""
        from practorflow.llm.tools.api.tool import ApiTool
        from pydantic_ai import Tool
        from unittest.mock import patch

        registry = MagicMock()
        tools_in = []
        for i in range(3):
            t = MagicMock(spec=ApiTool)
            t.user_id = f"u{i}"
            t.name = f"tool_{i}"
            t.get_schema.return_value = {"function": {"name": f"tool_{i}"}}
            t._config = _make_tool_config(description=f"Tool {i}")
            tools_in.append(t)
        registry.get_all_tools.return_value = tools_in

        built = [MagicMock(spec=Tool) for _ in range(3)]
        with patch(
            "practorflow.services.tools.builders.api_tools.Tool.from_schema",
            side_effect=built,
        ):
            result = build_api_tools(registry)

        assert len(result) == 3
