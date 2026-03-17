"""
Tool preferences endpoints.

User-accessible endpoints for managing tool preferences and listing available tools.
"""

from typing import Dict, List

from fastapi import APIRouter, Depends

from api.auth.dependencies import get_current_user
from api.auth.schemas import UserContext
from api.dependencies import (
    get_api_tool_store,
    get_tool_preferences_store,
)
from api.routes.tools.schemas import (
    ApiToolInfo,
    BuiltinToolInfo,
    ToolPreferenceUpdateRequest,
    ToolPreferenceUpdateResponse,
    ToolsListResponse,
)
from practorflow.llm.tools.api.store.base_store import ApiToolStore
from practorflow.llm.tools.user_preferences import (
    EnabledToolEntry,
    UserToolPreferencesStore,
)
from practorflow.logger.logger import get_logger

logger = get_logger("tools-api", level="INFO")

router = APIRouter(prefix="/tools", tags=["tools"])

# Static built-in tools list — matches register_default_tools() in registration.py
BUILTIN_TOOLS: List[Dict[str, str]] = [
    {"id": "search_knowledge", "name": "search_knowledge", "description": "Search the knowledge base for relevant documents."},
    {"id": "web_search", "name": "web_search", "description": "Search the web using DuckDuckGo."},
    {"id": "web_fetch", "name": "web_fetch", "description": "Fetch and extract content from a web page."},
    {"id": "summarize_text", "name": "summarize_text", "description": "Summarize text content."},
    {"id": "json_transform", "name": "json_transform", "description": "Parse, query, and transform JSON data."},
    {"id": "calculator", "name": "calculator", "description": "Perform mathematical calculations and unit conversions."},
]


@router.get(
    "",
    response_model=ToolsListResponse,
    summary="List all available tools",
    description=(
        "List all tools across all types (Built-in, API, MCP) "
        "with their current enabled/disabled state for the requesting user."
    ),
)
async def list_tools(
    current_user: UserContext = Depends(get_current_user),
    prefs_store: UserToolPreferencesStore = Depends(get_tool_preferences_store),
    api_store: ApiToolStore = Depends(get_api_tool_store),
) -> ToolsListResponse:
    """
    List all available tools with user's enabled/disabled state.

    Args:
        current_user: Authenticated user context.
        prefs_store: User tool preferences store.
        api_store: API tool store.

    Returns:
        ToolsListResponse with all tools categorized by type.
    """
    # Get user preferences
    prefs = prefs_store.get(current_user.user_id)
    if prefs is None:
        # Create default preferences if not exist
        prefs = prefs_store.create_default(current_user.user_id)

    # Build set of enabled tool IDs for quick lookup
    enabled_tool_ids = {
        (entry.type, entry.id) for entry in prefs.enabled_tools
    }

    # Get built-in tools from static list
    builtin_tools: List[BuiltinToolInfo] = []
    for tool_def in BUILTIN_TOOLS:
        builtin_tools.append(
            BuiltinToolInfo(
                id=tool_def["id"],
                type="builtin",
                name=tool_def["name"],
                description=tool_def["description"],
                enabled=("builtin", tool_def["id"]) in enabled_tool_ids,
            )
        )

    # Get API tools
    api_tools: List[ApiToolInfo] = []
    user_api_tools = api_store.list(current_user.user_id)
    system_api_tools = api_store.list("")  # System tools have empty user_id

    for tool_config in user_api_tools + system_api_tools:
        api_tools.append(
            ApiToolInfo(
                id=tool_config.tool_id,
                type="api",
                name=tool_config.name,
                description=tool_config.description,
                system=tool_config.system,
                enabled=("api", tool_config.tool_id) in enabled_tool_ids,
            )
        )

    total_count = len(builtin_tools) + len(api_tools)

    logger.info(
        f"[Tools] Listed {total_count} tools for user {current_user.user_id}: "
        f"{len(builtin_tools)} builtin, {len(api_tools)} API"
    )

    return ToolsListResponse(
        builtin_tools=builtin_tools,
        api_tools=api_tools,
        count=total_count,
    )


@router.put(
    "/preferences",
    response_model=ToolPreferenceUpdateResponse,
    summary="Update a single tool preference",
    description="Enable or disable a single tool for the current user.",
)
async def update_preference(
    request: ToolPreferenceUpdateRequest,
    current_user: UserContext = Depends(get_current_user),
    prefs_store: UserToolPreferencesStore = Depends(get_tool_preferences_store),
) -> ToolPreferenceUpdateResponse:
    """
    Enable or disable a single tool for the current user.

    Args:
        request: Single tool preference update (type, id, enabled).
        current_user: Authenticated user context.
        prefs_store: User tool preferences store.

    Returns:
        ToolPreferenceUpdateResponse with resulting state.
    """
    prefs = prefs_store.get(current_user.user_id)
    if prefs is None:
        prefs = prefs_store.create_default(current_user.user_id)

    # Update tool entry
    if request.enabled:
        already = any(
            e.type == request.type and e.id == request.id
            for e in prefs.enabled_tools
        )
        if not already:
            prefs.enabled_tools.append(
                EnabledToolEntry(type=request.type, id=request.id)
            )
    else:
        prefs.enabled_tools = [
            e for e in prefs.enabled_tools
            if not (e.type == request.type and e.id == request.id)
        ]

    # For MCP tools, also update the server list if server_id provided
    if request.type == "mcp" and request.server_id:
        if request.enabled:
            if request.server_id not in prefs.enabled_mcp_servers:
                prefs.enabled_mcp_servers.append(request.server_id)
        else:
            # Only remove server if no other tools from it remain enabled
            has_other = any(
                e.type == "mcp" and e.id != request.id
                for e in prefs.enabled_tools
            )
            if not has_other:
                prefs.enabled_mcp_servers = [
                    s for s in prefs.enabled_mcp_servers
                    if s != request.server_id
                ]

    prefs_store.update(prefs)

    logger.info(
        f"[Tools] Updated preference {request.type}:{request.id} "
        f"enabled={request.enabled} for user {current_user.user_id}"
    )

    return ToolPreferenceUpdateResponse(
        type=request.type,
        id=request.id,
        enabled=request.enabled,
    )
