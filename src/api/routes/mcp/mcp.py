"""
MCP Server CRUD endpoints.

Admin-only endpoints for managing MCP server configurations:
- Creating, listing, getting, updating, deleting servers
- Testing server connections
- Reloading server tools
"""

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth.dependencies import require_llm_admin
from api.auth.schemas import UserContext
from api.dependencies import get_mcp_server_store
from api.routes.mcp.schemas import (
    MCPServerCreateRequest,
    MCPServerDeleteResponse,
    MCPServerListResponse,
    MCPServerReloadResponse,
    MCPServerResponse,
    MCPServerTestResponse,
    MCPServerUpdateRequest,
    MCPToolInfo,
)
from practorflow.llm.tools.mcp.client import MCPClient
from practorflow.llm.tools.mcp.store import MCPServerStore
from practorflow.llm.tools.mcp.types import MCPServerConfig
from practorflow.logger.logger import get_logger

logger = get_logger("mcp-api", level="INFO")

router = APIRouter(prefix="/mcp/servers", tags=["mcp"])


@router.post(
    "",
    response_model=MCPServerResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create MCP server",
    description=(
        "Register a new MCP server configuration. "
        "Requires llm_admin permission."
    ),
)
async def create_server(
    request: MCPServerCreateRequest,
    current_user: UserContext = Depends(require_llm_admin),
    store: MCPServerStore = Depends(get_mcp_server_store),
) -> MCPServerResponse:
    """
    Create a new MCP server configuration.

    Args:
        request: Server creation request.
        current_user: Authenticated admin user.
        store: MCP server store instance.

    Returns:
        MCPServerResponse with created server details.

    Raises:
        HTTPException: 400 if configuration is invalid, 409 if name exists.
    """
    # Validate transport-specific config
    if request.transport.value == "stdio" and not request.stdio_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="stdio_config required for stdio transport",
        )
    if request.transport.value in ("sse", "streamable_http") and not request.http_config:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="http_config required for SSE/HTTP transport",
        )

    # Check for duplicate name
    existing_servers = store.list()
    if any(s.name == request.name for s in existing_servers):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Server with name '{request.name}' already exists",
        )

    # Create config
    config = MCPServerConfig(
        name=request.name,
        transport=request.transport,
        stdio_config=request.stdio_config,
        http_config=request.http_config,
        tools=request.tools,
    )

    created = store.create(config)

    logger.info(
        f"[MCP] Created server '{created.name}' (id={created.server_id}) "
        f"by user: {current_user.user_id}"
    )

    return MCPServerResponse.from_config(created)


@router.get(
    "",
    response_model=MCPServerListResponse,
    summary="List MCP servers",
    description="List all registered MCP servers. Requires llm_admin permission.",
)
async def list_servers(
    current_user: UserContext = Depends(require_llm_admin),
    store: MCPServerStore = Depends(get_mcp_server_store),
) -> MCPServerListResponse:
    """
    List all MCP server configurations.

    Args:
        current_user: Authenticated admin user.
        store: MCP server store instance.

    Returns:
        MCPServerListResponse with all servers.
    """
    servers = store.list()

    logger.info(
        f"[MCP] Listed {len(servers)} servers by user: {current_user.user_id}"
    )

    return MCPServerListResponse(
        servers=[MCPServerResponse.from_config(s) for s in servers],
        count=len(servers),
    )


@router.get(
    "/{server_id}",
    response_model=MCPServerResponse,
    summary="Get MCP server",
    description="Get a single MCP server configuration by ID. Requires llm_admin permission.",
)
async def get_server(
    server_id: str,
    current_user: UserContext = Depends(require_llm_admin),
    store: MCPServerStore = Depends(get_mcp_server_store),
) -> MCPServerResponse:
    """
    Get a single MCP server configuration.

    Args:
        server_id: Server ID from path.
        current_user: Authenticated admin user.
        store: MCP server store instance.

    Returns:
        MCPServerResponse with server details.

    Raises:
        HTTPException: 404 if server not found.
    """
    config = store.get(server_id)
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server '{server_id}' not found",
        )

    logger.info(
        f"[MCP] Retrieved server '{config.name}' (id={server_id}) "
        f"by user: {current_user.user_id}"
    )

    return MCPServerResponse.from_config(config)


@router.put(
    "/{server_id}",
    response_model=MCPServerResponse,
    summary="Update MCP server",
    description=(
        "Update an MCP server configuration. "
        "Only provided fields are updated. Requires llm_admin permission."
    ),
)
async def update_server(
    server_id: str,
    request: MCPServerUpdateRequest,
    current_user: UserContext = Depends(require_llm_admin),
    store: MCPServerStore = Depends(get_mcp_server_store),
) -> MCPServerResponse:
    """
    Update an MCP server configuration.

    Args:
        server_id: Server ID from path.
        request: Server update request (partial).
        current_user: Authenticated admin user.
        store: MCP server store instance.

    Returns:
        MCPServerResponse with updated server.

    Raises:
        HTTPException: 404 if server not found, 409 if name conflict.
    """
    config = store.get(server_id)
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server '{server_id}' not found",
        )

    # Check for name conflict
    if request.name is not None and request.name != config.name:
        existing_servers = store.list()
        if any(s.name == request.name and s.server_id != server_id for s in existing_servers):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Server with name '{request.name}' already exists",
            )

    # Apply updates
    update_data = request.model_dump(exclude_none=True)
    for field_name, value in update_data.items():
        setattr(config, field_name, value)

    config.updated_at = datetime.now()

    updated = store.update(config)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update server",
        )

    logger.info(
        f"[MCP] Updated server '{updated.name}' (id={server_id}) "
        f"by user: {current_user.user_id}"
    )

    return MCPServerResponse.from_config(updated)


@router.delete(
    "/{server_id}",
    response_model=MCPServerDeleteResponse,
    summary="Delete MCP server",
    description="Delete an MCP server configuration. Requires llm_admin permission.",
)
async def delete_server(
    server_id: str,
    current_user: UserContext = Depends(require_llm_admin),
    store: MCPServerStore = Depends(get_mcp_server_store),
) -> MCPServerDeleteResponse:
    """
    Delete an MCP server configuration.

    Args:
        server_id: Server ID from path.
        current_user: Authenticated admin user.
        store: MCP server store instance.

    Returns:
        MCPServerDeleteResponse with deletion status.

    Raises:
        HTTPException: 404 if server not found.
    """
    config = store.get(server_id)
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server '{server_id}' not found",
        )

    deleted = store.delete(server_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete server",
        )

    logger.info(
        f"[MCP] Deleted server '{config.name}' (id={server_id}) "
        f"by user: {current_user.user_id}"
    )

    return MCPServerDeleteResponse(
        server_id=server_id,
        deleted=True,
        message="Server deleted successfully",
    )


@router.post(
    "/{server_id}/test",
    response_model=MCPServerTestResponse,
    summary="Test MCP server connection",
    description=(
        "Test connection to an MCP server and list available tools. "
        "Requires llm_admin permission."
    ),
)
async def test_server(
    server_id: str,
    current_user: UserContext = Depends(require_llm_admin),
    store: MCPServerStore = Depends(get_mcp_server_store),
) -> MCPServerTestResponse:
    """
    Test MCP server connection and list available tools.

    Admin uses this to see what tools a server offers before creating them.

    Args:
        server_id: Server ID from path.
        current_user: Authenticated admin user.
        store: MCP server store instance.

    Returns:
        MCPServerTestResponse with connection status and tools.

    Raises:
        HTTPException: 404 if server not found.
    """
    config = store.get(server_id)
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server '{server_id}' not found",
        )

    client = MCPClient(config)
    tools: List[MCPToolInfo] = []
    error_msg: str = None
    connected = False

    try:
        # Attempt connection
        await client.connect()
        connected = True

        # List available tools
        tools_data = await client.list_available_tools()
        tools = [
            MCPToolInfo(
                name=t["name"],
                description=t["description"],
                input_schema=t["inputSchema"],
            )
            for t in tools_data
        ]

        logger.info(
            f"[MCP] Tested server '{config.name}' (id={server_id}): "
            f"connected={connected}, tools={len(tools)} "
            f"by user: {current_user.user_id}"
        )

    except Exception as e:
        error_msg = str(e)
        logger.error(
            f"[MCP] Test failed for server '{config.name}' (id={server_id}): {e}"
        )

    finally:
        # Always disconnect
        try:
            await client.disconnect()
        except Exception as e:
            logger.warning(f"[MCP] Error during test cleanup: {e}")

    return MCPServerTestResponse(
        server_id=server_id,
        connected=connected,
        tools=tools,
        error=error_msg,
    )


@router.post(
    "/{server_id}/reload",
    response_model=MCPServerReloadResponse,
    summary="Reload MCP server",
    description=(
        "Reconnect to MCP server and refresh tools in registry. "
        "Requires llm_admin permission."
    ),
)
async def reload_server(
    server_id: str,
    current_user: UserContext = Depends(require_llm_admin),
    store: MCPServerStore = Depends(get_mcp_server_store),
) -> MCPServerReloadResponse:
    """
    Reload MCP server connection and refresh tools.

    This endpoint would trigger ToolRegistry to reload the server's tools.
    Implementation requires integration with ToolRegistry.

    Args:
        server_id: Server ID from path.
        current_user: Authenticated admin user.
        store: MCP server store instance.

    Returns:
        MCPServerReloadResponse with reload status.

    Raises:
        HTTPException: 404 if server not found, 501 if not implemented.
    """
    config = store.get(server_id)
    if config is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Server '{server_id}' not found",
        )

    # TODO: Integrate with ToolRegistry to reload server tools
    # For now, return not implemented
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Server reload requires ToolRegistry integration (Phase 4)",
    )
