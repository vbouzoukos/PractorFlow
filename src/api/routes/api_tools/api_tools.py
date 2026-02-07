"""
API Tool CRUD endpoints.

Provides endpoints for managing API tool configurations:
- Listing tools (user + system)
- Creating tools (user or system with llm_admin)
- Getting, updating, deleting tools (ownership or llm_admin for system)
- Unmasking secrets (ownership or llm_admin for system)
"""

from fastapi import APIRouter, Depends, HTTPException, status

from api.auth.dependencies import get_current_user
from api.auth.schemas import UserContext
from api.dependencies import get_api_tool_store, get_encryption_service, resolve_tool
from api.routes.api_tools.schemas import (
    ApiToolCreateRequest,
    ApiToolListResponse,
    ApiToolResponse,
    ApiToolSecretsResponse,
    ApiToolUpdateRequest,
)
from practorflow.llm.tools.api.encryption import EncryptionService
from practorflow.llm.tools.api.factory import get_factory, is_factory_initialized
from practorflow.llm.tools.api.models.models import ApiToolConfig
from practorflow.llm.tools.api.store.base_store import ApiToolStore
from practorflow.logger.logger import get_logger

logger = get_logger("api-tools-api", level="INFO")

router = APIRouter(prefix="/api-tools", tags=["api-tools"])

SYSTEM_USER_ID = ""


@router.get(
    "",
    response_model=ApiToolListResponse,
    summary="List API tools",
    description=(
        "List all API tool configurations for the current user. "
        "Includes system tools."
    ),
)
async def list_tools(
    current_user: UserContext = Depends(get_current_user),
    store: ApiToolStore = Depends(get_api_tool_store),
) -> ApiToolListResponse:
    """
    List API tools for the current user.
    
    Returns user-owned tools plus system tools.
    
    Args:
        current_user: Authenticated user context.
        store: API tool store instance.
    
    Returns:
        ApiToolListResponse with tools and count.
    """
    logger.info(f"[API Tools] Listing tools for user: {current_user.user_id}")

    user_tools = store.list(current_user.user_id)
    system_tools = store.list(SYSTEM_USER_ID)
    all_tools = user_tools + system_tools

    logger.info(
        f"[API Tools] Found {len(user_tools)} user tools, "
        f"{len(system_tools)} system tools"
    )

    return ApiToolListResponse(
        tools=[ApiToolResponse.from_config(t) for t in all_tools],
        count=len(all_tools),
    )


@router.post(
    "",
    response_model=ApiToolResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API tool",
    description=(
        "Create a new API tool configuration. "
        "Set system=true to create a system tool (requires llm_admin). "
        "Secrets are encrypted before storage."
    ),
)
async def create_tool(
    request: ApiToolCreateRequest,
    current_user: UserContext = Depends(get_current_user),
    store: ApiToolStore = Depends(get_api_tool_store),
    encryption: EncryptionService = Depends(get_encryption_service),
) -> ApiToolResponse:
    """
    Create a new API tool configuration.
    
    Args:
        request: Tool creation request.
        current_user: Authenticated user context.
        store: API tool store instance.
        encryption: Encryption service instance.
    
    Returns:
        ApiToolResponse with created tool (secrets masked).
    
    Raises:
        HTTPException: 403 if system tool without llm_admin.
    """
    if request.system and "llm_admin" not in current_user.permissions:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="llm_admin permission required to create system tools",
        )

    config_data = request.model_dump(
        exclude={"auth_secret", "auth_username", "auth_password", "system"},
    )
    config_data["user_id"] = SYSTEM_USER_ID if request.system else current_user.user_id
    config_data["system"] = request.system

    if request.auth_secret is not None:
        config_data["auth_secret"] = encryption.encrypt(request.auth_secret)
    if request.auth_username is not None:
        config_data["auth_username"] = encryption.encrypt(request.auth_username)
    if request.auth_password is not None:
        config_data["auth_password"] = encryption.encrypt(request.auth_password)

    created = store.create(ApiToolConfig(**config_data))

    logger.info(
        f"[API Tools] Created tool '{created.name}' (id={created.tool_id}, "
        f"system={created.system}) by user: {current_user.user_id}"
    )

    return ApiToolResponse.from_config(created)


@router.get(
    "/{tool_id}",
    response_model=ApiToolResponse,
    summary="Get API tool",
    description="Get a single API tool configuration by ID. Secrets are masked.",
)
async def get_tool(
    tool: ApiToolConfig = Depends(resolve_tool),
    current_user: UserContext = Depends(get_current_user),
) -> ApiToolResponse:
    """
    Get a single API tool configuration.
    
    Args:
        tool: Resolved tool (injected by resolve_tool dependency).
        current_user: Authenticated user context.
    
    Returns:
        ApiToolResponse with masked secrets.
    """
    logger.info(
        f"[API Tools] Retrieved tool '{tool.name}' (id={tool.tool_id}) "
        f"by user: {current_user.user_id}"
    )

    return ApiToolResponse.from_config(tool)


@router.put(
    "/{tool_id}",
    response_model=ApiToolResponse,
    summary="Update API tool",
    description=(
        "Update an API tool configuration. "
        "Only provided fields are updated. Secrets are re-encrypted."
    ),
)
async def update_tool(
    request: ApiToolUpdateRequest,
    tool: ApiToolConfig = Depends(resolve_tool),
    current_user: UserContext = Depends(get_current_user),
    store: ApiToolStore = Depends(get_api_tool_store),
    encryption: EncryptionService = Depends(get_encryption_service),
) -> ApiToolResponse:
    """
    Update an API tool configuration.
    
    Args:
        request: Tool update request (partial).
        tool: Resolved tool (injected by resolve_tool dependency).
        current_user: Authenticated user context.
        store: API tool store instance.
        encryption: Encryption service instance.
    
    Returns:
        ApiToolResponse with updated tool (secrets masked).
    """
    update_data = request.model_dump(
        exclude_none=True,
        exclude={"auth_secret", "auth_username", "auth_password"},
    )
    for field_name, value in update_data.items():
        setattr(tool, field_name, value)

    if request.auth_secret is not None:
        tool.auth_secret = encryption.encrypt(request.auth_secret)
    if request.auth_username is not None:
        tool.auth_username = encryption.encrypt(request.auth_username)
    if request.auth_password is not None:
        tool.auth_password = encryption.encrypt(request.auth_password)

    updated = store.update(tool)
    if updated is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update tool",
        )

    if is_factory_initialized():
        get_factory().invalidate_rate_limiter(tool.tool_id)

    logger.info(
        f"[API Tools] Updated tool '{updated.name}' (id={updated.tool_id}) "
        f"by user: {current_user.user_id}"
    )

    return ApiToolResponse.from_config(updated)


@router.delete(
    "/{tool_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete API tool",
    description="Delete an API tool configuration.",
)
async def delete_tool(
    tool: ApiToolConfig = Depends(resolve_tool),
    current_user: UserContext = Depends(get_current_user),
    store: ApiToolStore = Depends(get_api_tool_store),
) -> None:
    """
    Delete an API tool configuration.
    
    Args:
        tool: Resolved tool (injected by resolve_tool dependency).
        current_user: Authenticated user context.
        store: API tool store instance.
    """
    deleted = store.delete(tool.tool_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete tool",
        )

    if is_factory_initialized():
        get_factory().invalidate_rate_limiter(tool.tool_id)

    logger.info(
        f"[API Tools] Deleted tool (id={tool.tool_id}) "
        f"by user: {current_user.user_id}"
    )


@router.get(
    "/{tool_id}/secrets",
    response_model=ApiToolSecretsResponse,
    summary="Get API tool secrets",
    description=(
        "Get decrypted secrets for an API tool. "
        "Requires ownership (user tool) or llm_admin (system tool)."
    ),
)
async def get_tool_secrets(
    tool: ApiToolConfig = Depends(resolve_tool),
    current_user: UserContext = Depends(get_current_user),
    encryption: EncryptionService = Depends(get_encryption_service),
) -> ApiToolSecretsResponse:
    """
    Get decrypted secrets for an API tool.
    
    Args:
        tool: Resolved tool (injected by resolve_tool dependency).
        current_user: Authenticated user context.
        encryption: Encryption service instance.
    
    Returns:
        ApiToolSecretsResponse with decrypted secret values.
    """
    try:
        decrypted_secret = encryption.decrypt(tool.auth_secret) if tool.auth_secret else None
        decrypted_username = encryption.decrypt(tool.auth_username) if tool.auth_username else None
        decrypted_password = encryption.decrypt(tool.auth_password) if tool.auth_password else None
    except Exception as e:
        logger.error(f"[API Tools] Failed to decrypt secrets for tool '{tool.tool_id}': {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to decrypt tool secrets",
        )

    logger.info(
        f"[API Tools] Secrets retrieved for tool (id={tool.tool_id}) "
        f"by user: {current_user.user_id}"
    )

    return ApiToolSecretsResponse(
        tool_id=tool.tool_id,
        auth_secret=decrypted_secret,
        auth_username=decrypted_username,
        auth_password=decrypted_password,
    )