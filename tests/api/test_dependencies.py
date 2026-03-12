import sys
import types
import pytest
from types import SimpleNamespace


@pytest.fixture(scope="module")
def mock_external_dependencies():
    """Mock external dependencies for api.dependencies module only."""
    modules_to_mock = {
        "practorflow": types.ModuleType("practorflow"),
        "practorflow.services": types.ModuleType("practorflow.services"),
        "practorflow.services.chat": types.ModuleType("practorflow.services.chat"),
        "practorflow.services.agent": types.ModuleType("practorflow.services.agent"),
        "practorflow.services.history": types.ModuleType("practorflow.services.history"),
        "practorflow.services.history.truncator": types.ModuleType(
            "practorflow.services.history.truncator"
        ),
        "practorflow.session_store": types.ModuleType("practorflow.session_store"),
        "practorflow.session_store.session_history": types.ModuleType(
            "practorflow.session_store.session_history"
        ),
        "api.auth": types.ModuleType("api.auth"),
        "api.auth.service": types.ModuleType("api.auth.service"),
    }

    modules_to_mock["practorflow.services.chat"].ChatService = type("ChatService", (), {})
    modules_to_mock["practorflow.services.agent"].AgentService = type("AgentService", (), {})
    modules_to_mock["practorflow.services.history.truncator"].DeleteSessionService = type(
        "DeleteSessionService", (), {}
    )
    modules_to_mock["practorflow.session_store.session_history"].SessionHistory = type(
        "SessionHistory", (), {}
    )
    modules_to_mock["api.auth.service"].AuthService = type("AuthService", (), {})

    original = {}
    for name, module in modules_to_mock.items():
        if name in sys.modules:
            original[name] = sys.modules[name]
        sys.modules[name] = module

    # Import after mocks are in place
    from api.dependencies import (
        container,
        get_chat_service,
        get_agent_service,
        get_session_history,
        get_auth_service,
        get_delete_session_service,
        get_api_tool_store,
        get_encryption_service,
        get_mcp_server_store,
        get_tool_preferences_store,
        resolve_tool,
    )

    yield {
        "container": container,
        "get_chat_service": get_chat_service,
        "get_agent_service": get_agent_service,
        "get_session_history": get_session_history,
        "get_auth_service": get_auth_service,
        "get_delete_session_service": get_delete_session_service,
        "get_api_tool_store": get_api_tool_store,
        "get_encryption_service": get_encryption_service,
        "get_mcp_server_store": get_mcp_server_store,
        "get_tool_preferences_store": get_tool_preferences_store,
        "resolve_tool": resolve_tool,
    }

    # Cleanup: restore original modules
    for name in modules_to_mock:
        if name in original:
            sys.modules[name] = original[name]
        else:
            sys.modules.pop(name, None)

    # Remove cached api.dependencies
    sys.modules.pop("api.dependencies", None)


@pytest.fixture(autouse=True)
def reset_container(mock_external_dependencies):
    """Reset container before and after each test."""
    container = mock_external_dependencies["container"]
    container.chat_service = None
    container.agent_service = None
    container.session_history = None
    container.auth_service = None
    container.delete_session_service = None
    container.api_tool_store = None
    container.encryption_service = None
    container.mcp_server_store = None
    container.tool_preferences_store = None
    yield
    container.chat_service = None
    container.agent_service = None
    container.session_history = None
    container.auth_service = None
    container.delete_session_service = None
    container.api_tool_store = None
    container.encryption_service = None
    container.mcp_server_store = None
    container.tool_preferences_store = None


def test_get_chat_service_success(mock_external_dependencies):
    container = mock_external_dependencies["container"]
    get_chat_service = mock_external_dependencies["get_chat_service"]
    
    fake = SimpleNamespace()
    container.chat_service = fake
    assert get_chat_service() is fake


def test_get_chat_service_not_initialized(mock_external_dependencies):
    get_chat_service = mock_external_dependencies["get_chat_service"]
    
    with pytest.raises(RuntimeError):
        get_chat_service()


def test_get_agent_service_success(mock_external_dependencies):
    container = mock_external_dependencies["container"]
    get_agent_service = mock_external_dependencies["get_agent_service"]
    
    fake = SimpleNamespace()
    container.agent_service = fake
    assert get_agent_service() is fake


def test_get_agent_service_not_initialized(mock_external_dependencies):
    get_agent_service = mock_external_dependencies["get_agent_service"]
    
    with pytest.raises(RuntimeError):
        get_agent_service()


def test_get_session_history_success(mock_external_dependencies):
    container = mock_external_dependencies["container"]
    get_session_history = mock_external_dependencies["get_session_history"]
    
    fake = SimpleNamespace()
    container.session_history = fake
    assert get_session_history() is fake


def test_get_session_history_not_initialized(mock_external_dependencies):
    get_session_history = mock_external_dependencies["get_session_history"]
    
    with pytest.raises(RuntimeError):
        get_session_history()


def test_get_auth_service_success(mock_external_dependencies):
    container = mock_external_dependencies["container"]
    get_auth_service = mock_external_dependencies["get_auth_service"]
    
    fake = SimpleNamespace()
    container.auth_service = fake
    assert get_auth_service() is fake


def test_get_auth_service_not_initialized(mock_external_dependencies):
    get_auth_service = mock_external_dependencies["get_auth_service"]
    
    with pytest.raises(RuntimeError):
        get_auth_service()


def test_get_delete_session_service_success(mock_external_dependencies):
    container = mock_external_dependencies["container"]
    get_delete_session_service = mock_external_dependencies["get_delete_session_service"]
    
    fake = SimpleNamespace()
    container.delete_session_service = fake
    assert get_delete_session_service() is fake


def test_get_delete_session_service_not_initialized(mock_external_dependencies):
    get_delete_session_service = mock_external_dependencies["get_delete_session_service"]

    with pytest.raises(RuntimeError):
        get_delete_session_service()


def test_get_api_tool_store_success(mock_external_dependencies):
    container = mock_external_dependencies["container"]
    get_api_tool_store = mock_external_dependencies["get_api_tool_store"]

    fake = SimpleNamespace()
    container.api_tool_store = fake
    assert get_api_tool_store() is fake


def test_get_api_tool_store_not_initialized(mock_external_dependencies):
    get_api_tool_store = mock_external_dependencies["get_api_tool_store"]

    with pytest.raises(RuntimeError):
        get_api_tool_store()


def test_get_encryption_service_success(mock_external_dependencies):
    container = mock_external_dependencies["container"]
    get_encryption_service = mock_external_dependencies["get_encryption_service"]

    fake = SimpleNamespace()
    container.encryption_service = fake
    assert get_encryption_service() is fake


def test_get_encryption_service_not_initialized(mock_external_dependencies):
    get_encryption_service = mock_external_dependencies["get_encryption_service"]

    with pytest.raises(RuntimeError):
        get_encryption_service()


def test_get_mcp_server_store_success(mock_external_dependencies):
    container = mock_external_dependencies["container"]
    get_mcp_server_store = mock_external_dependencies["get_mcp_server_store"]

    fake = SimpleNamespace()
    container.mcp_server_store = fake
    assert get_mcp_server_store() is fake


def test_get_mcp_server_store_not_initialized(mock_external_dependencies):
    get_mcp_server_store = mock_external_dependencies["get_mcp_server_store"]

    with pytest.raises(RuntimeError):
        get_mcp_server_store()


def test_get_tool_preferences_store_success(mock_external_dependencies):
    container = mock_external_dependencies["container"]
    get_tool_preferences_store = mock_external_dependencies["get_tool_preferences_store"]

    fake = SimpleNamespace()
    container.tool_preferences_store = fake
    assert get_tool_preferences_store() is fake


def test_get_tool_preferences_store_not_initialized(mock_external_dependencies):
    get_tool_preferences_store = mock_external_dependencies["get_tool_preferences_store"]

    with pytest.raises(RuntimeError):
        get_tool_preferences_store()


async def test_resolve_tool_not_found(mock_external_dependencies):
    from fastapi import HTTPException

    resolve_tool = mock_external_dependencies["resolve_tool"]
    mock_store = SimpleNamespace(get=lambda tool_id: None)
    user = SimpleNamespace(user_id="user1", permissions=[])

    with pytest.raises(HTTPException) as exc_info:
        await resolve_tool("nonexistent", user, mock_store)
    assert exc_info.value.status_code == 404


async def test_resolve_tool_user_owns_tool(mock_external_dependencies):
    resolve_tool = mock_external_dependencies["resolve_tool"]
    tool = SimpleNamespace(user_id="user1", system=False)
    mock_store = SimpleNamespace(get=lambda tool_id: tool)
    user = SimpleNamespace(user_id="user1", permissions=[])

    result = await resolve_tool("tool1", user, mock_store)
    assert result is tool


async def test_resolve_tool_system_tool_with_admin(mock_external_dependencies):
    resolve_tool = mock_external_dependencies["resolve_tool"]
    tool = SimpleNamespace(user_id="", system=True)
    mock_store = SimpleNamespace(get=lambda tool_id: tool)
    user = SimpleNamespace(user_id="user1", permissions=["llm_admin"])

    result = await resolve_tool("tool1", user, mock_store)
    assert result is tool


async def test_resolve_tool_forbidden(mock_external_dependencies):
    from fastapi import HTTPException

    resolve_tool = mock_external_dependencies["resolve_tool"]
    tool = SimpleNamespace(user_id="other_user", system=False)
    mock_store = SimpleNamespace(get=lambda tool_id: tool)
    user = SimpleNamespace(user_id="user1", permissions=[])

    with pytest.raises(HTTPException) as exc_info:
        await resolve_tool("tool1", user, mock_store)
    assert exc_info.value.status_code == 403