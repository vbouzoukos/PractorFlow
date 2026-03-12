"""
Unit tests for api.main module.

Uses unittest.mock to patch all dependencies at import time.
Mocks are scoped to this module only using importlib.
"""

import sys
import types
import os
from unittest.mock import MagicMock, AsyncMock, patch
import pytest


# ===================== FIXTURE-BASED ISOLATED MOCKING =====================

@pytest.fixture(scope="module")
def main_module():
    """
    Import api.main with all dependencies mocked.
    
    This fixture isolates mocks to this test module only by:
    1. Saving original sys.modules state
    2. Adding mocks
    3. Importing api.main
    4. Restoring sys.modules after all tests complete
    """
    # Save original modules
    original_modules = dict(sys.modules)
    
    # Track modules we add
    added_modules = set()
    
    def create_mock_module(name, **attrs):
        """Create a mock module with given attributes."""
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        if name not in original_modules:
            added_modules.add(name)
        sys.modules[name] = mod
        return mod
    
    # Remove any cached api.* modules
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith("api."):
            del sys.modules[mod_name]
            added_modules.add(mod_name)
    
    # ---------- Mock classes ----------
    
    class MockModelPool:
        @staticmethod
        def get_instance(**kwargs):
            instance = MagicMock()
            instance.preload = AsyncMock()
            instance.unload_all = AsyncMock()
            return instance
    
    class MockCleanupScheduler:
        def __init__(self, **kwargs):
            pass
        def start(self):
            pass
        async def stop(self):
            pass
    
    class MockAuthConfig:
        is_open_mode = True
        is_oidc_mode = False
        oidc = MagicMock(issuer_url="http://test-issuer")
    
    class MockCleanupConfig:
        is_enabled = True
        interval_minutes = 1
    
    class MockAPIConfig:
        auth = MockAuthConfig()
        cleanup = MockCleanupConfig()
    
    class MockModelConfig:
        model_name = "test-model"
    
    class MockAppConfiguration:
        ModelConfiguration = MockModelConfig()
        KnowledgeChromaConfiguration = MagicMock()
    
    class MockLogger:
        def info(self, *args, **kwargs): pass
        def debug(self, *args, **kwargs): pass
        def warning(self, *args, **kwargs): pass
        def error(self, *args, **kwargs): pass
    
    class MockKnowledgeStore:
        def count_documents(self):
            return 0
    
    class MockServiceContainer:
        chat_service = None
        agent_service = None
        session_history = None
        auth_service = None
        delete_session_service = None
    
    class MockFastAPI:
        def __init__(self, **kwargs):
            pass
        def add_middleware(self, *args, **kwargs):
            pass
        def include_router(self, *args, **kwargs):
            pass
        def get(self, *args, **kwargs):
            def decorator(func):
                return func
            return decorator
    
    # ---------- Setup mocks ----------
    
    # practorflow.settings.app_settings
    app_settings_mod = create_mock_module(
        "practorflow.settings.app_settings",
        load_configuration=MagicMock(),
        appConfiguration=MockAppConfiguration(),
    )
    create_mock_module("practorflow.settings", app_settings=app_settings_mod)
    
    # api.config
    api_settings_submod = create_mock_module(
        "api.config.api_settings",
        load_api_configuration=MagicMock(),
        get_api_configuration=MagicMock(return_value=MockAPIConfig()),
        apiConfiguration=MockAPIConfig(),
    )
    create_mock_module(
        "api.config",
        load_api_configuration=MagicMock(),
        get_api_configuration=MagicMock(return_value=MockAPIConfig()),
        api_settings=api_settings_submod,
    )
    
    # practorflow.services.history.truncator
    create_mock_module(
        "practorflow.services.history.truncator",
        DeleteSessionService=MagicMock(return_value=MagicMock()),
    )
    create_mock_module("practorflow.services.history")
    create_mock_module("practorflow.services")
    
    # practorflow.logger.logger
    create_mock_module(
        "practorflow.logger.logger",
        get_logger=MagicMock(return_value=MockLogger()),
    )
    create_mock_module("practorflow.logger")
    
    # practorflow.llm
    create_mock_module("practorflow.llm", ModelPool=MockModelPool)
    
    # practorflow.llm.knowledge.chroma_knowledge_store
    create_mock_module(
        "practorflow.llm.knowledge.chroma_knowledge_store",
        ChromaKnowledgeStore=MagicMock(return_value=MockKnowledgeStore()),
    )
    create_mock_module("practorflow.llm.knowledge")
    
    # practorflow.services.chat
    create_mock_module(
        "practorflow.services.chat",
        ChatService=MagicMock(return_value=MagicMock()),
    )
    
    # practorflow.services.agent.agent_service
    create_mock_module(
        "practorflow.services.agent.agent_service",
        AgentService=MagicMock(return_value=MagicMock()),
    )
    create_mock_module(
        "practorflow.services.agent",
        AgentService=MagicMock(return_value=MagicMock()),
    )
    
    # practorflow.session_store.factory
    create_mock_module(
        "practorflow.session_store.factory",
        create_session_store=MagicMock(return_value=MagicMock()),
        create_session_history=MagicMock(return_value=MagicMock()),
    )
    create_mock_module("practorflow.session_store")
    create_mock_module(
        "practorflow.session_store.session_history",
        SessionHistory=MagicMock,
    )
    
    # api.dependencies
    create_mock_module(
        "api.dependencies",
        container=MockServiceContainer(),
        ServiceContainer=MockServiceContainer,
    )
    
    # api.auth
    create_mock_module(
        "api.auth",
        AuthService=MagicMock(return_value=MagicMock()),
        set_auth_service=MagicMock(),
    )
    create_mock_module(
        "api.auth.service",
        AuthService=MagicMock(return_value=MagicMock()),
    )
    
    # api.routes
    for route_name in (
        "api.routes.auth",
        "api.routes.chat",
        "api.routes.agent",
        "api.routes.session",
        "api.routes.api_tools",
        "api.routes.mcp",
        "api.routes.tools",
    ):
        create_mock_module(route_name, router=MagicMock())
    create_mock_module("api.routes")
    
    # api.services.maintenance.orphan_cleanup_service
    create_mock_module(
        "api.services.maintenance.orphan_cleanup_service",
        OrphanCleanupService=MagicMock(return_value=MagicMock()),
    )
    create_mock_module("api.services.maintenance")
    create_mock_module("api.services")
    
    # api.scheduler.cleanup_scheduler
    create_mock_module(
        "api.scheduler.cleanup_scheduler",
        CleanupScheduler=MockCleanupScheduler,
    )
    create_mock_module("api.scheduler")
    
    # fastapi
    create_mock_module("fastapi", FastAPI=MockFastAPI)
    create_mock_module("fastapi.middleware")
    create_mock_module("fastapi.middleware.cors", CORSMiddleware=MagicMock())
    
    # uvicorn
    uvicorn_run_mock = MagicMock()
    create_mock_module("uvicorn", run=uvicorn_run_mock)

    # practorflow.llm.tools.api submodules
    create_mock_module(
        "practorflow.llm.tools.api.encryption",
        EncryptionService=MagicMock(return_value=MagicMock()),
    )
    create_mock_module(
        "practorflow.llm.tools.api.factory",
        initialize_factory=MagicMock(),
    )
    create_mock_module("practorflow.llm.tools.api")
    create_mock_module("practorflow.llm.tools")

    # practorflow.tool_store submodules
    create_mock_module(
        "practorflow.tool_store.tinydb_api_store",
        TinyDBApiToolStore=MagicMock(return_value=MagicMock()),
    )
    create_mock_module(
        "practorflow.tool_store.tinydb_mcp_server_store",
        TinyDBMCPServerStore=MagicMock(return_value=MagicMock()),
    )
    create_mock_module(
        "practorflow.tool_store.tinydb_user_preferences",
        TinyDBUserToolPreferencesStore=MagicMock(return_value=MagicMock()),
    )
    create_mock_module("practorflow.tool_store")

    # practorflow root
    create_mock_module("practorflow")
    
    # ---------- Import module under test ----------
    import api.main as main
    
    # Attach uvicorn mock for test assertions
    main._test_uvicorn_run = uvicorn_run_mock
    main._test_MockAPIConfig = MockAPIConfig
    
    yield main
    
    # ---------- Cleanup: restore original modules ----------
    for mod_name in added_modules:
        if mod_name in sys.modules:
            del sys.modules[mod_name]
    
    # Restore any originals that were overwritten
    for mod_name, mod in original_modules.items():
        sys.modules[mod_name] = mod


# ===================== TESTS =====================

class TestLifespan:
    """Tests for the lifespan context manager."""
    
    @pytest.mark.asyncio
    async def test_lifespan_starts_and_stops_open_mode(self, main_module):
        """Test lifespan in open auth mode (default)."""
        async with main_module.lifespan(MagicMock()):
            pass


class TestLifespanAuthModes:
    """Tests specifically for authentication mode branches in lifespan."""
    
    @pytest.mark.asyncio
    async def test_oidc_auth_mode_branch(self, main_module):
        """Cover: elif api_config.auth.is_oidc_mode branch."""
        
        class OIDCAuth:
            is_open_mode = False
            is_oidc_mode = True
            oidc = type("OIDC", (), {"issuer_url": "http://oidc.example.com"})()
        
        class OIDCConfig:
            auth = OIDCAuth()
            cleanup = type("Cleanup", (), {"is_enabled": False, "interval_minutes": 0})()
        
        original = main_module.get_api_configuration
        main_module.get_api_configuration = lambda: OIDCConfig()
        
        try:
            async with main_module.lifespan(MagicMock()):
                pass
        finally:
            main_module.get_api_configuration = original
    
    @pytest.mark.asyncio
    async def test_local_auth_mode_branch(self, main_module):
        """Cover: else branch (Local mode - app secret required)."""
        
        class LocalAuth:
            is_open_mode = False
            is_oidc_mode = False
            oidc = type("OIDC", (), {"issuer_url": ""})()
        
        class LocalConfig:
            auth = LocalAuth()
            cleanup = type("Cleanup", (), {"is_enabled": False, "interval_minutes": 0})()
        
        original = main_module.get_api_configuration
        main_module.get_api_configuration = lambda: LocalConfig()
        
        try:
            async with main_module.lifespan(MagicMock()):
                pass
        finally:
            main_module.get_api_configuration = original


class TestLifespanCleanupScheduler:
    """Tests specifically for cleanup scheduler branches in lifespan."""
    
    @pytest.mark.asyncio
    async def test_cleanup_scheduler_disabled_branch(self, main_module):
        """Cover: else branch - logger.info('[API] Cleanup scheduler disabled')."""
        
        class OpenAuth:
            is_open_mode = True
            is_oidc_mode = False
            oidc = type("OIDC", (), {"issuer_url": ""})()
        
        class DisabledCleanupConfig:
            auth = OpenAuth()
            cleanup = type("Cleanup", (), {"is_enabled": False, "interval_minutes": 0})()
        
        original = main_module.get_api_configuration
        main_module.get_api_configuration = lambda: DisabledCleanupConfig()
        
        try:
            async with main_module.lifespan(MagicMock()):
                pass
        finally:
            main_module.get_api_configuration = original


class TestHealthCheck:
    """Tests for the health check endpoint."""
    
    @pytest.mark.asyncio
    async def test_health_check_returns_healthy(self, main_module):
        """Test that health check returns healthy status."""
        result = await main_module.health_check()
        assert result == {"status": "healthy"}


class TestRunFunctions:
    """Tests for the run and run_debug functions."""
    
    def test_run(self, main_module):
        """Test that run() calls _run with reload=False."""
        with patch.object(main_module, "_run") as mock_run:
            main_module.run()
            mock_run.assert_called_once_with(reload=False)
    
    def test_run_debug(self, main_module):
        """Test that run_debug() calls _run with reload=True."""
        with patch.object(main_module, "_run") as mock_run:
            main_module.run_debug()
            mock_run.assert_called_once_with(reload=True)
    
    def test__run_calls_uvicorn(self, main_module):
        """Test that _run calls uvicorn.run with correct parameters."""
        os.environ["PRACTORFLOW_API_HOST"] = "0.0.0.0"
        os.environ["PRACTORFLOW_API_PORT"] = "9000"
        
        main_module._test_uvicorn_run.reset_mock()
        
        try:
            main_module._run(reload=False)
            
            main_module._test_uvicorn_run.assert_called_once_with(
                "api.main:app",
                host="0.0.0.0",
                port=9000,
                reload=False,
            )
        finally:
            os.environ.pop("PRACTORFLOW_API_HOST", None)
            os.environ.pop("PRACTORFLOW_API_PORT", None)
    
    def test__run_uses_default_host_and_port(self, main_module):
        """Test that _run uses default host and port when not set."""
        os.environ.pop("PRACTORFLOW_API_HOST", None)
        os.environ.pop("PRACTORFLOW_API_PORT", None)
        
        main_module._test_uvicorn_run.reset_mock()
        
        main_module._run(reload=True)
        
        main_module._test_uvicorn_run.assert_called_once_with(
            "api.main:app",
            host="localhost",
            port=8000,
            reload=True,
        )


class TestAppCreation:
    """Tests for FastAPI app creation."""
    
    def test_app_exists(self, main_module):
        """Test that app is created."""
        assert hasattr(main_module, "app")
        assert main_module.app is not None
    
    def test_lifespan_function_exists(self, main_module):
        """Test that lifespan function exists."""
        assert hasattr(main_module, "lifespan")
        assert callable(main_module.lifespan)