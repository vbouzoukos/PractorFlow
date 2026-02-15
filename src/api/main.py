"""
FastAPI application entry point.

Sets up the FastAPI application with:
- Lifespan management for service initialization/cleanup
- Authentication configuration and service
- Chat routes
- Agent routes
- Session routes
- API tool routes
- CORS middleware
- Background cleanup scheduler for orphaned documents
"""

import os

# Load configuration FIRST, before any other practorflow imports
from practorflow.services.history.truncator import DeleteSessionService
from practorflow.settings.app_settings import load_configuration
from api.config import load_api_configuration

# Set config path first
os.environ.setdefault("_PRACTORFLOW_CONFIG_PATH", "./config")

# Import directly from module path, not through practorflow package
from practorflow.settings import app_settings
from api.config import api_settings

config_path = os.environ.get("_PRACTORFLOW_CONFIG_PATH", "./config")
app_settings.load_configuration(os.path.join(config_path, "llm/options"))
api_settings.load_api_configuration(os.path.join(config_path, "api"))

# Now safe to import other modules
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

config_path = os.environ.get("_PRACTORFLOW_CONFIG_PATH", "../config")
llm_options_path = os.path.join(config_path, "llm/options")
llm_api_path = os.path.join(config_path, "api")
load_configuration(llm_options_path)
load_api_configuration(llm_api_path)

from api.dependencies import container
from api.config import get_api_configuration
from api.auth import AuthService, set_auth_service
from api.routes.auth import router as auth_router
from api.routes.chat import router as chat_router
from api.routes.agent import router as agent_router
from api.routes.session import router as session_router
from api.routes.api_tools import router as api_tools_router
from api.routes.mcp import router as mcp_router
from api.routes.tools import router as tools_router
from api.services.maintenance.orphan_cleanup_service import OrphanCleanupService
from api.scheduler.cleanup_scheduler import CleanupScheduler

from practorflow.llm import ModelPool
from practorflow.llm.knowledge.chroma_knowledge_store import ChromaKnowledgeStore
from practorflow.services.chat import ChatService
from practorflow.services.agent.agent_service import AgentService
from practorflow.settings.app_settings import appConfiguration
from practorflow.logger.logger import get_logger
from practorflow.session_store.factory import (
    create_session_history,
    create_session_store,
)
from practorflow.llm.tools.api.encryption import EncryptionService
from practorflow.tool_store.tinydb_api_store import TinyDBApiToolStore
from practorflow.llm.tools.api.factory import initialize_factory
from practorflow.tool_store.tinydb_mcp_server_store import TinyDBMCPServerStore
from practorflow.tool_store.tinydb_user_preferences import TinyDBUserToolPreferencesStore

logger = get_logger("agent-api", level="INFO")

# Module-level scheduler reference for cleanup on shutdown
_cleanup_scheduler = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Initializes services on startup and cleans up on shutdown.
    """
    global _cleanup_scheduler

    logger.info("[API] Starting application...")

    # Get configuration (already loaded at module level)
    api_config = get_api_configuration()

    # Log authentication mode
    if api_config.auth.is_open_mode:
        logger.info("[API] Authentication: Open mode (no credentials required)")
    elif api_config.auth.is_oidc_mode:
        logger.info(
            f"[API] Authentication: OIDC mode (issuer: {api_config.auth.oidc.issuer_url})"
        )
    else:
        logger.info("[API] Authentication: Local mode (app secret required)")

    # Initialize authentication service
    auth_service = AuthService(api_config.auth)
    container.auth_service = auth_service
    set_auth_service(auth_service)
    logger.info("[API] Authentication service initialized")

    # Initialize encryption service (uses JWT secret key)
    encryption_service = EncryptionService(api_config.auth.jwt.secret_key)
    container.encryption_service = encryption_service
    logger.info("[API] Encryption service initialized")

    # Initialize API tool store
    api_tools_db_path = os.getenv("API_TOOLS_DB_PATH")
    api_tool_store = TinyDBApiToolStore(db_path=api_tools_db_path)
    container.api_tool_store = api_tool_store
    logger.info(f"[API] API tool store initialized: {api_tools_db_path}")

    # Initialize API tool factory
    initialize_factory(api_tool_store)
    logger.info("[API] API tool factory initialized")

    # Initialize MCP server store
    mcp_servers_db_path = os.getenv("MCP_SERVERS_DB_PATH")
    mcp_server_store = TinyDBMCPServerStore(db_path=mcp_servers_db_path)
    container.mcp_server_store = mcp_server_store
    logger.info(f"[API] MCP server store initialized: {mcp_servers_db_path}")

    # Initialize user tool preferences store
    tool_prefs_db_path = os.getenv("USER_TOOL_DB_PATH")
    tool_prefs_store = TinyDBUserToolPreferencesStore(db_path=tool_prefs_db_path)
    container.tool_preferences_store = tool_prefs_store
    logger.info(f"[API] Tool preferences store initialized: {tool_prefs_db_path}")

    # Initialize configuration
    model_config = appConfiguration.ModelConfiguration
    knowledge_config = appConfiguration.KnowledgeChromaConfiguration

    # Initialize model pool
    logger.info(f"[API] Initializing model pool with model: {model_config.model_name}")
    model_pool = ModelPool.get_instance(max_models=1)

    # Preload model
    logger.info("[API] Preloading model...")
    await model_pool.preload(model_config)
    logger.info("[API] Model preloaded successfully")

    # Initialize knowledge store
    logger.info("[API] Initializing knowledge store...")
    knowledge_store = ChromaKnowledgeStore(knowledge_config)
    logger.info(
        f"[API] Knowledge store initialized: {knowledge_store.count_documents()} documents"
    )

    # Initialize session store
    session_store = create_session_store(llm_options_path)
    # Initialize session history
    session_history = create_session_history(llm_options_path)

    # Initialize chat service
    chat_service = ChatService(
        model_pool=model_pool,
        model_config=model_config,
        knowledge_store=knowledge_store,
        session_store=session_store,
    )

    # Initialize agent service
    agent_service = AgentService(
        model_pool=model_pool,
        model_config=model_config,
        knowledge_store=knowledge_store,
        session_store=session_store,
    )
    # Initialize delete session service
    delete_session_service = DeleteSessionService(
        knowledge_store=knowledge_store,
        session_store=session_store,
    )
    # Set services in container for dependency injection
    container.chat_service = chat_service
    container.agent_service = agent_service
    container.delete_session_service = delete_session_service
    # Set session history in container
    container.session_history = session_history

    # Initialize and start cleanup scheduler if enabled
    if api_config.cleanup.is_enabled:
        cleanup_service = OrphanCleanupService(
            knowledge_store=knowledge_store,
            session_history=session_history,
        )
        _cleanup_scheduler = CleanupScheduler(
            cleanup_service=cleanup_service,
            interval_minutes=api_config.cleanup.interval_minutes,
        )
        _cleanup_scheduler.start()
        logger.info(
            f"[API] Cleanup scheduler started (interval: {api_config.cleanup.interval_minutes} minutes)"
        )
    else:
        logger.info("[API] Cleanup scheduler disabled")

    logger.info("[API] Application started successfully")

    yield

    # Cleanup on shutdown
    logger.info("[API] Shutting down application...")

    # Stop cleanup scheduler
    if _cleanup_scheduler is not None:
        await _cleanup_scheduler.stop()
        logger.info("[API] Cleanup scheduler stopped")

    # Close stores
    api_tool_store.close()
    logger.info("[API] API tool store closed")

    mcp_server_store.close()
    logger.info("[API] MCP server store closed")

    tool_prefs_store.close()
    logger.info("[API] Tool preferences store closed")

    # Unload all models
    await model_pool.unload_all()

    logger.info("[API] Application shutdown complete")


# Create FastAPI application
app = FastAPI(
    title="PractorFlow API",
    description="Private AI Service API for local LLM inference with RAG support",
    version="0.0.1",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(agent_router)
app.include_router(session_router)
app.include_router(api_tools_router)
app.include_router(mcp_router)
app.include_router(tools_router)


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


def _run(reload: bool = False):
    import uvicorn

    # Set config path for module-level loading
    os.environ["_PRACTORFLOW_CONFIG_PATH"] = "./config"

    host = os.environ.get("PRACTORFLOW_API_HOST", "localhost")
    port = int(os.environ.get("PRACTORFLOW_API_PORT", "8000"))

    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=reload,
    )


def run():
    _run(reload=False)


def run_debug():
    _run(reload=True)