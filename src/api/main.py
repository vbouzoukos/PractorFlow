"""
FastAPI application entry point.

Sets up the FastAPI application with:
- Lifespan management for service initialization/cleanup
- Authentication configuration and service
- Chat routes
- CORS middleware
"""

from contextlib import asynccontextmanager
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import container
from api.config import load_api_configuration, get_api_configuration
from api.auth import AuthService, set_auth_service
from api.routes.auth import router as auth_router
from api.routes.chat import router as chat_router
from practorflow.llm import ModelPool
from practorflow.llm.knowledge.chroma_knowledge_store import ChromaKnowledgeStore
from practorflow.services.chat import ChatService
from practorflow.settings.app_settings import appConfiguration
from practorflow.logger.logger import get_logger
from session_store.factory import create_session_history, create_session_store

logger = get_logger("agent-api", level="INFO")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Initializes services on startup and cleans up on shutdown.
    """
    logger.info("[API] Starting application...")

    # Load API configuration (auth settings)
    logger.info("[API] Loading API configuration...")
    load_api_configuration("../config/api")
    api_config = get_api_configuration()
    
    # Log authentication mode
    if api_config.auth.is_open_mode:
        logger.info("[API] Authentication: Open mode (no credentials required)")
    elif api_config.auth.is_oidc_mode:
        logger.info(f"[API] Authentication: OIDC mode (issuer: {api_config.auth.oidc.issuer_url})")
    else:
        logger.info("[API] Authentication: Local mode (app secret required)")

    # Initialize authentication service
    auth_service = AuthService(api_config.auth)
    container.auth_service = auth_service
    set_auth_service(auth_service)
    logger.info("[API] Authentication service initialized")

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
    session_store = create_session_store()
    # Initialize session history
    session_history = create_session_history()

    # Initialize chat service
    chat_service = ChatService(
        model_pool=model_pool,
        model_config=model_config,
        knowledge_store=knowledge_store,
        session_store=session_store,
    )

    # Set service in container for dependency injection
    container.chat_service = chat_service
    # Set session history in container
    container.session_history = session_history
    logger.info("[API] Application started successfully")

    yield

    # Cleanup on shutdown
    logger.info("[API] Shutting down application...")

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


@app.get("/health", tags=["health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}