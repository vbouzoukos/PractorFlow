"""
Session store factory.

Creates session store and session history instances based on configuration.

Environment Variables:
    STORE_SESSION: Storage type - "MEMORY" or "LOCAL" (case insensitive)
    STORE_SESSION_DB_PATH: Path to TinyDB file (default: "./sessions.json")
"""

import os

from dotenv import load_dotenv

from practorflow.session_store.memory_session_store import InMemorySessionStore
from practorflow.session_store.persist_session_store import TinyDBSessionStore
from practorflow.session_store.memory_session_history import InMemorySessionHistory
from practorflow.session_store.persist_session_history import PersistSessionHistory
from practorflow.session_store.session_history import SessionHistory
from practorflow.llm.base.session_store import SessionStore


# Supported store types
STORE_TYPE_MEMORY = "memory"
STORE_TYPE_LOCAL = "local"

# Default configuration
DEFAULT_STORE_TYPE = STORE_TYPE_MEMORY
DEFAULT_DB_PATH = "./sessions.json"

def create_session_store(config_path: str = "../config/llm/options") -> SessionStore:
    """
    Create a session store based on configuration.
    
    Returns:
        SessionStore instance.
    
    Raises:
        ValueError: If store_type is not supported.
    """
    session_env = os.path.join(config_path, "session.env")
    load_dotenv(dotenv_path=session_env, override=True)

    store_type = os.getenv("STORE_SESSION", DEFAULT_STORE_TYPE)
    
    # Normalize to lowercase for case-insensitive comparison
    store_type_normalized = store_type.strip().lower()
    
    if store_type_normalized == STORE_TYPE_MEMORY:
        return InMemorySessionStore()
    
    if store_type_normalized == STORE_TYPE_LOCAL:
        path = os.getenv("STORE_SESSION_DB_PATH", DEFAULT_DB_PATH)
        return TinyDBSessionStore(db_path=path)
    
    supported_types = [STORE_TYPE_MEMORY, STORE_TYPE_LOCAL]
    raise ValueError(
        f"Unsupported session store type: '{store_type}'. "
        f"Supported types: {', '.join(supported_types)}"
    )


def create_session_history(config_path: str = "../config/llm/options") -> SessionHistory:
    """
    Create a session history based on configuration.
    
    Returns:
        SessionHistory instance.
    
    Raises:
        ValueError: If store_type is not supported.
    """
    session_env = os.path.join(config_path, "session.env")
    load_dotenv(dotenv_path=session_env, override=True)
    
    store_type = os.getenv("STORE_SESSION", DEFAULT_STORE_TYPE)
    # Normalize to lowercase for case-insensitive comparison
    store_type_normalized = store_type.strip().lower()
    
    if store_type_normalized == STORE_TYPE_MEMORY:
        return InMemorySessionHistory()
    
    if store_type_normalized == STORE_TYPE_LOCAL:
        path = os.getenv("STORE_SESSION_DB_PATH", DEFAULT_DB_PATH)
        return PersistSessionHistory(db_path=path)
    
    supported_types = [STORE_TYPE_MEMORY, STORE_TYPE_LOCAL]
    raise ValueError(
        f"Unsupported session history type: '{store_type}'. "
        f"Supported types: {', '.join(supported_types)}"
    )