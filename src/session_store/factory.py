"""
Session store factory.

Creates session store instances based on configuration.

Environment Variables:
    STORE_SESSION: Storage type - "MEMORY" or "LOCAL" (case insensitive)
"""

import os

from session_store.memory_session_store import InMemorySessionStore
from session_store.persist_session_store import TinyDBSessionStore
from practorflow.llm.base.session_store import SessionStore


# Supported store types
STORE_TYPE_MEMORY = "memory"
STORE_TYPE_LOCAL = "local"

# Default configuration
DEFAULT_STORE_TYPE = STORE_TYPE_MEMORY
DEFAULT_DB_PATH = "./sessions.json"


def create_session_store() -> SessionStore:
    """
    Create a session store based on configuration.
    
    Returns:
        SessionStore instance.
    
    Raises:
        ValueError: If store_type is not supported.
    """
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