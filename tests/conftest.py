"""
Root conftest.py - Shared fixtures for all tests.

This file contains fixtures used across both practorflow and api test suites.
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Dict, Any, Generator
from unittest.mock import MagicMock, patch

import pytest


# Ensure src is in path for imports
@pytest.fixture(scope="session", autouse=True)
def setup_python_path():
    """Add src directory to Python path for test imports."""
    src_path = Path(__file__).parent.parent / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Provide a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_file(temp_dir: Path) -> Generator[Path, None, None]:
    """Provide a temporary file path within temp_dir."""
    file_path = temp_dir / "test_file.txt"
    file_path.write_text("test content")
    yield file_path


@pytest.fixture
def isolated_env() -> Generator[Dict[str, str], None, None]:
    """
    Isolate environment variables for the test.
    
    Saves current environment, allows modifications during test,
    and restores original environment after test completes.
    """
    original_env = os.environ.copy()
    try:
        yield os.environ
    finally:
        os.environ.clear()
        os.environ.update(original_env)


@pytest.fixture
def clean_env(isolated_env: Dict[str, str]) -> Dict[str, str]:
    """
    Provide a clean environment with test-specific variables removed.
    
    Removes common configuration variables that might interfere with tests.
    """
    keys_to_remove = [
        "APP_SECRET",
        "JWT_SECRET_KEY",
        "JWT_ALGORITHM",
        "JWT_TOKEN_EXPIRY_MINUTES",
        "OIDC_ISSUER_URL",
        "OIDC_AUDIENCE",
        "OIDC_CLIENT_ID",
        "OIDC_CLIENT_SECRET",
    ]
    for key in keys_to_remove:
        isolated_env.pop(key, None)
    return isolated_env


@pytest.fixture
def mock_logger() -> MagicMock:
    """Provide a mock logger for tests that need logging without side effects."""
    logger = MagicMock()
    logger.debug = MagicMock()
    logger.info = MagicMock()
    logger.warning = MagicMock()
    logger.error = MagicMock()
    logger.critical = MagicMock()
    return logger


@pytest.fixture
def sample_text_content() -> str:
    """Provide sample text content for document processing tests."""
    return """
    This is a sample document for testing purposes.
    It contains multiple paragraphs and sentences.
    
    The document loader should be able to parse this content
    and create appropriate chunks for retrieval and context.
    
    Testing is important for maintaining code quality
    and ensuring that changes don't break existing functionality.
    """


@pytest.fixture
def sample_metadata() -> Dict[str, Any]:
    """Provide sample metadata for document tests."""
    return {
        "filename": "test_document.txt",
        "source": "test",
        "doc_id": "test-doc-001",
        "chunk_index": 0,
    }


@pytest.fixture
def sample_chunks() -> list:
    """Provide sample text chunks for knowledge store tests."""
    return [
        "First chunk of text for testing retrieval.",
        "Second chunk contains different information.",
        "Third chunk has unique content for search.",
    ]