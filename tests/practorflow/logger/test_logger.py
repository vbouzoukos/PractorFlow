import logging
import sys

import pytest

from practorflow.logger.logger import (
    ColorFormatter,
    _normalize_level,
    get_logger,
    _logger_cache,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def clear_logger_state():
    """
    Ensure logger cache and handlers are cleared between tests.
    This prevents global state leakage from logging module.
    """
    _logger_cache.clear()

    yield

    for logger in list(logging.Logger.manager.loggerDict.values()):
        if isinstance(logger, logging.Logger):
            logger.handlers.clear()


# ============================================================================
# Tests for _normalize_level
# ============================================================================


def test_normalize_level_valid_strings():
    assert _normalize_level("DEBUG") == logging.DEBUG
    assert _normalize_level("info") == logging.INFO
    assert _normalize_level("Warning") == logging.WARNING
    assert _normalize_level("ERROR") == logging.ERROR
    assert _normalize_level("CRITICAL") == logging.CRITICAL


def test_normalize_level_invalid_string_defaults_to_info():
    assert _normalize_level("NOT_A_LEVEL") == logging.INFO


def test_normalize_level_non_string():
    assert _normalize_level(None) == logging.NOTSET
    assert _normalize_level(123) == logging.NOTSET
    assert _normalize_level([]) == logging.NOTSET


# ============================================================================
# Tests for ColorFormatter
# ============================================================================


def test_color_formatter_adds_color_for_known_level():
    formatter = ColorFormatter("%(levelname)s - %(message)s")

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="hello",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)

    assert "\033[" in formatted  # ANSI color code
    assert "INFO - hello" in formatted
    assert formatted.endswith("\033[0m")


def test_color_formatter_no_color_for_unknown_level():
    formatter = ColorFormatter("%(message)s")

    record = logging.LogRecord(
        name="test",
        level=999,
        pathname=__file__,
        lineno=10,
        msg="plain",
        args=(),
        exc_info=None,
    )

    formatted = formatter.format(record)

    assert formatted == "plain"


# ============================================================================
# Tests for get_logger
# ============================================================================


def test_get_logger_returns_cached_instance():
    logger1 = get_logger("cached_logger")
    logger2 = get_logger("cached_logger")

    assert logger1 is logger2


def test_get_logger_sets_level_and_propagation():
    logger = get_logger("level_test", level="DEBUG")

    assert logger.level == logging.DEBUG
    assert logger.propagate is False


def test_get_logger_stdout_handler_added_once():
    logger = get_logger("stdout_test", level="INFO", stdout=True)

    handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
    ]

    assert len(handlers) == 1

    # Call again — should not duplicate handler
    logger = get_logger("stdout_test", level="INFO", stdout=True)

    handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
    ]

    assert len(handlers) == 1


def test_get_logger_stdout_disabled():
    logger = get_logger("no_stdout", stdout=False)

    handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.StreamHandler)
    ]

    assert handlers == []


def test_get_logger_file_handler_created(tmp_path):
    log_file = tmp_path / "test.log"

    logger = get_logger(
        "file_logger",
        log_file=str(log_file),
        level="INFO",
    )

    handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]

    assert len(handlers) == 1
    assert handlers[0].baseFilename == str(log_file)


def test_get_logger_file_handler_not_duplicated(tmp_path):
    log_file = tmp_path / "dup.log"

    logger = get_logger("dup_logger", log_file=str(log_file))
    logger = get_logger("dup_logger", log_file=str(log_file))

    handlers = [
        h for h in logger.handlers
        if isinstance(h, logging.handlers.RotatingFileHandler)
    ]

    assert len(handlers) == 1
