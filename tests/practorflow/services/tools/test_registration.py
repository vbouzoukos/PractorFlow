"""Tests for registration module."""

from unittest.mock import MagicMock, patch

import pytest

from practorflow.services.tools.registration import build_tools_for_registry


def test_build_tools_for_registry_with_user_preferences():
    """load_mcp_toolsets is called when user_preferences is not None."""
    registry = MagicMock()
    preferences = MagicMock()

    with patch("practorflow.services.tools.registration.build_builtin_tools", return_value=[]), \
         patch("practorflow.services.tools.registration.build_api_tools", return_value=[]):
        result = build_tools_for_registry(registry, user_preferences=preferences)

    registry.load_mcp_toolsets.assert_called_once_with(preferences)
    assert result == []


def test_build_tools_for_registry_without_user_preferences():
    """load_mcp_toolsets is NOT called when user_preferences is None."""
    registry = MagicMock()

    with patch("practorflow.services.tools.registration.build_builtin_tools", return_value=[]), \
         patch("practorflow.services.tools.registration.build_api_tools", return_value=[]):
        result = build_tools_for_registry(registry, user_preferences=None)

    registry.load_mcp_toolsets.assert_not_called()
    assert result == []
