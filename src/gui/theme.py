"""
Theme management for PractorFlow GUI.

Applies light/dark/auto themes using qdarktheme.
"""

import qdarktheme

from PySide6.QtWidgets import QApplication

from gui.logger import get_logger

logger = get_logger("practorflow-client", level="INFO", log_file="logs/practorflow-client.log")


def apply_theme(app: QApplication, theme: str):
    """
    Apply theme to application.
    
    Args:
        app: QApplication instance.
        theme: Theme name - "auto", "dark", or "light".
    """
    try:
        qdarktheme.setup_theme(theme)
        logger.info(f"Theme applied: {theme}")
    except Exception as e:
        logger.error(f"Failed to apply theme: {e}")