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
        if hasattr(qdarktheme, "setup_theme"):
            qdarktheme.setup_theme(theme)
        else:
            if theme == "auto":
                from PySide6.QtCore import Qt
                scheme = app.styleHints().colorScheme()
                theme = "dark" if scheme == Qt.ColorScheme.Dark else "light"
            app.setStyleSheet(qdarktheme.load_stylesheet(theme))
        logger.info(f"Theme applied: {theme}")
    except Exception as e:
        logger.error(f"Failed to apply theme: {e}")