"""
PractorFlow GUI Client - Application Entry Point.

Usage:
    python main.py
"""

import sys
from importlib.metadata import version

from PySide6.QtWidgets import QApplication, QMessageBox

from gui.logger import get_logger

logger = get_logger("practorflow-client", level="INFO", log_file="logs/practorflow-client.log")

from gui.theme import apply_theme
from gui.settings.settings import load_settings


def excepthook(exc_type, exc_value, exc_tb):
    """Global exception handler to prevent silent crashes."""
    import traceback
    
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    tb_text = ''.join(tb_lines)
    
    logger.critical(f"Unhandled exception:\n{tb_text}")
    
    try:
        app = QApplication.instance()
        if app:
            QMessageBox.critical(
                None,
                "Unexpected Error",
                f"An unexpected error occurred:\n\n{exc_value}\n\nThe application may be unstable."
            )
    except Exception:
        pass


def main():
    """Application entry point."""
    sys.excepthook = excepthook
    
    try:
        logger.info("Starting PractorFlow GUI")
        
        app = QApplication(sys.argv)
        app.setApplicationName("PractorFlow")
        app.setApplicationVersion(version("practorflow-gui"))
        app.setStyle("Fusion")
        
        # Apply theme from settings
        settings = load_settings()
        apply_theme(app, settings.theme)
        
        from gui.main_window import MainWindow
        
        window = MainWindow()
        window.show()
        
        logger.info("Application started")
        
        sys.exit(app.exec())
        
    except Exception as e:
        logger.critical(f"Failed to start application: {e}")
        
        try:
            app = QApplication.instance()
            if app is None:
                app = QApplication(sys.argv)
            
            QMessageBox.critical(
                None,
                "Startup Error",
                f"Failed to start application:\n\n{e}"
            )
        except Exception:
            pass
        
        sys.exit(1)


if __name__ == "__main__":
    main()