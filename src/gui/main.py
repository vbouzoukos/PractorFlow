"""
PractorFlow GUI Chat Client - Application Entry Point.

Usage:
    python main.py

Environment Variables:
    PRACTORFLOW_API_URL - API base URL (default: http://localhost:8000)
"""

import sys
import os

# Enable dark mode (must be before QApplication)
sys.argv += ['-platform', 'windows:darkmode=2']

from dotenv import load_dotenv
from PySide6.QtWidgets import QApplication, QMessageBox


def excepthook(exc_type, exc_value, exc_tb):
    """Global exception handler to prevent silent crashes."""
    import traceback
    
    # Format the exception
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_tb)
    tb_text = ''.join(tb_lines)
    
    # Print to stderr
    print(f"Unhandled exception:\n{tb_text}", file=sys.stderr)
    
    # Show error dialog if QApplication exists
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
    # Install global exception handler
    sys.excepthook = excepthook
    
    try:
        # Load environment variables
        load_dotenv()
        
        # Get API URL from environment
        api_url = os.getenv("PRACTORFLOW_API_URL", "http://localhost:8000")
        
        # Create application
        app = QApplication(sys.argv)
        app.setApplicationName("PractorFlow Chat")
        app.setApplicationVersion("0.0.1")
        app.setStyle("Fusion")
        
        # Import here to ensure QApplication exists first
        from chat_window import ChatWindow
        
        # Create and show main window
        window = ChatWindow(api_url=api_url)
        window.show()
        
        # Run event loop
        sys.exit(app.exec())
        
    except Exception as e:
        print(f"Failed to start application: {e}", file=sys.stderr)
        
        # Try to show error dialog
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