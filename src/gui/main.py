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
from PySide6.QtWidgets import QApplication

from chat_window import ChatWindow


def main():
    """Application entry point."""
    # Load environment variables
    load_dotenv()
    
    # Get API URL from environment
    api_url = os.getenv("PRACTORFLOW_API_URL", "http://localhost:8000")
    
    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName("PractorFlow Chat")
    app.setApplicationVersion("0.0.1")
    app.setStyle("Fusion")
    
    # Create and show main window
    window = ChatWindow(api_url=api_url)
    window.show()
    
    # Run event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()