"""
Main window container for PractorFlow GUI.

Provides menu navigation between different views (Chat, Settings).
"""

from PySide6.QtWidgets import (
    QMainWindow,
    QStackedWidget,
    QStatusBar,
    QMessageBox,
    QApplication,
)
from PySide6.QtCore import Slot
from PySide6.QtGui import QAction

from gui.logger import get_logger

logger = get_logger("practorflow-client", level="INFO", log_file="logs/practorflow-client.log")

from gui.settings.settings import settings_exist
from gui.settings.settings_window import SettingsWindow
from gui.chat.chat_window import ChatWindow


class MainWindow(QMainWindow):
    """
    Main application window with menu navigation.
    
    Contains a stacked widget to switch between Chat and Settings views.
    Opens Settings view if no settings.json exists, otherwise Chat view.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        try:
            self._setup_ui()
            self._setup_menu()
            self._connect_signals()
            
            # Show appropriate initial view
            if settings_exist():
                self._show_chat()
                logger.info("MainWindow initialized with Chat view")
            else:
                self._show_settings()
                logger.info("MainWindow initialized with Settings view (no settings.json)")
        except Exception as e:
            logger.error(f"Failed to initialize MainWindow: {e}")
            raise
    
    def _setup_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("PractorFlow")
        self.setMinimumSize(800, 600)
        self.resize(1100, 750)
        
        # Central stacked widget for view switching
        self._stack = QStackedWidget()
        self.setCentralWidget(self._stack)
        
        # Create views
        self._settings_window = SettingsWindow()
        self._chat_window = ChatWindow()
        
        # Add views to stack
        self._stack.addWidget(self._chat_window)      # index 0
        self._stack.addWidget(self._settings_window)  # index 1
        
        # Status bar
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
    
    def _setup_menu(self):
        """Initialize the menu bar."""
        menu_bar = self.menuBar()
        
        # View menu
        view_menu = menu_bar.addMenu("&View")
        
        self._chat_action = QAction("&Chat", self)
        self._chat_action.setShortcut("Ctrl+1")
        self._chat_action.setStatusTip("Open chat view")
        view_menu.addAction(self._chat_action)
        
        self._settings_action = QAction("&Settings", self)
        self._settings_action.setShortcut("Ctrl+,")
        self._settings_action.setStatusTip("Open settings")
        view_menu.addAction(self._settings_action)
        
        view_menu.addSeparator()
        
        self._exit_action = QAction("E&xit", self)
        self._exit_action.setShortcut("Ctrl+Q")
        self._exit_action.setStatusTip("Exit application")
        view_menu.addAction(self._exit_action)
        
        # Help menu
        help_menu = menu_bar.addMenu("&Help")
        
        self._about_action = QAction("&About", self)
        self._about_action.setStatusTip("About PractorFlow")
        help_menu.addAction(self._about_action)
    
    def _connect_signals(self):
        """Connect signals to slots."""
        self._chat_action.triggered.connect(self._show_chat)
        self._settings_action.triggered.connect(self._show_settings)
        self._exit_action.triggered.connect(self.close)
        self._about_action.triggered.connect(self._show_about)
    
    @Slot()
    def _show_chat(self):
        """Switch to chat view."""
        try:
            self._stack.setCurrentIndex(0)
            self._status_bar.showMessage("Chat", 2000)
            logger.debug("Switched to Chat view")
        except Exception as e:
            logger.error(f"Error switching to Chat view: {e}")
    
    @Slot()
    def _show_settings(self):
        """Switch to settings view."""
        try:
            self._stack.setCurrentIndex(1)
            self._status_bar.showMessage("Settings", 2000)
            logger.debug("Switched to Settings view")
        except Exception as e:
            logger.error(f"Error switching to Settings view: {e}")
    
    @Slot()
    def _show_about(self):
        """Show about dialog."""
        try:
            app = QApplication.instance()
            version = app.applicationVersion() if app else "unknown"
            
            QMessageBox.about(
                self,
                "About PractorFlow",
                f"<h3>PractorFlow</h3>"
                f"<p>Version: {version}</p>"
                f"<p>A desktop client for PractorFlow AI services.</p>"
            )
        except Exception as e:
            logger.error(f"Error showing about dialog: {e}")
    
    def closeEvent(self, event):
        """Handle window close - cleanup."""
        try:
            logger.info("MainWindow closing")
            self._chat_window.shutdown()
            logger.info("MainWindow closed")
        except Exception as e:
            logger.error(f"Error during close: {e}")
        event.accept()