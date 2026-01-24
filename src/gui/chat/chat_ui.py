"""
Chat UI setup and widget creation.

Handles all UI initialization for the chat window.
"""

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QSplitter,
    QCheckBox,
)
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QIcon

from gui.logger import get_logger

logger = get_logger("practorflow-client", level="INFO", log_file="logs/practorflow-client.log")

from gui.widgets.chat_display import ChatDisplay
from gui.widgets.input_widget import InputWidget
from gui.widgets.history_panel import HistoryPanel
from gui.widgets.documents_panel import DocumentsPanel
from gui.api.session_client import SessionClient
import gui.widgets.ui_state as ui_state


class ChatUI:
    """
    Encapsulates chat window UI setup.
    
    Creates and configures all widgets, returning references
    for the ChatWindow to connect signals.
    """
    
    def __init__(self, parent: QWidget, session_client: SessionClient):
        """
        Initialize chat UI.
        
        Args:
            parent: Parent widget to build UI on.
            session_client: Session client for panels.
        """
        self._parent = parent
        self._session_client = session_client
        
        # Widget references (populated by setup)
        self.history_panel: HistoryPanel = None
        self.documents_panel: DocumentsPanel = None
        self.chat_display: ChatDisplay = None
        self.input_widget: InputWidget = None
        self.session_label: QLabel = None
        self.connect_btn: QPushButton = None
        self.reconnect_btn: QPushButton = None
        self.new_session_btn: QPushButton = None
        self.documents_btn: QPushButton = None
        self.agent_mode_checkbox: QCheckBox = None
        
        self._setup()
        logger.debug("ChatUI initialized")
    
    def _setup(self):
        """Initialize the user interface."""
        try:
            # Main horizontal layout
            main_layout = QHBoxLayout(self._parent)
            main_layout.setContentsMargins(4, 4, 4, 4)
            main_layout.setSpacing(4)
            
            # History panel (collapsible)
            self.history_panel = HistoryPanel(self._session_client)
            main_layout.addWidget(self.history_panel)
            
            # Chat area container
            chat_container = QWidget()
            chat_layout = QVBoxLayout(chat_container)
            chat_layout.setContentsMargins(4, 4, 4, 4)
            chat_layout.setSpacing(8)
            
            # Header with session controls
            self._setup_header(chat_layout)
            
            # Documents panel (foldable, below header)
            self.documents_panel = DocumentsPanel(self._session_client)
            chat_layout.addWidget(self.documents_panel)
            
            # Chat display and input splitter
            self._setup_chat_area(chat_layout)
            
            main_layout.addWidget(chat_container, stretch=1)
            
            # Disable input until connected
            self.input_widget.set_enabled(False)
            self.history_panel.set_enabled(False)           
            self.documents_btn.setEnabled(False) 
            logger.debug("ChatUI setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup ChatUI: {e}")
            raise
    
    def _setup_header(self, parent_layout: QVBoxLayout):
        """Setup header with session controls."""
        header_layout = QHBoxLayout()
        
        self.session_label = QLabel("Session: Not connected")
        header_layout.addWidget(self.session_label)
        
        # Connect button (visible when not connected)
        self.connect_btn = QPushButton("Connect")
        self.connect_btn.setToolTip("Connect to server")
        header_layout.addWidget(self.connect_btn)
        
        # Reconnect button (hidden by default)
        self.reconnect_btn = QPushButton("Reconnect")
        self.reconnect_btn.setToolTip("Reconnect to server")
        self.reconnect_btn.hide()
        header_layout.addWidget(self.reconnect_btn)
        
        # Agent mode toggle
        self.agent_mode_checkbox = QCheckBox("Agent Mode")
        self.agent_mode_checkbox.setToolTip(
            "Enable multi-agent task execution (plan → execute → verify)"
        )
        header_layout.addWidget(self.agent_mode_checkbox)
        
        # Documents button (icon)
        self.documents_btn = QPushButton()
        self.documents_btn.setFixedSize(28, 28)
        self.documents_btn.setToolTip("Show session documents")
        self.documents_btn.setCursor(Qt.PointingHandCursor)
        
        doc_icon = QIcon.fromTheme("folder-documents")
        if doc_icon.isNull():
            doc_icon = QIcon.fromTheme("document-multiple")
        if doc_icon.isNull():
            self.documents_btn.setText("📄")
        else:
            self.documents_btn.setIcon(doc_icon)
            self.documents_btn.setIconSize(QSize(18, 18))
        
        header_layout.addWidget(self.documents_btn)
        
        header_layout.addStretch()
        
        self.new_session_btn = QPushButton("New Session")
        self.new_session_btn.setToolTip("Start a new chat session")
        header_layout.addWidget(self.new_session_btn)
        
        parent_layout.addLayout(header_layout)
    
    def _setup_chat_area(self, parent_layout: QVBoxLayout):
        """Setup chat display and input with splitter."""
        splitter = QSplitter(Qt.Vertical)
        
        self.chat_display = ChatDisplay()
        self.input_widget = InputWidget()
        
        splitter.addWidget(self.chat_display)
        splitter.addWidget(self.input_widget)
        splitter.setSizes([500, 100])
        
        parent_layout.addWidget(splitter)
    
    def set_connected_state(self, connected: bool):
        """Update UI for connected/disconnected state."""
        self.connect_btn.setVisible(not connected)
        self.input_widget.set_enabled(connected)
        self.history_panel.set_enabled(connected)           
        self.documents_btn.setEnabled(connected) 

    def set_busy_state(self, busy: bool):
        """Update UI for busy/idle state."""
        ui_state.generating = busy
        self.input_widget.set_enabled(not busy)
    
    def shutdown(self):
        """Shutdown UI panels."""
        try:
            self.history_panel.shutdown()
            self.documents_panel.shutdown()
            logger.debug("ChatUI panels shutdown")
        except Exception as e:
            logger.error(f"Error shutting down ChatUI: {e}")