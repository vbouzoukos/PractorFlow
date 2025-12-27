"""
Chat display - Scrollable message history.

Displays the conversation history with automatic scrolling
and support for streaming message updates.
"""

from PySide6.QtWidgets import (
    QScrollArea,
    QWidget,
    QVBoxLayout,
    QSizePolicy,
)
from PySide6.QtCore import Qt

from widgets.message_widget import MessageWidget


class ChatDisplay(QScrollArea):
    """
    Scrollable chat message display.
    
    Manages a list of MessageWidget instances and provides
    methods for adding messages and updating streaming content.
    """
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._messages = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the user interface."""
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Container widget
        self._container = QWidget()
        self.setWidget(self._container)
        
        # Layout for messages
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(8)
        self._layout.setAlignment(Qt.AlignTop)
        
        # Add stretch at bottom to push messages to top
        self._layout.addStretch()
        
        # No hardcoded styles - use system palette
    
    def add_user_message(self, content: str):
        """
        Add a user message to the display.
        
        Args:
            content: Message text.
        """
        self._add_message(MessageWidget.ROLE_USER, content)
    
    def add_assistant_message(self, content: str = ""):
        """
        Add an assistant message to the display.
        
        Args:
            content: Initial message text (can be empty for streaming).
        """
        self._add_message(MessageWidget.ROLE_ASSISTANT, content)
    
    def add_system_message(self, content: str):
        """
        Add a system message to the display.
        
        Args:
            content: Message text.
        """
        self._add_message(MessageWidget.ROLE_SYSTEM, content)
    
    def _add_message(self, role: str, content: str):
        """
        Add a message widget to the display.
        
        Args:
            role: Message role (user, assistant, system).
            content: Message text.
        """
        widget = MessageWidget(role, content)
        self._messages.append(widget)
        
        # Insert before the stretch
        count = self._layout.count()
        self._layout.insertWidget(count - 1, widget)
        
        # Scroll to bottom
        self._scroll_to_bottom()
    
    def append_to_last_message(self, text: str):
        """
        Append text to the last message (for streaming).
        
        Args:
            text: Text chunk to append.
        """
        if not self._messages:
            return
        
        last_message = self._messages[-1]
        last_message.append_content(text)
        
        # Scroll to bottom
        self._scroll_to_bottom()
    
    def finalize_last_message(self):
        """
        Finalize the last message after streaming completes.
        
        Triggers Markdown rendering for assistant messages.
        """
        if not self._messages:
            return
        
        last_message = self._messages[-1]
        last_message.finalize()
    
    def clear_messages(self):
        """Clear all messages from the display."""
        for widget in self._messages:
            self._layout.removeWidget(widget)
            widget.deleteLater()
        
        self._messages.clear()
    
    def get_message_count(self) -> int:
        """Get the number of messages."""
        return len(self._messages)
    
    def _scroll_to_bottom(self):
        """Scroll to the bottom of the display."""
        # Process events to ensure layout is updated
        from PySide6.QtWidgets import QApplication
        QApplication.processEvents()
        
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())