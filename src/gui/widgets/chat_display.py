"""
Chat display - Scrollable message history.

Displays the conversation history with automatic scrolling
and support for streaming message updates.
Supports message editing and truncation.
"""

from PySide6.QtWidgets import (
    QScrollArea,
    QWidget,
    QVBoxLayout,
    QSizePolicy,
    QApplication,
)
from PySide6.QtCore import Qt, Signal, Slot

from gui.widgets.message_widget import MessageWidget


class ChatDisplay(QScrollArea):
    """
    Scrollable chat message display.
    
    Manages a list of MessageWidget instances and provides
    methods for adding messages and updating streaming content.
    Supports message editing and truncation.
    
    Signals:
        message_edit_requested: Emitted when user edits a message (index, new_content).
    """
    
    message_edit_requested = Signal(int, str)  # index, new_content
    
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
        try:
            widget = MessageWidget(role, content)
            index = len(self._messages)
            widget.set_index(index)
            widget.edit_requested.connect(lambda i, c: self.message_edit_requested.emit(i, c))
            
            self._messages.append(widget)
            
            # Insert before the stretch
            count = self._layout.count()
            self._layout.insertWidget(count - 1, widget)
            
            # Scroll to bottom
            self._scroll_to_bottom()
        except Exception:
            pass  # pragma: no cover
    
    def truncate_from_index(self, from_index: int) -> int:
        """
        Remove all messages from the given index onwards.
        
        Args:
            from_index: Index from which to truncate (inclusive).
        
        Returns:
            Number of messages removed.
        """
        try:
            if from_index < 0 or from_index >= len(self._messages):
                return 0
            
            removed_count = len(self._messages) - from_index
            
            # Remove widgets from layout and delete
            for i in range(len(self._messages) - 1, from_index - 1, -1):
                widget = self._messages[i]
                self._layout.removeWidget(widget)
                widget.deleteLater()
            
            # Truncate the list
            self._messages = self._messages[:from_index]
            
            return removed_count
        except Exception:
            return 0
    
    def append_to_last_message(self, text: str):
        """
        Append text to the last message (for streaming).
        
        Args:
            text: Text chunk to append.
        """
        try:
            if not self._messages:
                return
            
            last_message = self._messages[-1]
            last_message.append_content(text)
            
            # Scroll to bottom
            self._scroll_to_bottom()
        except Exception:
            pass  # pragma: no cover
    
    def finalize_last_message(self):
        """
        Finalize the last message after streaming completes.
        
        Triggers Markdown rendering for assistant messages.
        """
        try:
            if not self._messages:
                return
            
            last_message = self._messages[-1]
            last_message.finalize()
        except Exception:
            pass  # pragma: no cover
    
    def clear_messages(self):
        """Clear all messages from the display."""
        try:
            for widget in self._messages:
                self._layout.removeWidget(widget)
                widget.deleteLater()
            
            self._messages.clear()
        except Exception:
            self._messages = []
    
    def get_message_count(self) -> int:
        """Get the number of messages."""
        return len(self._messages)
    
    def get_message_at(self, index: int) -> MessageWidget:
        """
        Get the message widget at the given index.
        
        Args:
            index: Index of the message.
        
        Returns:
            MessageWidget or None if index is out of bounds.
        """
        if 0 <= index < len(self._messages):
            return self._messages[index]
        return None
    
    def _scroll_to_bottom(self):
        """Scroll to the bottom of the display."""
        try:
            # Process events to ensure layout is updated
            QApplication.processEvents()
            
            scrollbar = self.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())
        except Exception:
            pass  # pragma: no cover