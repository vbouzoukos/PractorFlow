"""
Chat display - Scrollable message history.

Displays the conversation history with automatic scrolling
and support for streaming message updates.
Supports message editing and truncation.
"""

from PySide6.QtWidgets import (
    QAbstractScrollArea,
    QWidget,
    QVBoxLayout,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, Slot, QTimer

from gui.widgets.message_widget import MessageWidget


class ChatDisplay(QAbstractScrollArea):
    """
    Scrollable chat message display.
    
    Manages a list of MessageWidget instances and provides
    methods for adding messages and updating streaming content.
    Supports message editing and truncation.
    
    Signals:
        message_edit_requested: Emitted when user edits a message (index, new_content).
        message_added: Emitted when a new message is added.
    """
    
    message_edit_requested = Signal(int, str)  # index, new_content
    message_added = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._messages = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the user interface."""
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        # Container widget as child of viewport
        self._container = QWidget(self.viewport())
        
        # Layout for messages
        self._layout = QVBoxLayout(self._container)
        self._layout.setContentsMargins(4, 4, 4, 4)
        self._layout.setSpacing(8)
        self._layout.setAlignment(Qt.AlignTop)
        
        # Add stretch at bottom to push messages to top
        self._layout.addStretch()
        
        # Connect scrollbar
        self.verticalScrollBar().valueChanged.connect(self._on_scroll)
        
        # Connect message_added to scroll
        self.message_added.connect(self._scroll_to_bottom)

        # set up scroll
        vbar = self.verticalScrollBar()
        vbar.setSingleStep(12)      # wheel step (pixels)
        vbar.setPageStep(200)       # scrollbar click / PageUp-Down

    def _on_scroll(self, value: int):
        """Handle scrollbar value change."""
        self._container.move(0, -value)
    
    def resizeEvent(self, event):
        """Handle resize events."""
        super().resizeEvent(event)
        self._update_container_geometry()
    
    def _update_container_geometry(self):
        """Update container size and scrollbar range."""
        viewport_width = self.viewport().width()
        viewport_height = self.viewport().height()
        
        # Set container width to viewport width
        self._container.setFixedWidth(viewport_width)
        
        # Let container calculate its preferred height
        content_height = self._container.sizeHint().height()
        self._container.setMinimumHeight(max(content_height, viewport_height))
        
        # Update scrollbar
        vbar = self.verticalScrollBar()
        max_scroll = max(0, content_height - viewport_height)
        vbar.setRange(0, max_scroll)
        vbar.setPageStep(viewport_height)
        
        # Reposition container
        self._container.move(0, -vbar.value())
    
    def focusNextPrevChild(self, next: bool) -> bool:
        """Prevent auto-scroll on focus change."""
        return False
    
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
            
            # Defer geometry update to allow layout to settle
            QTimer.singleShot(0, self._update_container_geometry)
            
            # Emit signal after message added
            self.message_added.emit()
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
            
            # Update geometry
            QTimer.singleShot(0, self._update_container_geometry)
            
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
            
            # Update geometry and scroll
            self._update_container_geometry()
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
            
            # Update geometry after content change
            QTimer.singleShot(0, self._update_container_geometry)
        except Exception:
            pass  # pragma: no cover
    
    def clear_messages(self):
        """Clear all messages from the display."""
        try:
            for widget in self._messages:
                self._layout.removeWidget(widget)
                widget.deleteLater()
            
            self._messages.clear()
            
            # Update geometry
            QTimer.singleShot(0, self._update_container_geometry)
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
    
    @Slot()
    def _scroll_to_bottom(self):
        """Scroll to the bottom of the display."""
        try:
            vbar = self.verticalScrollBar()
            vbar.setValue(vbar.maximum())
        except Exception:
            pass  # pragma: no cover