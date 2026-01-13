"""
Message widget - Individual chat message display.

Displays a single message with role indicator and content.
Supports Markdown rendering for assistant messages.
Uses system palette for automatic dark/light mode support.
Supports edit functionality for user messages.
"""

from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QTextEdit,
    QPushButton,
    QSizePolicy,
    QApplication,
)
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QFont, QPalette, QColor


class MessageWidget(QFrame):
    """
    Widget for displaying a single chat message.
    
    Supports three roles:
    - user: Right-aligned, distinct background, editable
    - assistant: Left-aligned, Markdown rendering
    - system: Center-aligned, italic style
    
    Uses system palette colors for automatic dark/light mode support.
    
    Signals:
        edit_requested: Emitted when user confirms edit (index, new_content).
    """
    
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"
    
    edit_requested = Signal(int, str)  # index, new_content
    
    def __init__(self, role: str, content: str = "", parent=None):
        super().__init__(parent)
        
        self._role = role
        self._content = content
        self._is_streaming = False
        self._index = -1
        self._edit_mode = False
        
        self._setup_ui()
        self._apply_style()
        
        if content:
            self.set_content(content)
    
    def _setup_ui(self):
        """Initialize the user interface."""
        self.setFrameShape(QFrame.StyledPanel)
        self.setFrameShadow(QFrame.Plain)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)
        
        # Header row with role label and edit button
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        
        # Role label
        self._role_label = QLabel(self._get_role_display())
        role_font = QFont()
        role_font.setBold(True)
        role_font.setPointSize(9)
        self._role_label.setFont(role_font)
        header_layout.addWidget(self._role_label)
        
        header_layout.addStretch()
        
        # Edit button (only for user messages)
        self._edit_btn = QPushButton("✎")
        self._edit_btn.setFixedSize(24, 24)
        self._edit_btn.setToolTip("Edit message")
        self._edit_btn.setCursor(Qt.PointingHandCursor)
        self._edit_btn.clicked.connect(self._on_edit_clicked)
        self._edit_btn.hide()  # Hidden by default, shown on hover
        header_layout.addWidget(self._edit_btn)
        
        # Confirm/Cancel buttons for edit mode
        self._confirm_btn = QPushButton("✓")
        self._confirm_btn.setFixedSize(24, 24)
        self._confirm_btn.setToolTip("Confirm edit (will remove messages after this)")
        self._confirm_btn.setCursor(Qt.PointingHandCursor)
        self._confirm_btn.hide()
        header_layout.addWidget(self._confirm_btn)
        
        self._cancel_btn = QPushButton("✕")
        self._cancel_btn.setFixedSize(24, 24)
        self._cancel_btn.setToolTip("Cancel edit")
        self._cancel_btn.setCursor(Qt.PointingHandCursor)
        self._cancel_btn.hide()
        header_layout.addWidget(self._cancel_btn)
        
        # Connect button signals
        self._confirm_btn.clicked.connect(self._on_edit_confirmed)
        self._cancel_btn.clicked.connect(self._on_edit_cancelled)
        
        layout.addLayout(header_layout)
        
        # Content display
        self._content_display = QTextEdit()
        self._content_display.setReadOnly(True)
        self._content_display.setFrameShape(QFrame.NoFrame)
        self._content_display.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._content_display.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._content_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        
        layout.addWidget(self._content_display)
        
        # Connect to resize content properly
        self._content_display.document().contentsChanged.connect(
            self._adjust_height
        )
    
    def _get_role_display(self) -> str:
        """Get display name for role."""
        if self._role == self.ROLE_USER:
            return "You"
        elif self._role == self.ROLE_ASSISTANT:
            return "Assistant"
        elif self._role == self.ROLE_SYSTEM:
            return "System"
        return self._role.capitalize()
    
    def _get_role_color(self) -> QColor:
        """Get role-specific accent color from palette."""
        palette = QApplication.palette()
        
        if self._role == self.ROLE_USER:
            return palette.color(QPalette.Link)
        elif self._role == self.ROLE_ASSISTANT:
            return palette.color(QPalette.Highlight)
        elif self._role == self.ROLE_SYSTEM:
            return palette.color(QPalette.PlaceholderText)
        
        return palette.color(QPalette.Text)
    
    def _get_background_color(self) -> QColor:
        """Get background color with slight variation per role."""
        palette = QApplication.palette()
        base = palette.color(QPalette.Base)
        
        # Slightly adjust base color per role
        if self._role == self.ROLE_USER:
            return base.lighter(105) if base.lightness() < 128 else base.darker(105)
        elif self._role == self.ROLE_ASSISTANT:
            return base
        elif self._role == self.ROLE_SYSTEM:
            return base.lighter(110) if base.lightness() < 128 else base.darker(110)
        
        return base
    
    def _apply_style(self):
        """Apply role-specific styling using palette colors."""
        role_color = self._get_role_color()
        bg_color = self._get_background_color()
        
        self._role_label.setStyleSheet(f"color: {role_color.name()};")
        
        if self._role == self.ROLE_SYSTEM:
            self._role_label.setStyleSheet(
                f"color: {role_color.name()}; font-style: italic;"
            )
        
        self.setStyleSheet(
            f"MessageWidget {{ background-color: {bg_color.name()}; border-radius: 8px; }}"
        )
    
    def _adjust_height(self):
        """Adjust widget height to fit content."""
        try:
            doc = self._content_display.document()
            doc.setTextWidth(self._content_display.viewport().width())
            
            # Calculate required height
            height = doc.size().height() + 10
            min_height = 30
            max_height = 500
            
            height = max(min_height, min(height, max_height))
            self._content_display.setFixedHeight(int(height))
        except Exception:
            pass
    
    def set_content(self, content: str):
        """
        Set message content.
        
        Args:
            content: Message text (plain text for user, Markdown for assistant).
        """
        self._content = content
        
        if self._role == self.ROLE_ASSISTANT and not self._is_streaming:
            # Render as Markdown for finalized assistant messages
            self._content_display.setMarkdown(content)
        else:
            # Plain text for user messages and streaming
            self._content_display.setPlainText(content)
        
        self._adjust_height()
    
    def append_content(self, text: str):
        """
        Append text to message content (for streaming).
        
        Args:
            text: Text chunk to append.
        """
        self._is_streaming = True
        self._content += text
        
        # During streaming, use plain text for performance
        self._content_display.setPlainText(self._content)
        self._adjust_height()
    
    def finalize(self):
        """
        Finalize message after streaming completes.
        
        Renders Markdown for assistant messages.
        """
        self._is_streaming = False
        
        if self._role == self.ROLE_ASSISTANT:
            self._content_display.setMarkdown(self._content)
            self._adjust_height()
    
    def get_content(self) -> str:
        """Get the message content."""
        return self._content
    
    def get_role(self) -> str:
        """Get the message role."""
        return self._role
    
    def set_index(self, index: int):
        """
        Set the message index in the chat display.
        
        Args:
            index: Position of this message in the message list.
        """
        self._index = index
    
    def get_index(self) -> int:
        """Get the message index."""
        return self._index
    
    def is_editable(self) -> bool:
        """Check if this message can be edited."""
        return self._role == self.ROLE_USER and not self._is_streaming
    
    @Slot()
    def _on_edit_clicked(self):
        """Handle edit button click."""
        if not self.is_editable():
            return
        self._enter_edit_mode()
    
    def _enter_edit_mode(self):
        """Enter edit mode - make content editable."""
        self._edit_mode = True
        self._content_display.setReadOnly(False)
        self._content_display.setFocus()
        
        # Show confirm/cancel, hide edit
        self._edit_btn.hide()
        self._confirm_btn.show()
        self._cancel_btn.show()
        
        # Visual feedback
        self._content_display.setStyleSheet("border: 1px solid palette(highlight);")
    
    def _exit_edit_mode(self):
        """Exit edit mode - restore read-only state."""
        self._edit_mode = False
        self._content_display.setReadOnly(True)
        
        # Hide confirm/cancel
        self._confirm_btn.hide()
        self._cancel_btn.hide()
        
        # Remove visual feedback
        self._content_display.setStyleSheet("")
    
    @Slot()
    def _on_edit_confirmed(self):
        """Handle edit confirmation."""
        new_content = self._content_display.toPlainText().strip()
        
        if not new_content:
            self._on_edit_cancelled()
            return
        
        self._content = new_content
        self._exit_edit_mode()
        self.edit_requested.emit(self._index, new_content)
    
    @Slot()
    def _on_edit_cancelled(self):
        """Handle edit cancellation."""
        # Restore original content
        self._content_display.setPlainText(self._content)
        self._exit_edit_mode()
    
    def enterEvent(self, event):
        """Show edit button on hover for user messages."""
        if self.is_editable() and not self._edit_mode:
            self._edit_btn.show()
        super().enterEvent(event)
    
    def leaveEvent(self, event):
        """Hide edit button when mouse leaves."""
        if not self._edit_mode:
            self._edit_btn.hide()
        super().leaveEvent(event)
    
    def resizeEvent(self, event):
        """Handle resize to adjust content height."""
        super().resizeEvent(event)
        self._adjust_height()