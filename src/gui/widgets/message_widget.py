"""
Message widget - Individual chat message display.

Displays a single message with role indicator and content.
Supports Markdown rendering for assistant messages.
Uses system palette for automatic dark/light mode support.
"""

from PySide6.QtWidgets import (
    QFrame,
    QVBoxLayout,
    QLabel,
    QTextEdit,
    QSizePolicy,
    QApplication,
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont, QPalette, QColor


class MessageWidget(QFrame):
    """
    Widget for displaying a single chat message.
    
    Supports three roles:
    - user: Right-aligned, distinct background
    - assistant: Left-aligned, Markdown rendering
    - system: Center-aligned, italic style
    
    Uses system palette colors for automatic dark/light mode support.
    """
    
    ROLE_USER = "user"
    ROLE_ASSISTANT = "assistant"
    ROLE_SYSTEM = "system"
    
    def __init__(self, role: str, content: str = "", parent=None):
        super().__init__(parent)
        
        self._role = role
        self._content = content
        self._is_streaming = False
        
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
        
        # Role label
        self._role_label = QLabel(self._get_role_display())
        role_font = QFont()
        role_font.setBold(True)
        role_font.setPointSize(9)
        self._role_label.setFont(role_font)
        layout.addWidget(self._role_label)
        
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
        doc = self._content_display.document()
        doc.setTextWidth(self._content_display.viewport().width())
        
        # Calculate required height
        height = doc.size().height() + 10
        min_height = 30
        max_height = 500
        
        height = max(min_height, min(height, max_height))
        self._content_display.setFixedHeight(int(height))
    
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
    
    def resizeEvent(self, event):
        """Handle resize to adjust content height."""
        super().resizeEvent(event)
        self._adjust_height()