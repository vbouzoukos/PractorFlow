"""
Input widget - Message input with file attachment.

Provides a text input area with send button and file attachment
functionality.
"""

from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QTextEdit,
    QPushButton,
    QLabel,
    QFileDialog,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Signal

from gui.api.tools_client import ToolsClient


class MessageInput(QTextEdit):
    """
    Markdown text input.
    """
    
    submit_requested = Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setPlaceholderText("Type a message ...")
        self.setAcceptRichText(False)
        # Allow vertical resizing (no hard max height)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(100)

class InputWidget(QWidget):
    """
    Message input widget with file attachment support.

    Emits message_submitted signal with message text and file paths.
    """

    message_submitted = Signal(str, list)  # message, file_paths

    def __init__(self, parent=None):
        super().__init__(parent)

        self._file_paths = []
        self._tools_client: Optional[ToolsClient] = None
        self._tools_popup = None
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # File attachment display
        self._files_layout = QHBoxLayout()
        self._files_layout.setContentsMargins(0, 0, 0, 0)
        
        self._files_label = QLabel()
        self._files_label.hide()
        self._files_layout.addWidget(self._files_label)
        
        self._clear_files_btn = QPushButton("✕")
        self._clear_files_btn.setFixedSize(20, 20)
        self._clear_files_btn.setToolTip("Clear attached files")
        self._clear_files_btn.hide()
        self._files_layout.addWidget(self._clear_files_btn)
        
        self._files_layout.addStretch()
        layout.addLayout(self._files_layout)
        
        # Input row
        input_layout = QHBoxLayout()
        input_layout.setSpacing(8)
        
        # Attach button
        self._attach_btn = QPushButton("📎")
        self._attach_btn.setFixedSize(36, 36)
        self._attach_btn.setToolTip("Attach files")
        input_layout.addWidget(self._attach_btn)

        # Tools button
        self._tools_btn = QPushButton("🔧")
        self._tools_btn.setFixedSize(36, 36)
        self._tools_btn.setToolTip("Select tools")
        input_layout.addWidget(self._tools_btn)

        # Text input
        self._text_input = MessageInput()
        input_layout.addWidget(self._text_input)
        
        # Send button
        self._send_btn = QPushButton("Send")
        self._send_btn.setFixedWidth(60)
        self._send_btn.setDefault(True)
        input_layout.addWidget(self._send_btn)
        
        layout.addLayout(input_layout)
    
    def _connect_signals(self):
        """Connect widget signals."""
        self._attach_btn.clicked.connect(self._on_attach_clicked)
        self._tools_btn.clicked.connect(self._on_tools_btn_clicked)
        self._send_btn.clicked.connect(self._on_send_clicked)
        self._text_input.submit_requested.connect(self._on_send_clicked)
        self._clear_files_btn.clicked.connect(self._clear_files)
    
    def set_tools_client(self, client: Optional[ToolsClient]):
        """Set the tools client used by the tools popup."""
        self._tools_client = client

    def _on_tools_btn_clicked(self):
        """Handle tools button click - show tool selection popup."""
        try:
            from gui.widgets.tools_popup import ToolsPopup
            self._tools_popup = ToolsPopup(
                client=self._tools_client,
                anchor_widget=self._tools_btn,
                parent=self,
            )
            self._tools_popup.show()
        except Exception:
            pass  # pragma: no cover

    def _on_attach_clicked(self):
        """Handle attach button click."""
        try:
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Select Files",
                "",
                "All Files (*);;Documents (*.pdf *.docx *.txt *.md);;Images (*.png *.jpg *.jpeg)"
            )
            
            if file_paths:
                self._file_paths.extend(file_paths)
                self._update_files_display()
        except Exception:
            pass  # pragma: no cover
    
    def _on_send_clicked(self):
        """Handle send button click."""
        try:
            message = self._text_input.toMarkdown().strip()
            
            if not message and not self._file_paths:
                return
            
            # Emit signal with message and files
            self.message_submitted.emit(message, self._file_paths.copy())
        except Exception:
            pass  # pragma: no cover
    
    def _update_files_display(self):
        """Update the file attachment display."""
        try:
            if self._file_paths:
                # Show file names
                names = []
                for path in self._file_paths:
                    name = path.split("/")[-1].split("\\")[-1]
                    names.append(name)
                
                self._files_label.setText(f"Files: {', '.join(names)}")
                self._files_label.show()
                self._clear_files_btn.show()
            else:
                self._files_label.hide()
                self._clear_files_btn.hide()
        except Exception:
            pass  # pragma: no cover
    
    def _clear_files(self):
        """Clear attached files."""
        try:
            self._file_paths.clear()
            self._update_files_display()
        except Exception:
            self._file_paths = []
    
    def clear_input(self):
        """Clear the input text and files."""
        try:
            self._text_input.clear()
            self._clear_files()
        except Exception:
            pass  # pragma: no cover
    
    def set_enabled(self, enabled: bool):
        """
        Enable or disable the input widget.
        
        Args:
            enabled: Whether the widget should be enabled.
        """
        try:
            self._text_input.setEnabled(enabled)
            self._send_btn.setEnabled(enabled)
            self._attach_btn.setEnabled(enabled)
            self._tools_btn.setEnabled(enabled)
        except Exception:
            pass  # pragma: no cover
    
    def set_focus(self):
        """Set focus to the text input."""
        try:
            self._text_input.setFocus()
        except Exception:
            pass  # pragma: no cover