"""
Documents panel - Foldable panel for displaying session documents.

Provides a dropdown panel with:
- List of documents in the current session
- Delete document functionality
- Refresh capability
"""

from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QFrame,
    QSizePolicy,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal, Slot, QSize, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QIcon

from gui.api.session_client import SessionClient
from gui.api.client_data import DocumentInfo
from gui.workers.document_worker import ListDocumentsWorker, DeleteDocumentWorker


class DocumentItemWidget(QWidget):
    """
    Custom widget for document list item display.

    Shows document info with a delete button.

    Signals:
        delete_clicked: Emitted with document_id when delete button is clicked.
    """

    delete_clicked = Signal(str)

    def __init__(self, document: DocumentInfo, parent=None):
        super().__init__(parent)

        self.document = document
        self._setup_ui()

    def _setup_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        # Document icon based on file type
        icon_label = QLabel()
        icon_label.setFixedSize(20, 20)
        icon_label.setText(self._get_file_icon())
        icon_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(icon_label)

        # Document info label
        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        self._update_display()
        layout.addWidget(self._info_label, stretch=1)

        # Delete button
        self._delete_btn = QPushButton()
        self._delete_btn.setFixedSize(24, 24)
        self._delete_btn.setToolTip("Delete document")
        self._delete_btn.setCursor(Qt.PointingHandCursor)

        # Try to use trash icon, fallback to text
        trash_icon = QIcon.fromTheme("edit-delete")
        if trash_icon.isNull():
            trash_icon = QIcon.fromTheme("user-trash")
        if trash_icon.isNull():
            self._delete_btn.setText("🗑")
        else:
            self._delete_btn.setIcon(trash_icon)
            self._delete_btn.setIconSize(QSize(16, 16))

        self._delete_btn.clicked.connect(self._on_delete_clicked)
        layout.addWidget(self._delete_btn)

    def _get_file_icon(self) -> str:
        """Get an emoji icon based on file type."""
        file_type = self.document.file_type.lower().lstrip(".")

        icon_map = {
            "pdf": "📄",
            "doc": "📝",
            "docx": "📝",
            "txt": "📃",
            "md": "📃",
            "csv": "📊",
            "xlsx": "📊",
            "xls": "📊",
            "json": "📋",
            "xml": "📋",
            "py": "🐍",
            "js": "📜",
            "html": "🌐",
            "css": "🎨",
        }

        return icon_map.get(file_type, "📎")

    def _update_display(self):
        """Update the display text."""
        try:
            filename = self.document.filename
            file_type = self.document.file_type.lstrip(".")

            # Truncate long filenames
            if len(filename) > 30:
                filename = filename[:27] + "..."

            display_text = f"<b>{filename}</b><br/><small>{file_type.upper()}</small>"
            self._info_label.setText(display_text)
        except Exception:
            self._info_label.setText("Document")

    def _on_delete_clicked(self):
        """Handle delete button click."""
        try:
            self.delete_clicked.emit(self.document.id)
        except Exception:
            pass

    def get_document_id(self) -> str:
        """Get the document ID."""
        return self.document.id


class DocumentsPanel(QFrame):
    """
    Foldable panel for session documents.

    Displays as a dropdown panel below the toolbar when expanded.

    Signals:
        document_deleted: Emitted with document_id when a document is deleted.
    """

    document_deleted = Signal(str)

    def __init__(self, client: SessionClient, parent=None):
        super().__init__(parent)

        self._client = client
        self._session_id: Optional[str] = None
        self._documents: list = []
        self._is_expanded = False

        # Workers
        self._list_worker: Optional[ListDocumentsWorker] = None
        self._delete_worker: Optional[DeleteDocumentWorker] = None

        self._setup_ui()
        self._connect_signals()

        # Start collapsed
        self.setMaximumHeight(0)
        self.hide()

    def _setup_ui(self):
        """Initialize the user interface."""
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            """
            DocumentsPanel {
                background-color: palette(window);
                border: 1px solid palette(mid);
                border-radius: 4px;
            }
        """
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(4)

        # Header
        header_layout = QHBoxLayout()
        header_layout.setSpacing(4)

        self._title_label = QLabel("Session Documents")
        self._title_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self._title_label)

        header_layout.addStretch()

        self._refresh_btn = QPushButton("⟳")
        self._refresh_btn.setFixedSize(24, 24)
        self._refresh_btn.setToolTip("Refresh documents")
        header_layout.addWidget(self._refresh_btn)

        layout.addLayout(header_layout)

        # Document list
        self._document_list = QListWidget()
        self._document_list.setSpacing(2)
        self._document_list.setAlternatingRowColors(True)
        self._document_list.setMaximumHeight(200)
        layout.addWidget(self._document_list)

        # Status label
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray; font-size: 11px;")
        self._status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_label)

    def set_enabled(self, enabled: bool):
        """
        Enable or disable the widget.

        Args:
            enabled: Whether the widget should be enabled.
        """
        try:
            self.set_enabled(enabled)
        except Exception:
            pass  # pragma: no cover

    def _connect_signals(self):
        """Connect widget signals."""
        self._refresh_btn.clicked.connect(self.refresh_documents)

    def set_session(self, session_id: Optional[str]):
        """
        Set the current session ID.

        Args:
            session_id: Session ID or None to clear.
        """
        self._session_id = session_id
        self._documents = []
        self._document_list.clear()

        if session_id:
            self._status_label.setText("Click refresh to load")
        else:
            self._status_label.setText("No session")

    def toggle_expanded(self):
        """Toggle panel expanded/collapsed state."""
        if self._is_expanded:
            self.collapse()
        else:
            self.expand()

    def expand(self):
        """Expand the panel."""
        if self._is_expanded:
            return

        self._is_expanded = True
        self.show()

        # Animate expansion
        self._animation = QPropertyAnimation(self, b"maximumHeight")
        self._animation.setDuration(200)
        self._animation.setStartValue(0)
        self._animation.setEndValue(280)
        self._animation.setEasingCurve(QEasingCurve.OutCubic)
        self._animation.start()

        # Auto-refresh when expanding
        if self._session_id:
            self.refresh_documents()

    def collapse(self):
        """Collapse the panel."""
        if not self._is_expanded:
            return

        self._is_expanded = False

        # Animate collapse
        self._animation = QPropertyAnimation(self, b"maximumHeight")
        self._animation.setDuration(200)
        self._animation.setStartValue(self.height())
        self._animation.setEndValue(0)
        self._animation.setEasingCurve(QEasingCurve.InCubic)
        self._animation.finished.connect(self.hide)
        self._animation.start()

    def is_expanded(self) -> bool:
        """Check if panel is expanded."""
        return self._is_expanded

    def refresh_documents(self):
        """Refresh the document list from the server."""
        try:
            if not self._session_id:
                self._status_label.setText("No session")
                return

            if self._list_worker and self._list_worker.isRunning():
                return

            self._status_label.setText("Loading...")
            self._document_list.setEnabled(False)

            self._list_worker = ListDocumentsWorker(
                self._client, self._session_id, parent=self
            )
            self._list_worker.documents_loaded.connect(self._on_documents_loaded)
            self._list_worker.error_occurred.connect(self._on_list_error)
            self._list_worker.finished.connect(self._cleanup_list_worker)
            self._list_worker.start()
        except Exception:
            self._status_label.setText("Error")
            self._document_list.setEnabled(True)

    @Slot()
    def _cleanup_list_worker(self):
        """Clean up list worker after it finishes."""
        try:
            if self._list_worker:
                self._list_worker.deleteLater()
                self._list_worker = None
        except Exception:
            self._list_worker = None

    @Slot(list)
    def _on_documents_loaded(self, documents: list):
        """Handle documents loaded from server."""
        try:
            self._documents = documents
            self._document_list.clear()

            if not documents:
                self._status_label.setText("No documents")
                self._document_list.setEnabled(True)
                return

            for doc in documents:
                # Create custom widget for the item
                item_widget = DocumentItemWidget(doc)
                item_widget.delete_clicked.connect(self._confirm_delete)

                # Create list item and set size hint
                item = QListWidgetItem(self._document_list)
                item.setSizeHint(item_widget.sizeHint())
                item.setData(Qt.UserRole, doc.id)

                # Add widget to list
                self._document_list.setItemWidget(item, item_widget)

            self._document_list.setEnabled(True)
            self._status_label.setText(f"{len(documents)} document(s)")
        except Exception:
            self._document_list.setEnabled(True)
            self._status_label.setText("Error loading")

    @Slot(str)
    def _on_list_error(self, error: str):
        """Handle error loading documents."""
        try:
            self._document_list.setEnabled(True)
            self._status_label.setText("Error loading")
        except Exception:
            pass

    @Slot(str)
    def _confirm_delete(self, document_id: str):
        """Show confirmation dialog before deleting."""
        try:
            # Find document info
            doc_name = document_id
            for doc in self._documents:
                if doc.id == document_id:
                    doc_name = doc.filename
                    break

            reply = QMessageBox.question(
                self,
                "Delete Document",
                f"Delete '{doc_name}' from this session?\n\n"
                "This will remove the document from the knowledge base.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )

            if reply == QMessageBox.Yes:
                self._delete_document(document_id)
        except Exception:
            pass

    def _delete_document(self, document_id: str):
        """Delete a document."""
        try:
            if not self._session_id:
                return

            if self._delete_worker and self._delete_worker.isRunning():
                return

            self._status_label.setText("Deleting...")

            self._delete_worker = DeleteDocumentWorker(
                self._client, self._session_id, document_id, parent=self
            )
            self._delete_worker.document_deleted.connect(self._on_document_deleted)
            self._delete_worker.error_occurred.connect(self._on_delete_error)
            self._delete_worker.finished.connect(self._cleanup_delete_worker)
            self._delete_worker.start()
        except Exception:
            self._status_label.setText("Error")

    @Slot()
    def _cleanup_delete_worker(self):
        """Clean up delete worker after it finishes."""
        try:
            if self._delete_worker:
                self._delete_worker.deleteLater()
                self._delete_worker = None
        except Exception:
            self._delete_worker = None

    @Slot(str)
    def _on_document_deleted(self, document_id: str):
        """Handle document deleted."""
        try:
            self.document_deleted.emit(document_id)
            self.refresh_documents()
        except Exception:
            pass

    @Slot(str)
    def _on_delete_error(self, error: str):
        """Handle error deleting document."""
        try:
            self._status_label.setText("Delete failed")
            QMessageBox.warning(
                self, "Delete Failed", f"Failed to delete document:\n{error}"
            )
        except Exception:
            pass

    def shutdown(self):
        """Shutdown all workers - call before destroying."""
        try:
            if self._list_worker and self._list_worker.isRunning():
                self._list_worker.quit()
                self._list_worker.wait(1000)
            if self._delete_worker and self._delete_worker.isRunning():
                self._delete_worker.quit()
                self._delete_worker.wait(1000)
        except Exception:
            pass
