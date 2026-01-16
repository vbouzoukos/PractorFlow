"""
History panel - Collapsible panel for displaying and selecting past sessions.

Provides a sidebar panel with:
- List of past sessions
- Click to load/switch session
- Delete session from history
- Search/filter (placeholder for future LLM-generated titles)
"""

from datetime import datetime
from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QFrame,
    QSizePolicy,
    QMenu,
    QMessageBox,
)
from PySide6.QtCore import Qt, Signal, Slot, QSize
from PySide6.QtGui import QAction, QIcon

from gui.api.session_client import SessionClient
from gui.api.client_data import SessionSummary, SessionHistory
from gui.workers.history_worker import ListSessionsWorker, GetHistoryWorker, DeleteSessionWorker


class SessionItemWidget(QWidget):
    """
    Custom widget for session list item display.
    
    Shows session info with a delete button.
    
    Signals:
        delete_clicked: Emitted with session_id when delete button is clicked.
        item_clicked: Emitted with session_id when item area is clicked.
        item_double_clicked: Emitted with session_id when item is double-clicked.
    """
    
    delete_clicked = Signal(str)
    item_clicked = Signal(str)
    item_double_clicked = Signal(str)
    
    def __init__(self, session: SessionSummary, parent=None):
        super().__init__(parent)
        
        self.session = session
        self._setup_ui()
    
    def _setup_ui(self):
        """Initialize the user interface."""
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Session info label
        self._info_label = QLabel()
        self._info_label.setWordWrap(True)
        self._update_display()
        layout.addWidget(self._info_label, stretch=1)
        
        # Delete button with trash icon
        self._delete_btn = QPushButton()
        self._delete_btn.setFixedSize(24, 24)
        self._delete_btn.setToolTip("Delete session")
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
    
    def _update_display(self):
        """Update the display text."""
        try:
            # Format timestamp
            try:
                dt = datetime.fromisoformat(self.session.updated_at)
                time_str = dt.strftime("%b %d, %H:%M")
            except (ValueError, TypeError):
                time_str = "Unknown"
            
            # Create display text
            msg_count = self.session.message_count
            msg_text = f"{msg_count} msg" if msg_count == 1 else f"{msg_count} msgs"
            
            # Use title if available, otherwise "Untitled"
            session_title = self.session.title if self.session.title else "Untitled"
            
            display_text = f"<b>{session_title}</b><br/><small>{time_str} · {msg_text}</small>"
            self._info_label.setText(display_text)
        except Exception:
            self._info_label.setText("Session")
    
    def _on_delete_clicked(self):
        """Handle delete button click."""
        try:
            self.delete_clicked.emit(self.session.session_id)
        except Exception:
            pass  # pragma: no cover
    
    def mouseDoubleClickEvent(self, event):
        """Handle double click to load session."""
        try:
            self.item_double_clicked.emit(self.session.session_id)
            super().mouseDoubleClickEvent(event)
        except Exception:
            pass  # pragma: no cover
    
    def get_session_id(self) -> str:
        """Get the session ID."""
        return self.session.session_id


class HistoryPanel(QFrame):
    """
    Collapsible panel for session history.
    
    Signals:
        session_selected: Emitted with SessionHistory when a session is loaded.
        session_deleted: Emitted with session_id when a session is deleted.
    """
    
    session_selected = Signal(object)
    session_deleted = Signal(str)
    
    def __init__(self, client: SessionClient, parent=None):
        super().__init__(parent)
        
        self._client = client
        self._sessions = []
        self._session_widgets = {}  # Map session_id -> SessionItemWidget
        self._current_session_id = None
        self._is_collapsed = False
        
        # Workers
        self._list_worker = None
        self._history_worker = None
        self._delete_worker = None
        
        self._setup_ui()
        self._connect_signals()
    
    def _setup_ui(self):
        """Initialize the user interface."""
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setMinimumWidth(200)
        self.setMaximumWidth(280)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)
        
        # Header with collapse button
        header_layout = QHBoxLayout()
        header_layout.setSpacing(4)
        
        self._collapse_btn = QPushButton("◀")
        self._collapse_btn.setFixedSize(24, 24)
        self._collapse_btn.setToolTip("Collapse panel")
        header_layout.addWidget(self._collapse_btn)
        
        self._title_label = QLabel("Session History")
        self._title_label.setStyleSheet("font-weight: bold;")
        header_layout.addWidget(self._title_label)
        
        header_layout.addStretch()
        
        self._refresh_btn = QPushButton("⟳")
        self._refresh_btn.setFixedSize(24, 24)
        self._refresh_btn.setToolTip("Refresh sessions")
        header_layout.addWidget(self._refresh_btn)
        
        layout.addLayout(header_layout)
        
        # Search box (disabled for now - placeholder for future title search)
        self._search_box = QLineEdit()
        self._search_box.setPlaceholderText("Search...")
        self._search_box.setEnabled(False)
        self._search_box.setToolTip("Search by title")
        layout.addWidget(self._search_box)
        
        # Session list
        self._session_list = QListWidget()
        self._session_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._session_list.setSpacing(2)
        self._session_list.setAlternatingRowColors(False)
        layout.addWidget(self._session_list, stretch=1)
        
        # Status label
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: gray; font-size: 11px;")
        self._status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self._status_label)
        
        # Content widget (to hide when collapsed)
        self._content_widget = QWidget()
        content_layout = QVBoxLayout(self._content_widget)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(4)
        
        # Move widgets to content widget for collapsing
        layout.removeWidget(self._search_box)
        layout.removeWidget(self._session_list)
        layout.removeWidget(self._status_label)
        
        content_layout.addWidget(self._search_box)
        content_layout.addWidget(self._session_list, stretch=1)
        content_layout.addWidget(self._status_label)
        
        layout.addWidget(self._content_widget)
    
    def _connect_signals(self):
        """Connect widget signals."""
        self._collapse_btn.clicked.connect(self._toggle_collapse)
        self._refresh_btn.clicked.connect(self.refresh_sessions)
        self._session_list.itemDoubleClicked.connect(self._on_item_double_clicked)
        self._session_list.customContextMenuRequested.connect(self._show_context_menu)
    
    def _toggle_collapse(self):
        """Toggle panel collapsed state."""
        try:
            self._is_collapsed = not self._is_collapsed
            
            if self._is_collapsed:
                self._content_widget.hide()
                self._collapse_btn.setText("▶")
                self._collapse_btn.setToolTip("Expand panel")
                self._title_label.hide()
                self._refresh_btn.hide()
                self.setMaximumWidth(32)
                self.setMinimumWidth(32)
            else:
                self._content_widget.show()
                self._collapse_btn.setText("◀")
                self._collapse_btn.setToolTip("Collapse panel")
                self._title_label.show()
                self._refresh_btn.show()
                self.setMaximumWidth(280)
                self.setMinimumWidth(200)
        except Exception:
            pass  # pragma: no cover
    
    def refresh_sessions(self):
        """Refresh the session list from the server."""
        try:
            if self._list_worker and self._list_worker.isRunning():
                return
            
            self._status_label.setText("Loading...")
            self._session_list.setEnabled(False)
            
            self._list_worker = ListSessionsWorker(self._client, parent=self)
            self._list_worker.sessions_loaded.connect(self._on_sessions_loaded)
            self._list_worker.error_occurred.connect(self._on_list_error)
            self._list_worker.finished.connect(self._cleanup_list_worker)
            self._list_worker.start()
        except Exception:
            self._status_label.setText("Error")
            self._session_list.setEnabled(True)
    
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
    def _on_sessions_loaded(self, sessions):
        """Handle sessions loaded from server."""
        try:
            self._sessions = sessions
            self._session_list.clear()
            self._session_widgets.clear()
            
            for session in sessions:
                # Create custom widget for the item
                item_widget = SessionItemWidget(session)
                item_widget.delete_clicked.connect(self._confirm_delete)
                item_widget.item_double_clicked.connect(self._load_session)
                
                # Create list item and set size hint
                item = QListWidgetItem(self._session_list)
                item.setSizeHint(item_widget.sizeHint())
                item.setData(Qt.UserRole, session.session_id)
                
                # Add widget to list
                self._session_list.setItemWidget(item, item_widget)
                self._session_widgets[session.session_id] = item_widget
            
            self._session_list.setEnabled(True)
            self._status_label.setText(f"{len(sessions)} session(s)")
            
            # Highlight current session if set
            self._highlight_current_session()
        except Exception:
            self._session_list.setEnabled(True)
            self._status_label.setText("Error loading")
    
    @Slot(str)
    def _on_list_error(self, error):
        """Handle error loading sessions."""
        try:
            self._session_list.setEnabled(True)
            self._status_label.setText("Error loading")
        except Exception:
            pass  # pragma: no cover
    
    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Handle double click to load session."""
        try:
            session_id = item.data(Qt.UserRole)
            self._load_session(session_id)
        except Exception:
            pass  # pragma: no cover
    
    def _load_session(self, session_id: str):
        """Load a session's history."""
        try:
            if self._history_worker and self._history_worker.isRunning():
                return
            
            self._status_label.setText("Loading session...")
            
            self._history_worker = GetHistoryWorker(self._client, session_id, parent=self)
            self._history_worker.history_loaded.connect(self._on_history_loaded)
            self._history_worker.not_found.connect(self._on_history_not_found)
            self._history_worker.error_occurred.connect(self._on_history_error)
            self._history_worker.finished.connect(self._cleanup_history_worker)
            self._history_worker.start()
        except Exception:
            self._status_label.setText("Error")
    
    @Slot()
    def _cleanup_history_worker(self):
        """Clean up history worker after it finishes."""
        try:
            if self._history_worker:
                self._history_worker.deleteLater()
                self._history_worker = None
        except Exception:
            self._history_worker = None
    
    @Slot(object)
    def _on_history_loaded(self, history: SessionHistory):
        """Handle session history loaded."""
        try:
            self._current_session_id = history.session_id
            self._status_label.setText(f"{len(self._sessions)} session(s)")
            
            self._highlight_current_session()
            self.session_selected.emit(history)
        except Exception:
            pass  # pragma: no cover
    
    @Slot(str)
    def _on_history_not_found(self, session_id: str):
        """Handle session not found."""
        try:
            self._status_label.setText("Session not found")
            
            # Refresh list to remove stale entry
            self.refresh_sessions()
        except Exception:
            pass  # pragma: no cover
    
    @Slot(str)
    def _on_history_error(self, error: str):
        """Handle error loading history."""
        try:
            self._status_label.setText(f"Error loading {error}")
        except Exception:
            pass  # pragma: no cover
    
    def _show_context_menu(self, position):
        """Show context menu for session item."""
        try:
            item = self._session_list.itemAt(position)
            if item is None:
                return
            
            session_id = item.data(Qt.UserRole)
            
            menu = QMenu(self)
            
            load_action = QAction("Load Session", self)
            load_action.triggered.connect(lambda: self._load_session(session_id))
            menu.addAction(load_action)
            
            menu.addSeparator()
            
            delete_action = QAction("Delete Session", self)
            delete_action.triggered.connect(lambda: self._confirm_delete(session_id))
            menu.addAction(delete_action)
            
            menu.exec_(self._session_list.mapToGlobal(position))
        except Exception:
            pass  # pragma: no cover
    
    @Slot(str)
    def _confirm_delete(self, session_id: str):
        """Show confirmation dialog before deleting."""
        try:
            reply = QMessageBox.question(
                self,
                "Delete Session",
                "Delete this session?\nThis cannot be undone.",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            
            if reply == QMessageBox.Yes:
                self._delete_session(session_id)
        except Exception:
            pass  # pragma: no cover
    
    def _delete_session(self, session_id: str):
        """Delete a session."""
        try:
            if self._delete_worker and self._delete_worker.isRunning():
                return
            
            self._status_label.setText("Deleting...")
            
            self._delete_worker = DeleteSessionWorker(self._client, session_id, parent=self)
            self._delete_worker.session_deleted.connect(self._on_session_deleted)
            self._delete_worker.error_occurred.connect(self._on_delete_error)
            self._delete_worker.finished.connect(self._cleanup_delete_worker)
            self._delete_worker.start()
        except Exception:
            self._status_label.setText("Delete failed")
    
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
    def _on_session_deleted(self, session_id: str):
        """Handle session deleted."""
        try:
            # Clear current session if it was deleted
            if self._current_session_id == session_id:
                self._current_session_id = None
            
            self.session_deleted.emit(session_id)
            self.refresh_sessions()
        except Exception:
            pass  # pragma: no cover
    
    @Slot(str)
    def _on_delete_error(self, error: str):
        """Handle error deleting session."""
        try:
            self._status_label.setText("Delete failed")
        except Exception:
            pass  # pragma: no cover
    
    def _highlight_current_session(self):
        """Highlight the current session in the list."""
        try:
            for i in range(self._session_list.count()):
                item = self._session_list.item(i)
                session_id = item.data(Qt.UserRole)
                
                if session_id == self._current_session_id:
                    item.setSelected(True)
                    self._session_list.scrollToItem(item)
                else:
                    item.setSelected(False)
        except Exception:
            pass  # pragma: no cover
    
    def set_current_session(self, session_id: Optional[str]):
        """
        Set the current session ID.
        
        Args:
            session_id: Current session ID or None.
        """
        try:
            self._current_session_id = session_id
            self._highlight_current_session()
        except Exception:
            pass  # pragma: no cover
    
    def is_collapsed(self) -> bool:
        """Check if panel is collapsed."""
        return self._is_collapsed
    
    def set_collapsed(self, collapsed: bool):
        """Set panel collapsed state."""
        try:
            if collapsed != self._is_collapsed:
                self._toggle_collapse()
        except Exception:
            pass  # pragma: no cover
    
    def shutdown(self):
        """Shutdown all workers - call before destroying."""
        try:
            if self._list_worker:
                if self._list_worker.isRunning():
                    self._list_worker.wait(2000)
                self._list_worker.deleteLater()
                self._list_worker = None
        except Exception:
            self._list_worker = None
        
        try:
            if self._history_worker:
                if self._history_worker.isRunning():
                    self._history_worker.wait(2000)
                self._history_worker.deleteLater()
                self._history_worker = None
        except Exception:
            self._history_worker = None
        
        try:
            if self._delete_worker:
                if self._delete_worker.isRunning():
                    self._delete_worker.wait(2000)
                self._delete_worker.deleteLater()
                self._delete_worker = None
        except Exception:
            self._delete_worker = None