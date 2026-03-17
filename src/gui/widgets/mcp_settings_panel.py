"""
MCP Settings panel - View for managing MCP server configurations.

Provides a two-page internal stack:
- index 0: list view (servers list with Edit/Delete per row)
- index 1: form view (create/edit form)
"""

from typing import Optional

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QLabel,
    QMessageBox,
    QGroupBox,
    QFormLayout,
    QLineEdit,
    QTextEdit,
    QComboBox,
    QSpinBox,
    QCheckBox,
    QScrollArea,
    QSizePolicy,
    QFrame,
)
from PySide6.QtCore import Qt, Slot, QSize
from PySide6.QtGui import QIcon

from gui.api.mcp_client import McpClient
from gui.workers.mcp_workers import (
    ListServersWorker,
    CreateServerWorker,
    UpdateServerWorker,
    DeleteServerWorker,
)
from gui.logger import get_logger

logger = get_logger(
    "practorflow-client", level="INFO", log_file="logs/practorflow-client.log"
)


class McpSettingsPanel(QWidget):
    """
    Panel for managing MCP server configurations.

    Contains an internal QStackedWidget with:
    - index 0: list view
    - index 1: form view (create/edit)
    """

    def __init__(self, client: McpClient, parent=None):
        super().__init__(parent)
        self._client = client
        self._current_server_id: Optional[str] = None   # None = create mode, str = edit mode
        self._servers_cache: list = []                   # cached server dicts from last list load

        # Worker references
        self._list_worker = None
        self._delete_worker = None
        self._save_worker = None

        self._setup_ui()
        self._connect_signals()
        self._load_servers()                             # fetch initial server list

    def _setup_ui(self):
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._internal_stack = QStackedWidget()

        self._list_page = self._create_list_page()
        self._form_page = self._create_form_page()

        self._internal_stack.addWidget(self._list_page)   # index 0
        self._internal_stack.addWidget(self._form_page)   # index 1

        layout.addWidget(self._internal_stack)

        # Set initial transport card visibility
        self._on_transport_changed(self._transport_combo.currentText())

    # ------------------------------------------------------------------
    # List Page
    # ------------------------------------------------------------------

    def _create_list_page(self) -> QWidget:
        """Build and return the list page widget."""
        page = QWidget()
        layout = QVBoxLayout(page)

        # Header row — right-aligned "+ New Server" button
        header = QHBoxLayout()
        header.addStretch()
        self._new_server_btn = QPushButton("+ New Server")
        header.addWidget(self._new_server_btn)
        layout.addLayout(header)

        # Server list
        self._server_list = QListWidget()
        layout.addWidget(self._server_list)

        return page

    def _create_server_row(self, server: dict) -> QWidget:
        """Create one row widget for a server."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 8, 8, 8)

        # Left — text labels
        text_layout = QVBoxLayout()

        # Name with enabled/disabled indicator
        name_text = server["name"]
        enabled = server.get("enabled", True)
        if not enabled:
            name_text = f"{name_text}  [disabled]"

        name_label = QLabel(name_text)
        name_label.setStyleSheet(
            "font-weight: bold;" if enabled else "font-weight: bold; color: gray;"
        )
        text_layout.addWidget(name_label)

        transport = server.get("transport", "")
        stdio_config = server.get("stdio_config")
        http_config = server.get("http_config")

        if transport == "stdio" and stdio_config:
            cmd = stdio_config.get("command", "")
            args = " ".join(stdio_config.get("args", []))
            conn_info = f"{cmd} {args}".strip()
        elif http_config:
            conn_info = http_config.get("url", "")
        else:
            conn_info = ""

        subtitle = f"{transport} \u2014 {conn_info}" if conn_info else transport
        subtitle_label = QLabel(subtitle)
        subtitle_label.setStyleSheet("color: gray;")
        subtitle_label.setWordWrap(True)
        text_layout.addWidget(subtitle_label)

        row_layout.addLayout(text_layout, stretch=1)

        # Right — delete button with icon
        delete_btn = QPushButton()
        delete_btn.setFixedSize(24, 24)
        delete_btn.setToolTip("Delete server")
        delete_btn.setCursor(Qt.PointingHandCursor)

        trash_icon = QIcon.fromTheme("edit-delete")
        if trash_icon.isNull():
            trash_icon = QIcon.fromTheme("user-trash")
        if trash_icon.isNull():
            delete_btn.setText("🗑")
        else:
            delete_btn.setIcon(trash_icon)
            delete_btn.setIconSize(QSize(16, 16))

        server_id = server["server_id"]
        delete_btn.clicked.connect(
            lambda checked, sid=server_id, name=server["name"]: self._on_delete_server(sid, name)
        )

        row_layout.addWidget(delete_btn)

        return row

    def _populate_server_list(self, servers: list):
        """Clear and rebuild the list widget from a list of server dicts."""
        self._server_list.clear()

        for server in servers:
            row_widget = self._create_server_row(server)

            item = QListWidgetItem()
            item.setSizeHint(row_widget.sizeHint())
            item.setData(Qt.UserRole, server)           # store full dict for later access

            self._server_list.addItem(item)
            self._server_list.setItemWidget(item, row_widget)

    # ------------------------------------------------------------------
    # Form Page
    # ------------------------------------------------------------------

    def _create_form_page(self) -> QWidget:
        """Build and return the form page widget."""
        page = QWidget()
        layout = QVBoxLayout(page)

        # Top bar
        top_bar = QHBoxLayout()
        self._back_btn = QPushButton("\u2190 Back to Servers")
        self._form_header = QLabel("")
        self._form_header.setStyleSheet("font-weight: bold; font-size: 14px;")
        top_bar.addWidget(self._back_btn)
        top_bar.addStretch()
        top_bar.addWidget(self._form_header)
        layout.addLayout(top_bar)

        # Scroll area with cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        scroll_layout.addWidget(self._create_basic_card())
        scroll_layout.addWidget(self._create_stdio_card())
        scroll_layout.addWidget(self._create_http_card())
        scroll_layout.addWidget(self._create_guidance_card())
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll, stretch=1)

        # Button bar (fixed at bottom, outside scroll)
        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        self._save_btn = QPushButton("Save")
        self._cancel_btn = QPushButton("Cancel")
        btn_bar.addWidget(self._save_btn)
        btn_bar.addWidget(self._cancel_btn)
        layout.addLayout(btn_bar)

        return page

    # ------------------------------------------------------------------
    # Form Cards
    # ------------------------------------------------------------------

    def _create_basic_card(self) -> QGroupBox:
        """Build the Basic group box."""
        card = QGroupBox("Basic")
        form = QFormLayout(card)

        self._name_input = QLineEdit()
        form.addRow("Name:", self._name_input)

        self._transport_combo = QComboBox()
        self._transport_combo.addItems(["stdio", "streamable_http"])
        form.addRow("Transport:", self._transport_combo)

        self._enabled_check = QCheckBox("Enabled")
        self._enabled_check.setChecked(True)
        form.addRow("", self._enabled_check)

        return card

    def _create_stdio_card(self) -> QGroupBox:
        """Build the Stdio Configuration group box."""
        self._stdio_card = QGroupBox("Stdio Configuration")
        form = QFormLayout(self._stdio_card)

        self._stdio_command_input = QLineEdit()
        self._stdio_command_input.setPlaceholderText("npx, python, node")
        form.addRow("Command:", self._stdio_command_input)

        self._stdio_args_input = QLineEdit()
        self._stdio_args_input.setPlaceholderText("-y, @modelcontextprotocol/server-filesystem, /path")
        form.addRow("Arguments:", self._stdio_args_input)

        self._stdio_env_input = QTextEdit()
        self._stdio_env_input.setMaximumHeight(80)
        self._stdio_env_input.setPlaceholderText("KEY=VALUE\nANOTHER_KEY=value")
        form.addRow("Environment Variables:", self._stdio_env_input)

        return self._stdio_card

    def _create_http_card(self) -> QGroupBox:
        """Build the HTTP Configuration group box."""
        self._http_card = QGroupBox("HTTP Configuration")
        form = QFormLayout(self._http_card)

        self._http_url_input = QLineEdit()
        self._http_url_input.setPlaceholderText("https://mcp.example.com/sse")
        form.addRow("URL:", self._http_url_input)

        self._http_headers_input = QTextEdit()
        self._http_headers_input.setMaximumHeight(80)
        self._http_headers_input.setPlaceholderText("Authorization: Bearer token\nX-Custom-Header: value")
        form.addRow("Headers:", self._http_headers_input)

        self._http_timeout_spin = QSpinBox()
        self._http_timeout_spin.setRange(1, 300)
        self._http_timeout_spin.setValue(30)
        form.addRow("Timeout (sec):", self._http_timeout_spin)

        return self._http_card

    def _create_guidance_card(self) -> QGroupBox:
        """Build the Agent Guidance group box."""
        card = QGroupBox("Agent Guidance")
        form = QFormLayout(card)

        self._purpose_input = QLineEdit()
        self._purpose_input.setPlaceholderText("What this server does")
        form.addRow("Purpose:", self._purpose_input)

        self._keywords_input = QLineEdit()
        self._keywords_input.setPlaceholderText("keyword1, keyword2, keyword3")
        form.addRow("Keywords:", self._keywords_input)

        self._category_input = QLineEdit()
        self._category_input.setPlaceholderText("e.g. filesystem, database, search")
        form.addRow("Category:", self._category_input)

        self._tags_input = QLineEdit()
        self._tags_input.setPlaceholderText("tag1, tag2, tag3")
        form.addRow("Tags:", self._tags_input)

        self._use_when_input = QTextEdit()
        self._use_when_input.setMaximumHeight(80)
        self._use_when_input.setPlaceholderText("One condition per line")
        form.addRow("Use when:", self._use_when_input)

        self._do_not_use_when_input = QTextEdit()
        self._do_not_use_when_input.setMaximumHeight(80)
        self._do_not_use_when_input.setPlaceholderText("One condition per line")
        form.addRow("Do not use when:", self._do_not_use_when_input)

        return card

    # ------------------------------------------------------------------
    # Transport Dynamic Visibility
    # ------------------------------------------------------------------

    def _on_transport_changed(self, transport: str):
        """Show/hide config cards based on transport type."""
        is_stdio = transport == "stdio"
        is_http = transport == "streamable_http"
        self._stdio_card.setVisible(is_stdio)
        self._http_card.setVisible(is_http)

    # ------------------------------------------------------------------
    # Form Data Helpers
    # ------------------------------------------------------------------

    def _parse_comma_args(self, text: str) -> list:
        """Split comma-separated args string, strip whitespace, filter empty."""
        return [s.strip() for s in text.split(",") if s.strip()]

    def _parse_env_vars(self, text: str) -> dict:
        """Parse KEY=VALUE lines into a dict."""
        result = {}
        for line in text.split("\n"):
            line = line.strip()
            if "=" in line:
                parts = line.split("=", 1)
                result[parts[0].strip()] = parts[1].strip()
        return result

    def _parse_headers(self, text: str) -> dict:
        """Parse 'Key: Value' lines into a dict."""
        result = {}
        for line in text.split("\n"):
            line = line.strip()
            if ":" in line:
                parts = line.split(":", 1)
                result[parts[0].strip()] = parts[1].strip()
        return result

    def _parse_lines(self, text: str) -> list:
        """Split text into non-empty lines."""
        return [line.strip() for line in text.split("\n") if line.strip()]

    def _collect_form_data(self) -> dict:
        """Read all form widgets into a dict matching the API schema."""
        data = {}

        # Basic
        data["name"] = self._name_input.text().strip()
        data["transport"] = self._transport_combo.currentText()
        data["enabled"] = self._enabled_check.isChecked()

        # Transport-specific config
        transport = data["transport"]
        if transport == "stdio":
            data["stdio_config"] = {
                "command": self._stdio_command_input.text().strip(),
                "args": self._parse_comma_args(self._stdio_args_input.text()),
                "env": self._parse_env_vars(self._stdio_env_input.toPlainText()),
            }
        else:
            data["http_config"] = {
                "url": self._http_url_input.text().strip(),
                "headers": self._parse_headers(self._http_headers_input.toPlainText()),
                "timeout_seconds": self._http_timeout_spin.value(),
            }

        # Agent guidance
        data["purpose"] = self._purpose_input.text().strip()
        data["keywords"] = self._parse_comma_args(self._keywords_input.text())
        data["category"] = self._category_input.text().strip()
        data["tags"] = self._parse_comma_args(self._tags_input.text())
        data["use_when"] = self._parse_lines(self._use_when_input.toPlainText())
        data["do_not_use_when"] = self._parse_lines(self._do_not_use_when_input.toPlainText())

        return data

    # ------------------------------------------------------------------
    # Form Population / Clear
    # ------------------------------------------------------------------

    def _populate_form(self, server: dict):
        """Set all form fields from a server dict (edit mode)."""
        self._name_input.setText(server.get("name", ""))

        transport = server.get("transport", "stdio")
        self._transport_combo.setCurrentText(transport)
        # _on_transport_changed is connected to currentTextChanged so visibility updates automatically

        self._enabled_check.setChecked(server.get("enabled", True))

        stdio_config = server.get("stdio_config") or {}
        self._stdio_command_input.setText(stdio_config.get("command", ""))
        self._stdio_args_input.setText(", ".join(stdio_config.get("args", [])))
        env_lines = "\n".join(f"{k}={v}" for k, v in (stdio_config.get("env") or {}).items())
        self._stdio_env_input.setPlainText(env_lines)

        http_config = server.get("http_config") or {}
        self._http_url_input.setText(http_config.get("url", ""))
        header_lines = "\n".join(f"{k}: {v}" for k, v in (http_config.get("headers") or {}).items())
        self._http_headers_input.setPlainText(header_lines)
        self._http_timeout_spin.setValue(http_config.get("timeout_seconds", 30))

        # Agent guidance
        self._purpose_input.setText(server.get("purpose", ""))
        self._keywords_input.setText(", ".join(server.get("keywords", [])))
        self._category_input.setText(server.get("category", ""))
        self._tags_input.setText(", ".join(server.get("tags", [])))
        self._use_when_input.setPlainText("\n".join(server.get("use_when", [])))
        self._do_not_use_when_input.setPlainText("\n".join(server.get("do_not_use_when", [])))

    def _clear_form(self):
        """Reset every field to defaults (create mode)."""
        self._name_input.clear()
        self._transport_combo.setCurrentIndex(0)   # stdio
        self._enabled_check.setChecked(True)

        self._stdio_command_input.clear()
        self._stdio_args_input.clear()
        self._stdio_env_input.clear()

        self._http_url_input.clear()
        self._http_headers_input.clear()
        self._http_timeout_spin.setValue(30)

        self._purpose_input.clear()
        self._keywords_input.clear()
        self._category_input.clear()
        self._tags_input.clear()
        self._use_when_input.clear()
        self._do_not_use_when_input.clear()

    # ------------------------------------------------------------------
    # Signal Wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        """Connect signals to slots."""
        # List page
        self._new_server_btn.clicked.connect(self._on_new_server)
        self._server_list.itemDoubleClicked.connect(self._on_item_double_clicked)

        # Form page buttons
        self._back_btn.clicked.connect(self._on_back_to_list)
        self._cancel_btn.clicked.connect(self._on_back_to_list)
        self._save_btn.clicked.connect(self._on_save)
        self._transport_combo.currentTextChanged.connect(self._on_transport_changed)

    # ------------------------------------------------------------------
    # Slots — List View Actions
    # ------------------------------------------------------------------

    @Slot(QListWidgetItem)
    def _on_item_double_clicked(self, item: QListWidgetItem):
        """Open edit form on double-click."""
        server = item.data(Qt.UserRole)
        if server:
            self._on_edit_server(server["server_id"])

    @Slot()
    def _on_new_server(self):
        """Navigate to form in create mode."""
        self._current_server_id = None
        self._clear_form()
        self._form_header.setText("Creating New Server")
        # Set initial transport visibility
        self._on_transport_changed(self._transport_combo.currentText())
        self._internal_stack.setCurrentIndex(1)

    def _on_edit_server(self, server_id: str):
        """Find the server in cache and navigate to form in edit mode."""
        server = None
        for s in self._servers_cache:
            if s["server_id"] == server_id:
                server = s
                break

        if server is None:
            return

        self._current_server_id = server_id
        self._populate_form(server)
        self._form_header.setText(f"Editing: {server['name']}")
        self._internal_stack.setCurrentIndex(1)

    def _on_delete_server(self, server_id: str, name: str):
        """Confirm and delete a server."""
        reply = QMessageBox.question(
            self,
            "Delete Server",
            f'Delete "{name}"?\nThis cannot be undone.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        worker = DeleteServerWorker(self._client, server_id, parent=self)
        worker.server_deleted.connect(self._on_server_deleted)
        worker.error_occurred.connect(self._on_worker_error)
        worker.start()
        self._delete_worker = worker

    # ------------------------------------------------------------------
    # Slots — Form View Actions
    # ------------------------------------------------------------------

    @Slot()
    def _on_back_to_list(self):
        """Navigate back to list view without saving."""
        self._internal_stack.setCurrentIndex(0)

    @Slot()
    def _on_save(self):
        """Collect, validate, and submit the form."""
        data = self._collect_form_data()

        # Validate required fields
        errors = []
        if not data.get("name", "").strip():
            errors.append("Name")

        transport = data.get("transport", "")
        if transport == "stdio":
            if not data.get("stdio_config", {}).get("command", "").strip():
                errors.append("Command (stdio)")
        else:
            if not data.get("http_config", {}).get("url", "").strip():
                errors.append("URL (HTTP)")

        if errors:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Required fields:\n\u2022 " + "\n\u2022 ".join(errors),
            )
            return

        if self._current_server_id is None:
            # Create mode
            worker = CreateServerWorker(self._client, data, parent=self)
            worker.server_created.connect(self._on_server_created)
            worker.error_occurred.connect(self._on_worker_error)
            worker.start()
            self._save_worker = worker
        else:
            # Edit mode
            worker = UpdateServerWorker(self._client, self._current_server_id, data, parent=self)
            worker.server_updated.connect(self._on_server_updated)
            worker.error_occurred.connect(self._on_worker_error)
            worker.start()
            self._save_worker = worker

    # ------------------------------------------------------------------
    # Slots — Worker Callbacks
    # ------------------------------------------------------------------

    @Slot(dict)
    def _on_servers_loaded(self, data: dict):
        """Called after ListServersWorker succeeds."""
        self._servers_cache = data.get("servers", [])
        self._populate_server_list(self._servers_cache)
        self._list_worker.safe_delete()
        self._list_worker = None

    @Slot(str)
    def _on_server_deleted(self, server_id: str):
        """Called after DeleteServerWorker succeeds."""
        QMessageBox.information(self, "Deleted", "Server deleted.")
        self._load_servers()                            # refresh list
        self._delete_worker.safe_delete()
        self._delete_worker = None

    @Slot(dict)
    def _on_server_created(self, server: dict):
        """Called after CreateServerWorker succeeds."""
        QMessageBox.information(self, "Success", f"Server '{server['name']}' created.")
        self._load_servers()
        self._internal_stack.setCurrentIndex(0)         # back to list
        self._save_worker.safe_delete()
        self._save_worker = None

    @Slot(dict)
    def _on_server_updated(self, server: dict):
        """Called after UpdateServerWorker succeeds."""
        QMessageBox.information(self, "Success", f"Server '{server['name']}' updated.")
        self._load_servers()
        self._internal_stack.setCurrentIndex(0)         # back to list
        self._save_worker.safe_delete()
        self._save_worker = None

    @Slot(str)
    def _on_worker_error(self, error: str):
        """Handle worker error."""
        QMessageBox.critical(self, "Error", f"Operation failed:\n{error}")

    def _load_servers(self):
        """Fetch server list from backend."""
        worker = ListServersWorker(self._client, parent=self)
        worker.servers_loaded.connect(self._on_servers_loaded)
        worker.error_occurred.connect(self._on_worker_error)
        worker.start()
        self._list_worker = worker
