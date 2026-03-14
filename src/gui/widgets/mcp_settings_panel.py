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
from PySide6.QtCore import Qt, Slot

from gui.api.mcp_client import McpClient
from gui.workers.mcp_workers import (
    ListServersWorker,
    CreateServerWorker,
    UpdateServerWorker,
    DeleteServerWorker,
    DiscoverToolsWorker,
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
        self._discover_worker = None

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
        name_label = QLabel(server["name"])
        name_label.setStyleSheet("font-weight: bold;")
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

        # Right — action buttons
        edit_btn = QPushButton("Edit")
        delete_btn = QPushButton("Delete")

        # Capture server_id for button lambdas
        server_id = server["server_id"]
        edit_btn.clicked.connect(lambda checked, sid=server_id: self._on_edit_server(sid))
        delete_btn.clicked.connect(
            lambda checked, sid=server_id, name=server["name"]: self._on_delete_server(sid, name)
        )

        row_layout.addWidget(edit_btn)
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
        self._tools_card = self._create_tools_card()
        scroll_layout.addWidget(self._tools_card)
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
        self._transport_combo.addItems(["stdio", "sse", "streamable_http"])
        form.addRow("Transport:", self._transport_combo)

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

    def _create_tools_card(self) -> QGroupBox:
        """Build the Tools group box (edit mode only)."""
        card = QGroupBox("Tools")
        card_layout = QVBoxLayout(card)

        # Discover button + status label
        discover_row = QHBoxLayout()
        self._discover_btn = QPushButton("Discover Tools")
        self._discover_status_label = QLabel("")
        self._discover_status_label.setStyleSheet("color: gray;")
        discover_row.addWidget(self._discover_btn)
        discover_row.addWidget(self._discover_status_label)
        discover_row.addStretch()
        card_layout.addLayout(discover_row)

        # Tool rows container
        self._tool_rows_container = QVBoxLayout()
        card_layout.addLayout(self._tool_rows_container)
        card_layout.addStretch()

        self._tool_rows: list = []   # list of dicts holding widget references per row

        return card

    # ------------------------------------------------------------------
    # Transport Dynamic Visibility
    # ------------------------------------------------------------------

    def _on_transport_changed(self, transport: str):
        """Show/hide config cards based on transport type."""
        is_stdio = transport == "stdio"
        is_http = transport in ("sse", "streamable_http")
        self._stdio_card.setVisible(is_stdio)
        self._http_card.setVisible(is_http)

    # ------------------------------------------------------------------
    # Tool Rows
    # ------------------------------------------------------------------

    def _add_tool_row(self, tool_name: str, tool_desc: str, enabled: bool = True, desc_override: str = ""):
        """Add one discovered tool row to the tools card."""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QHBoxLayout(frame)

        enabled_check = QCheckBox()
        enabled_check.setChecked(enabled)
        layout.addWidget(enabled_check)

        name_label = QLabel(tool_name)
        name_label.setStyleSheet("font-weight: bold;")
        name_label.setMinimumWidth(150)
        layout.addWidget(name_label)

        desc_label = QLabel(tool_desc)
        desc_label.setStyleSheet("color: gray;")
        desc_label.setWordWrap(True)
        layout.addWidget(desc_label, stretch=1)

        desc_override_input = QLineEdit()
        desc_override_input.setPlaceholderText("Description override (optional)")
        desc_override_input.setText(desc_override)
        desc_override_input.setMaximumWidth(220)
        layout.addWidget(desc_override_input)

        row_data = {
            "frame": frame,
            "name": tool_name,
            "enabled": enabled_check,
            "desc_override": desc_override_input,
        }
        self._tool_rows.append(row_data)
        self._tool_rows_container.addWidget(frame)

    def _clear_tool_rows(self):
        """Remove all tool rows."""
        for row_data in self._tool_rows:
            frame = row_data["frame"]
            self._tool_rows_container.removeWidget(frame)
            frame.deleteLater()
        self._tool_rows.clear()

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

    def _collect_form_data(self) -> dict:
        """Read all form widgets into a dict matching the API schema."""
        data = {}

        # Basic
        data["name"] = self._name_input.text().strip()
        data["transport"] = self._transport_combo.currentText()

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

        # Tools (from tool rows)
        data["tools"] = []
        for row in self._tool_rows:
            tool = {
                "name": row["name"],
                "enabled": row["enabled"].isChecked(),
            }
            override = row["desc_override"].text().strip()
            if override:
                tool["description"] = override
            data["tools"].append(tool)

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

        # Pre-populate tool rows from existing server tools
        self._clear_tool_rows()
        for tool in server.get("tools", []):
            self._add_tool_row(
                tool_name=tool.get("name", ""),
                tool_desc="",
                enabled=tool.get("enabled", True),
                desc_override=tool.get("description") or "",
            )

    def _clear_form(self):
        """Reset every field to defaults (create mode)."""
        self._name_input.clear()
        self._transport_combo.setCurrentIndex(0)   # stdio

        self._stdio_command_input.clear()
        self._stdio_args_input.clear()
        self._stdio_env_input.clear()

        self._http_url_input.clear()
        self._http_headers_input.clear()
        self._http_timeout_spin.setValue(30)

        self._clear_tool_rows()
        self._discover_status_label.setText("")

    # ------------------------------------------------------------------
    # Signal Wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        """Connect signals to slots."""
        # List page
        self._new_server_btn.clicked.connect(self._on_new_server)

        # Form page buttons
        self._back_btn.clicked.connect(self._on_back_to_list)
        self._cancel_btn.clicked.connect(self._on_back_to_list)
        self._save_btn.clicked.connect(self._on_save)
        self._discover_btn.clicked.connect(self._on_discover_tools)
        self._transport_combo.currentTextChanged.connect(self._on_transport_changed)

    # ------------------------------------------------------------------
    # Slots — List View Actions
    # ------------------------------------------------------------------

    @Slot()
    def _on_new_server(self):
        """Navigate to form in create mode."""
        self._current_server_id = None
        self._clear_form()
        self._form_header.setText("Creating New Server")
        self._tools_card.setVisible(False)
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
        self._tools_card.setVisible(True)
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

    @Slot()
    def _on_discover_tools(self):
        """Discover tools from the current server (edit mode only)."""
        if self._current_server_id is None:
            return

        self._discover_status_label.setText("Discovering...")
        self._discover_btn.setEnabled(False)

        worker = DiscoverToolsWorker(self._client, self._current_server_id, parent=self)
        worker.tools_discovered.connect(self._on_tools_discovered)
        worker.error_occurred.connect(self._on_discover_error)
        worker.start()
        self._discover_worker = worker

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

    @Slot(dict)
    def _on_tools_discovered(self, data: dict):
        """Populate tool rows from discovery response."""
        self._discover_btn.setEnabled(True)

        connected = data.get("connected", False)
        error = data.get("error")
        tools = data.get("tools", [])

        if not connected or error:
            self._discover_status_label.setText(f"Error: {error or 'Connection failed'}")
            self._discover_status_label.setStyleSheet("color: red;")
        else:
            self._discover_status_label.setText(f"Connected \u2014 {len(tools)} tool(s) found")
            self._discover_status_label.setStyleSheet("color: green;")

        # Merge with existing tool configs: preserve enabled state/override for known tools
        existing_by_name = {row["name"]: row for row in self._tool_rows}
        self._clear_tool_rows()

        for tool in tools:
            name = tool.get("name", "")
            desc = tool.get("description", "")
            existing = existing_by_name.get(name)
            enabled = existing["enabled"].isChecked() if existing else True
            desc_override = existing["desc_override"].text().strip() if existing else ""
            self._add_tool_row(name, desc, enabled, desc_override)

        self._discover_worker.safe_delete()
        self._discover_worker = None

    @Slot(str)
    def _on_discover_error(self, error: str):
        """Handle discover worker error."""
        self._discover_btn.setEnabled(True)
        self._discover_status_label.setText(f"Error: {error}")
        self._discover_status_label.setStyleSheet("color: red;")
        if self._discover_worker:
            self._discover_worker.safe_delete()
            self._discover_worker = None

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
