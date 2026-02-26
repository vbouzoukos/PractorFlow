"""
API Tools panel - View for managing API tool configurations.

Provides a two-page internal stack:
- index 0: list view (tools list with Edit/Delete per row)
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

from gui.api.api_tools_client import ApiToolsClient
from gui.workers.api_tools_workers import (
    ListToolsWorker,
    CreateToolWorker,
    UpdateToolWorker,
    DeleteToolWorker,
    GetSecretsWorker,
)
from gui.logger import get_logger

logger = get_logger(
    "practorflow-client", level="INFO", log_file="logs/practorflow-client.log"
)


class ApiToolsPanel(QWidget):
    """
    Panel for managing API tool configurations.

    Contains an internal QStackedWidget with:
    - index 0: list view
    - index 1: form view (create/edit)
    """

    def __init__(self, client: ApiToolsClient, parent=None):
        super().__init__(parent)
        self._client = client
        self._current_tool_id: Optional[str] = None   # None = create mode, str = edit mode
        self._tools_cache: list = []                   # cached tool dicts from last list load

        # Worker references
        self._list_worker = None
        self._delete_worker = None
        self._save_worker = None
        self._secrets_worker = None

        self._setup_ui()
        self._connect_signals()
        self._load_tools()                             # fetch initial tool list

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

        # Header row — right-aligned "+ New Tool" button
        header = QHBoxLayout()
        header.addStretch()
        self._new_tool_btn = QPushButton("+ New Tool")
        header.addWidget(self._new_tool_btn)
        layout.addLayout(header)

        # Tool list
        self._tool_list = QListWidget()
        layout.addWidget(self._tool_list)

        return page

    def _create_tool_row(self, tool: dict) -> QWidget:
        """Create one row widget for a tool."""
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 8, 8, 8)

        # Left — text labels
        text_layout = QVBoxLayout()
        name_label = QLabel(tool["name"])
        name_label.setStyleSheet("font-weight: bold;")
        text_layout.addWidget(name_label)

        desc_label = QLabel(tool.get("description", ""))
        desc_label.setStyleSheet("color: gray;")
        desc_label.setWordWrap(True)
        text_layout.addWidget(desc_label)

        row_layout.addLayout(text_layout, stretch=1)

        # Right — action buttons
        edit_btn = QPushButton("Edit")
        delete_btn = QPushButton("Delete")

        # Capture tool_id for button lambdas
        tool_id = tool["tool_id"]
        edit_btn.clicked.connect(lambda checked, tid=tool_id: self._on_edit_tool(tid))
        delete_btn.clicked.connect(
            lambda checked, tid=tool_id, name=tool["name"]: self._on_delete_tool(tid, name)
        )

        row_layout.addWidget(edit_btn)
        row_layout.addWidget(delete_btn)

        return row

    def _populate_tool_list(self, tools: list):
        """Clear and rebuild the list widget from a list of tool dicts."""
        self._tool_list.clear()

        for tool in tools:
            row_widget = self._create_tool_row(tool)

            item = QListWidgetItem()
            item.setSizeHint(row_widget.sizeHint())
            item.setData(Qt.UserRole, tool)           # store full dict for later access

            self._tool_list.addItem(item)
            self._tool_list.setItemWidget(item, row_widget)

    # ------------------------------------------------------------------
    # Form Page
    # ------------------------------------------------------------------

    def _create_form_page(self) -> QWidget:
        """Build and return the form page widget."""
        page = QWidget()
        layout = QVBoxLayout(page)

        # Top bar
        top_bar = QHBoxLayout()
        self._back_btn = QPushButton("← Back to Tools")
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
        scroll_layout.addWidget(self._create_auth_card())
        scroll_layout.addWidget(self._create_reliability_card())
        scroll_layout.addWidget(self._create_discovery_card())
        scroll_layout.addWidget(self._create_parameters_card())
        scroll_layout.addWidget(self._create_response_card())
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

        self._description_input = QTextEdit()
        self._description_input.setMaximumHeight(60)
        form.addRow("Description:", self._description_input)

        self._base_url_input = QLineEdit()
        self._base_url_input.setPlaceholderText("https://api.example.com")
        form.addRow("Base URL:", self._base_url_input)

        # Method + Path on same conceptual row
        method_path = QHBoxLayout()
        self._method_combo = QComboBox()
        self._method_combo.addItems(["GET", "POST", "PUT", "PATCH", "DELETE"])
        method_path.addWidget(QLabel("Method:"))
        method_path.addWidget(self._method_combo)
        method_path.addSpacing(16)
        self._path_input = QLineEdit()
        self._path_input.setPlaceholderText("/v1/endpoint/{param}")
        method_path.addWidget(QLabel("Path:"))
        method_path.addWidget(self._path_input, stretch=1)
        form.addRow(method_path)

        self._body_type_combo = QComboBox()
        self._body_type_combo.addItems(["none", "json", "form", "multipart"])
        form.addRow("Body Type:", self._body_type_combo)

        return card

    def _create_auth_card(self) -> QGroupBox:
        """Build the Authentication group box."""
        card = QGroupBox("Authentication")
        self._auth_form = QFormLayout(card)

        self._auth_type_combo = QComboBox()
        self._auth_type_combo.addItems(["none", "api_key", "bearer", "basic"])
        self._auth_form.addRow("Auth Type:", self._auth_type_combo)

        self._auth_secret_input = QLineEdit()
        self._auth_secret_input.setEchoMode(QLineEdit.Password)
        self._auth_form.addRow("Secret/Token:", self._auth_secret_input)

        self._auth_username_input = QLineEdit()
        self._auth_form.addRow("Username:", self._auth_username_input)

        self._auth_password_input = QLineEdit()
        self._auth_password_input.setEchoMode(QLineEdit.Password)
        self._auth_form.addRow("Password:", self._auth_password_input)

        self._auth_key_location_combo = QComboBox()
        self._auth_key_location_combo.addItems(["header", "query", "body"])
        self._auth_form.addRow("Key Location:", self._auth_key_location_combo)

        self._auth_key_name_input = QLineEdit()
        self._auth_key_name_input.setPlaceholderText("X-API-Key")
        self._auth_form.addRow("Key Name:", self._auth_key_name_input)

        self._reveal_btn = QPushButton("Reveal Secrets")
        self._auth_form.addRow("", self._reveal_btn)

        # Set initial visibility
        self._on_auth_type_changed("none")

        return card

    def _create_reliability_card(self) -> QGroupBox:
        """Build the Reliability group box."""
        card = QGroupBox("Reliability")
        form = QFormLayout(card)

        # Timeout + Retries side by side
        row1 = QHBoxLayout()
        self._timeout_spin = QSpinBox()
        self._timeout_spin.setRange(1, 300)
        self._timeout_spin.setValue(30)
        row1.addWidget(QLabel("Timeout (sec):"))
        row1.addWidget(self._timeout_spin)
        row1.addSpacing(16)
        self._retry_spin = QSpinBox()
        self._retry_spin.setRange(0, 10)
        self._retry_spin.setValue(3)
        row1.addWidget(QLabel("Max Retries:"))
        row1.addWidget(self._retry_spin)
        row1.addStretch()
        form.addRow(row1)

        self._backoff_combo = QComboBox()
        self._backoff_combo.addItems(["none", "linear", "exponential"])
        self._backoff_combo.setCurrentText("exponential")
        form.addRow("Backoff:", self._backoff_combo)

        self._retry_status_input = QLineEdit("429, 500, 502, 503")
        self._retry_status_input.setPlaceholderText("429, 500, 502, 503")
        form.addRow("Retry On Status:", self._retry_status_input)

        # RPM + Concurrent side by side
        row2 = QHBoxLayout()
        self._rate_rpm_spin = QSpinBox()
        self._rate_rpm_spin.setRange(0, 9999)
        self._rate_rpm_spin.setValue(0)
        row2.addWidget(QLabel("RPM Limit:"))
        row2.addWidget(self._rate_rpm_spin)
        row2.addSpacing(16)
        self._rate_concurrent_spin = QSpinBox()
        self._rate_concurrent_spin.setRange(0, 9999)
        self._rate_concurrent_spin.setValue(0)
        row2.addWidget(QLabel("Concurrent:"))
        row2.addWidget(self._rate_concurrent_spin)
        row2.addStretch()
        form.addRow(row2)

        return card

    def _create_discovery_card(self) -> QGroupBox:
        """Build the Discovery group box."""
        card = QGroupBox("Discovery")
        form = QFormLayout(card)

        self._purpose_input = QTextEdit()
        self._purpose_input.setMaximumHeight(60)
        form.addRow("Purpose:", self._purpose_input)

        self._keywords_input = QLineEdit()
        self._keywords_input.setPlaceholderText("weather, forecast, temperature")
        form.addRow("Keywords:", self._keywords_input)

        # Category + Tags side by side
        cat_tags = QHBoxLayout()
        self._category_input = QLineEdit()
        cat_tags.addWidget(QLabel("Category:"))
        cat_tags.addWidget(self._category_input)
        cat_tags.addSpacing(16)
        self._tags_input = QLineEdit()
        self._tags_input.setPlaceholderText("tag1, tag2")
        cat_tags.addWidget(QLabel("Tags:"))
        cat_tags.addWidget(self._tags_input)
        form.addRow(cat_tags)

        self._use_when_input = QTextEdit()
        self._use_when_input.setMaximumHeight(60)
        self._use_when_input.setPlaceholderText("One condition per line")
        form.addRow("Use When:", self._use_when_input)

        self._do_not_use_when_input = QTextEdit()
        self._do_not_use_when_input.setMaximumHeight(60)
        self._do_not_use_when_input.setPlaceholderText("One condition per line")
        form.addRow("Don't Use When:", self._do_not_use_when_input)

        return card

    def _create_parameters_card(self) -> QGroupBox:
        """Build the Parameters group box."""
        card = QGroupBox("Parameters")
        card_layout = QVBoxLayout(card)

        self._add_param_btn = QPushButton("+ Add Parameter")
        card_layout.addWidget(self._add_param_btn)

        self._params_container = QVBoxLayout()
        card_layout.addLayout(self._params_container)
        card_layout.addStretch()

        self._param_rows: list = []    # list of dicts holding widget references per row

        return card

    def _create_response_card(self) -> QGroupBox:
        """Build the Response group box."""
        card = QGroupBox("Response")
        form = QFormLayout(card)

        self._success_codes_input = QLineEdit("200")
        self._success_codes_input.setPlaceholderText("200, 201")
        form.addRow("Success Codes:", self._success_codes_input)

        self._parse_as_combo = QComboBox()
        self._parse_as_combo.addItems(["json", "text", "binary"])
        form.addRow("Parse As:", self._parse_as_combo)

        self._output_mapping_input = QTextEdit()
        self._output_mapping_input.setMaximumHeight(80)
        self._output_mapping_input.setPlaceholderText("field_name:$.json.path\nOne mapping per line")
        form.addRow("Output Mapping:", self._output_mapping_input)

        self._error_mapping_input = QTextEdit()
        self._error_mapping_input.setMaximumHeight(80)
        self._error_mapping_input.setPlaceholderText("field_name:$.json.path\nOne mapping per line")
        form.addRow("Error Mapping:", self._error_mapping_input)

        return card

    # ------------------------------------------------------------------
    # Auth Dynamic Visibility
    # ------------------------------------------------------------------

    def _set_auth_field_visible(self, widget, visible: bool):
        """Show/hide a form field and its label."""
        widget.setVisible(visible)
        label = self._auth_form.labelForField(widget)
        if label:
            label.setVisible(visible)

    def _on_auth_type_changed(self, auth_type: str):
        """Show/hide auth fields based on auth_type."""
        is_api_key = auth_type == "api_key"
        is_bearer = auth_type == "bearer"
        is_basic = auth_type == "basic"
        has_secret = is_api_key or is_bearer or is_basic

        self._set_auth_field_visible(self._auth_secret_input, is_api_key or is_bearer)
        self._set_auth_field_visible(self._auth_username_input, is_basic)
        self._set_auth_field_visible(self._auth_password_input, is_basic)
        self._set_auth_field_visible(self._auth_key_location_combo, is_api_key)
        self._set_auth_field_visible(self._auth_key_name_input, is_api_key)
        self._set_auth_field_visible(self._reveal_btn, has_secret)

    # ------------------------------------------------------------------
    # Parameter Rows
    # ------------------------------------------------------------------

    def _add_parameter_row(self, data: dict = None):
        """Create one parameter row and add it to the container."""
        frame = QFrame()
        frame.setFrameShape(QFrame.StyledPanel)
        layout = QVBoxLayout(frame)

        # Row 1: Name, Location, Type, Remove button
        row1 = QHBoxLayout()
        name_input = QLineEdit()
        name_input.setPlaceholderText("param_name")
        location_combo = QComboBox()
        location_combo.addItems(["path", "query", "header", "body"])
        type_combo = QComboBox()
        type_combo.addItems(["string", "integer", "number", "boolean", "array"])
        remove_btn = QPushButton("Remove")

        row1.addWidget(QLabel("Name:"))
        row1.addWidget(name_input)
        row1.addWidget(QLabel("Location:"))
        row1.addWidget(location_combo)
        row1.addWidget(QLabel("Type:"))
        row1.addWidget(type_combo)
        row1.addWidget(remove_btn)
        layout.addLayout(row1)

        # Row 2: Description, Required
        row2 = QHBoxLayout()
        desc_input = QLineEdit()
        desc_input.setPlaceholderText("Parameter description")
        required_check = QCheckBox("Required")
        required_check.setChecked(True)
        row2.addWidget(QLabel("Desc:"))
        row2.addWidget(desc_input, stretch=1)
        row2.addWidget(required_check)
        layout.addLayout(row2)

        # Row 3: Default, Enum values
        row3 = QHBoxLayout()
        default_input = QLineEdit()
        default_input.setPlaceholderText("default value")
        enum_input = QLineEdit()
        enum_input.setPlaceholderText("val1, val2, val3")
        row3.addWidget(QLabel("Default:"))
        row3.addWidget(default_input)
        row3.addWidget(QLabel("Enum:"))
        row3.addWidget(enum_input)
        layout.addLayout(row3)

        # Store widget references
        row_data = {
            "frame": frame,
            "name": name_input,
            "location": location_combo,
            "type": type_combo,
            "description": desc_input,
            "required": required_check,
            "default_value": default_input,
            "enum_values": enum_input,
        }

        # Populate from data if provided (edit mode)
        if data:
            name_input.setText(data.get("name", ""))
            location_combo.setCurrentText(data.get("location", "query"))
            type_combo.setCurrentText(data.get("type", "string"))
            desc_input.setText(data.get("description", ""))
            required_check.setChecked(data.get("required", True))
            default_input.setText(str(data.get("default_value", "") or ""))
            enum_vals = data.get("enum_values")
            if enum_vals:
                enum_input.setText(", ".join(enum_vals))

        # Connect remove button
        remove_btn.clicked.connect(lambda checked, f=frame: self._remove_parameter_row(f))

        self._param_rows.append(row_data)
        self._params_container.addWidget(frame)

    def _remove_parameter_row(self, frame: QFrame):
        """Remove a parameter row from the container."""
        for i, row_data in enumerate(self._param_rows):
            if row_data["frame"] is frame:
                self._param_rows.pop(i)
                self._params_container.removeWidget(frame)
                frame.deleteLater()
                break

    def _clear_parameter_rows(self):
        """Remove all parameter rows."""
        for row_data in self._param_rows:
            frame = row_data["frame"]
            self._params_container.removeWidget(frame)
            frame.deleteLater()
        self._param_rows.clear()

    # ------------------------------------------------------------------
    # Form Data Helpers
    # ------------------------------------------------------------------

    def _parse_comma_list(self, text: str) -> list:
        """Split comma-separated string, strip whitespace, filter empty."""
        return [s.strip() for s in text.split(",") if s.strip()]

    def _parse_int_list(self, text: str) -> list:
        """Parse comma-separated integers, skip non-numeric values."""
        result = []
        for s in text.split(","):
            s = s.strip()
            if s:
                try:
                    result.append(int(s))
                except ValueError:
                    pass
        return result

    def _parse_line_list(self, text: str) -> list:
        """Split by newline, strip, filter empty."""
        return [line.strip() for line in text.split("\n") if line.strip()]

    def _parse_mapping(self, text: str) -> list:
        """Parse 'field_name:$.json.path' lines into list of dicts."""
        result = []
        for line in text.split("\n"):
            line = line.strip()
            if ":" in line:
                parts = line.split(":", 1)
                result.append({
                    "field_name": parts[0].strip(),
                    "json_path": parts[1].strip(),
                })
        return result

    def _collect_form_data(self) -> dict:
        """Read all form widgets into a dict matching the API schema field names."""
        data = {}

        # Basic
        data["name"] = self._name_input.text().strip()
        data["description"] = self._description_input.toPlainText().strip()
        data["base_url"] = self._base_url_input.text().strip()
        data["method"] = self._method_combo.currentText()
        data["path"] = self._path_input.text().strip()
        data["body_content_type"] = self._body_type_combo.currentText()

        # Auth
        data["auth_type"] = self._auth_type_combo.currentText()

        # Only include secret fields if user typed something
        # (empty = user didn't change, don't overwrite encrypted values on update)
        secret = self._auth_secret_input.text().strip()
        if secret:
            data["auth_secret"] = secret
        username = self._auth_username_input.text().strip()
        if username:
            data["auth_username"] = username
        password = self._auth_password_input.text().strip()
        if password:
            data["auth_password"] = password

        auth_type = data["auth_type"]
        if auth_type == "api_key":
            data["auth_key_location"] = self._auth_key_location_combo.currentText()
            data["auth_key_name"] = self._auth_key_name_input.text().strip()

        # Reliability
        data["timeout_seconds"] = self._timeout_spin.value()
        data["retry_max_attempts"] = self._retry_spin.value()
        data["retry_backoff"] = self._backoff_combo.currentText()
        data["retry_on_status"] = self._parse_int_list(self._retry_status_input.text())
        data["rate_limit_rpm"] = self._rate_rpm_spin.value()
        data["rate_limit_concurrent"] = self._rate_concurrent_spin.value()

        # Discovery
        data["purpose"] = self._purpose_input.toPlainText().strip()
        data["keywords"] = self._parse_comma_list(self._keywords_input.text())
        data["category"] = self._category_input.text().strip()
        data["tags"] = self._parse_comma_list(self._tags_input.text())
        data["use_when"] = self._parse_line_list(self._use_when_input.toPlainText())
        data["do_not_use_when"] = self._parse_line_list(self._do_not_use_when_input.toPlainText())

        # Parameters
        data["parameters"] = []
        for row in self._param_rows:
            param = {
                "name": row["name"].text().strip(),
                "location": row["location"].currentText(),
                "type": row["type"].currentText(),
                "description": row["description"].text().strip(),
                "required": row["required"].isChecked(),
            }
            default_val = row["default_value"].text().strip()
            if default_val:
                param["default_value"] = default_val
            enum_text = row["enum_values"].text().strip()
            if enum_text:
                param["enum_values"] = self._parse_comma_list(enum_text)
            data["parameters"].append(param)

        # Response
        data["success_codes"] = self._parse_int_list(self._success_codes_input.text())
        data["parse_as"] = self._parse_as_combo.currentText()
        data["output_mapping"] = self._parse_mapping(self._output_mapping_input.toPlainText())
        data["error_mapping"] = self._parse_mapping(self._error_mapping_input.toPlainText())

        return data

    # ------------------------------------------------------------------
    # Form Population / Clear
    # ------------------------------------------------------------------

    def _populate_form(self, tool: dict):
        """Set all form fields from a tool dict (edit mode)."""
        # Basic
        self._name_input.setText(tool.get("name", ""))
        self._description_input.setPlainText(tool.get("description", ""))
        self._base_url_input.setText(tool.get("base_url", ""))
        self._method_combo.setCurrentText(tool.get("method", "GET"))
        self._path_input.setText(tool.get("path", ""))
        self._body_type_combo.setCurrentText(tool.get("body_content_type", "none"))

        # Auth — setting auth_type triggers _on_auth_type_changed automatically
        self._auth_type_combo.setCurrentText(tool.get("auth_type", "none"))
        # Clear secret fields (they come masked as "***" from API — don't show that)
        self._auth_secret_input.clear()
        self._auth_secret_input.setEchoMode(QLineEdit.Password)
        self._auth_username_input.clear()
        self._auth_password_input.clear()
        self._auth_password_input.setEchoMode(QLineEdit.Password)
        self._auth_key_location_combo.setCurrentText(tool.get("auth_key_location") or "header")
        self._auth_key_name_input.setText(tool.get("auth_key_name") or "")

        # Reliability
        self._timeout_spin.setValue(tool.get("timeout_seconds", 30))
        self._retry_spin.setValue(tool.get("retry_max_attempts", 3))
        self._backoff_combo.setCurrentText(tool.get("retry_backoff", "exponential"))
        self._retry_status_input.setText(
            ", ".join(str(c) for c in tool.get("retry_on_status", [429, 500, 502, 503]))
        )
        self._rate_rpm_spin.setValue(tool.get("rate_limit_rpm", 0))
        self._rate_concurrent_spin.setValue(tool.get("rate_limit_concurrent", 0))

        # Discovery
        self._purpose_input.setPlainText(tool.get("purpose", ""))
        self._keywords_input.setText(", ".join(tool.get("keywords", [])))
        self._category_input.setText(tool.get("category", ""))
        self._tags_input.setText(", ".join(tool.get("tags", [])))
        self._use_when_input.setPlainText("\n".join(tool.get("use_when", [])))
        self._do_not_use_when_input.setPlainText("\n".join(tool.get("do_not_use_when", [])))

        # Parameters — clear existing rows, then add from tool data
        self._clear_parameter_rows()
        for param in tool.get("parameters", []):
            self._add_parameter_row(param)

        # Response
        self._success_codes_input.setText(
            ", ".join(str(c) for c in tool.get("success_codes", [200]))
        )
        self._parse_as_combo.setCurrentText(tool.get("parse_as", "json"))

        output_lines = [
            f"{m['field_name']}:{m['json_path']}"
            for m in tool.get("output_mapping", [])
        ]
        self._output_mapping_input.setPlainText("\n".join(output_lines))

        error_lines = [
            f"{m['field_name']}:{m['json_path']}"
            for m in tool.get("error_mapping", [])
        ]
        self._error_mapping_input.setPlainText("\n".join(error_lines))

    def _clear_form(self):
        """Reset every field to defaults (create mode)."""
        # Basic
        self._name_input.clear()
        self._description_input.clear()
        self._base_url_input.clear()
        self._method_combo.setCurrentIndex(0)      # GET
        self._path_input.clear()
        self._body_type_combo.setCurrentIndex(0)   # none

        # Auth
        self._auth_type_combo.setCurrentText("none")  # triggers visibility update
        self._auth_secret_input.clear()
        self._auth_secret_input.setEchoMode(QLineEdit.Password)
        self._auth_username_input.clear()
        self._auth_password_input.clear()
        self._auth_password_input.setEchoMode(QLineEdit.Password)
        self._auth_key_location_combo.setCurrentIndex(0)
        self._auth_key_name_input.clear()

        # Reliability
        self._timeout_spin.setValue(30)
        self._retry_spin.setValue(3)
        self._backoff_combo.setCurrentText("exponential")
        self._retry_status_input.setText("429, 500, 502, 503")
        self._rate_rpm_spin.setValue(0)
        self._rate_concurrent_spin.setValue(0)

        # Discovery
        self._purpose_input.clear()
        self._keywords_input.clear()
        self._category_input.clear()
        self._tags_input.clear()
        self._use_when_input.clear()
        self._do_not_use_when_input.clear()

        # Parameters
        self._clear_parameter_rows()

        # Response
        self._success_codes_input.setText("200")
        self._parse_as_combo.setCurrentIndex(0)    # json
        self._output_mapping_input.clear()
        self._error_mapping_input.clear()

    # ------------------------------------------------------------------
    # Signal Wiring
    # ------------------------------------------------------------------

    def _connect_signals(self):
        """Connect signals to slots."""
        # List page
        self._new_tool_btn.clicked.connect(self._on_new_tool)

        # Form page buttons
        self._back_btn.clicked.connect(self._on_back_to_list)
        self._cancel_btn.clicked.connect(self._on_back_to_list)
        self._save_btn.clicked.connect(self._on_save)
        self._reveal_btn.clicked.connect(self._on_reveal_secrets)
        self._auth_type_combo.currentTextChanged.connect(self._on_auth_type_changed)
        self._add_param_btn.clicked.connect(lambda: self._add_parameter_row())

    # ------------------------------------------------------------------
    # Slots — List View Actions
    # ------------------------------------------------------------------

    @Slot()
    def _on_new_tool(self):
        """Navigate to form in create mode."""
        self._current_tool_id = None
        self._clear_form()
        self._form_header.setText("Creating New Tool")
        self._internal_stack.setCurrentIndex(1)

    def _on_edit_tool(self, tool_id: str):
        """Find the tool in cache and navigate to form in edit mode."""
        tool = None
        for t in self._tools_cache:
            if t["tool_id"] == tool_id:
                tool = t
                break

        if tool is None:
            return

        self._current_tool_id = tool_id
        self._populate_form(tool)
        self._form_header.setText(f"Editing: {tool['name']}")
        self._internal_stack.setCurrentIndex(1)

    def _on_delete_tool(self, tool_id: str, name: str):
        """Confirm and delete a tool."""
        reply = QMessageBox.question(
            self,
            "Delete Tool",
            f'Delete "{name}"?\nThis cannot be undone.',
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return

        worker = DeleteToolWorker(self._client, tool_id, parent=self)
        worker.tool_deleted.connect(self._on_tool_deleted)
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
        if not data.get("base_url", "").strip():
            errors.append("Base URL")
        if not data.get("path", "").strip():
            errors.append("Path")
        if not data.get("description", "").strip():
            errors.append("Description")
        if not data.get("keywords"):
            errors.append("Keywords (at least one)")

        if errors:
            QMessageBox.warning(
                self,
                "Validation Error",
                "Required fields:\n• " + "\n• ".join(errors),
            )
            return

        if self._current_tool_id is None:
            # Create mode
            worker = CreateToolWorker(self._client, data, parent=self)
            worker.tool_created.connect(self._on_tool_created)
            worker.error_occurred.connect(self._on_worker_error)
            worker.start()
            self._save_worker = worker
        else:
            # Edit mode
            worker = UpdateToolWorker(self._client, self._current_tool_id, data, parent=self)
            worker.tool_updated.connect(self._on_tool_updated)
            worker.error_occurred.connect(self._on_worker_error)
            worker.start()
            self._save_worker = worker

    @Slot()
    def _on_reveal_secrets(self):
        """Fetch and display decrypted secrets (edit mode only)."""
        if self._current_tool_id is None:
            return

        worker = GetSecretsWorker(self._client, self._current_tool_id, parent=self)
        worker.secrets_loaded.connect(self._on_secrets_loaded)
        worker.error_occurred.connect(self._on_worker_error)
        worker.start()
        self._secrets_worker = worker

    # ------------------------------------------------------------------
    # Slots — Worker Callbacks
    # ------------------------------------------------------------------

    @Slot(dict)
    def _on_tools_loaded(self, data: dict):
        """Called after ListToolsWorker succeeds."""
        self._tools_cache = data.get("tools", [])
        self._populate_tool_list(self._tools_cache)
        self._list_worker.safe_delete()
        self._list_worker = None

    @Slot(str)
    def _on_tool_deleted(self, tool_id: str):
        """Called after DeleteToolWorker succeeds."""
        QMessageBox.information(self, "Deleted", "Tool deleted.")
        self._load_tools()                         # refresh list
        self._delete_worker.safe_delete()
        self._delete_worker = None

    @Slot(dict)
    def _on_tool_created(self, tool: dict):
        """Called after CreateToolWorker succeeds."""
        QMessageBox.information(self, "Success", f"Tool '{tool['name']}' created.")
        self._load_tools()
        self._internal_stack.setCurrentIndex(0)     # back to list
        self._save_worker.safe_delete()
        self._save_worker = None

    @Slot(dict)
    def _on_tool_updated(self, tool: dict):
        """Called after UpdateToolWorker succeeds."""
        QMessageBox.information(self, "Success", f"Tool '{tool['name']}' updated.")
        self._load_tools()
        self._internal_stack.setCurrentIndex(0)     # back to list
        self._save_worker.safe_delete()
        self._save_worker = None

    @Slot(dict)
    def _on_secrets_loaded(self, data: dict):
        """Populate secret fields with decrypted values."""
        if data.get("auth_secret"):
            self._auth_secret_input.setText(data["auth_secret"])
            self._auth_secret_input.setEchoMode(QLineEdit.Normal)
        if data.get("auth_username"):
            self._auth_username_input.setText(data["auth_username"])
        if data.get("auth_password"):
            self._auth_password_input.setText(data["auth_password"])
            self._auth_password_input.setEchoMode(QLineEdit.Normal)
        self._secrets_worker.safe_delete()
        self._secrets_worker = None

    @Slot(str)
    def _on_worker_error(self, error: str):
        """Handle worker error."""
        QMessageBox.critical(self, "Error", f"Operation failed:\n{error}")

    def _load_tools(self):
        """Fetch tool list from backend."""
        worker = ListToolsWorker(self._client, parent=self)
        worker.tools_loaded.connect(self._on_tools_loaded)
        worker.error_occurred.connect(self._on_worker_error)
        worker.start()
        self._list_worker = worker
