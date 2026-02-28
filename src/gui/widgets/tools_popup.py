"""
Tools popup - Frameless tool selection dialog.

Displays all available tools grouped by type (Built-in, API, MCP)
with collapsible sections and enable/disable checkboxes.
Positioned above the tools button in the input row.
"""

from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QCheckBox,
    QScrollArea,
    QWidget,
    QFrame,
)
from PySide6.QtCore import Qt, Slot

from gui.api.tools_client import ToolsClient
from gui.workers.tools_workers import ListAllToolsWorker, UpdateToolPreferenceWorker
from gui.logger import get_logger

logger = get_logger("practorflow-client", level="INFO", log_file="logs/practorflow-client.log")


class ToolsPopup(QDialog):
    """
    Frameless popup dialog for tool selection.

    Displays tools grouped by type (Built-in, API, MCP) with collapsible
    sections. Each tool can be enabled/disabled via checkbox.
    Toggle calls PUT /tools/preferences immediately via a background worker.
    """

    def __init__(self, client: ToolsClient, anchor_widget: QWidget, parent=None):
        super().__init__(parent, Qt.Popup | Qt.FramelessWindowHint)
        self._client = client
        self._anchor = anchor_widget
        self._workers = []
        self._section_counts = {}   # section_key -> [enabled, total]
        self._section_labels = {}   # section_key -> QLabel (count label)

        self._setup_ui()
        self._load_tools()

    def _setup_ui(self):
        """Initialize the popup UI."""
        self.setFixedWidth(320)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Title bar
        title_bar = QWidget()
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(12, 8, 8, 8)

        title_label = QLabel("Tools")
        title_layout.addWidget(title_label)
        title_layout.addStretch()

        close_btn = QPushButton("✕")
        close_btn.setFixedSize(20, 20)
        close_btn.setFlat(True)
        close_btn.clicked.connect(self.close)
        title_layout.addWidget(close_btn)

        main_layout.addWidget(title_bar)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        main_layout.addWidget(sep)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMaximumHeight(400)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.NoFrame)

        self._content_widget = QWidget()
        self._content_layout = QVBoxLayout(self._content_widget)
        self._content_layout.setContentsMargins(8, 8, 8, 8)
        self._content_layout.setSpacing(4)
        self._content_layout.addStretch()

        scroll.setWidget(self._content_widget)
        main_layout.addWidget(scroll)

    def _load_tools(self):
        """Start loading tools from the API."""
        worker = ListAllToolsWorker(self._client, parent=self)
        worker.tools_loaded.connect(self._on_tools_loaded)
        worker.error_occurred.connect(self._on_load_error)
        worker.finished.connect(worker.safe_delete)
        self._workers.append(worker)
        worker.start()

    @Slot(dict)
    def _on_tools_loaded(self, data: dict):
        """Populate the popup with loaded tools."""
        try:
            # Remove trailing stretch before adding sections
            count = self._content_layout.count()
            if count > 0:
                item = self._content_layout.takeAt(count - 1)
                del item

            builtin = data.get("builtin_tools", [])
            api = data.get("api_tools", [])
            mcp = data.get("mcp_tools", [])

            if builtin:
                self._add_section("builtin", "Built-in", builtin)
            if api:
                self._add_section("api", "API Tools", api)
            if mcp:
                self._add_section("mcp", "MCP Tools", mcp, show_server=True)

            self._content_layout.addStretch()
            self._reposition()

        except Exception as e:
            logger.error(f"ToolsPopup._on_tools_loaded failed: {e}")

    def _add_section(self, key: str, title: str, tools: list, show_server: bool = False):
        """Add a collapsible section for a group of tools."""
        enabled_count = sum(1 for t in tools if t.get("enabled", False))
        total = len(tools)

        self._section_counts[key] = [enabled_count, total]

        # Header row: arrow+title button + count label
        header_widget = QWidget()
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 0)

        header_btn = QPushButton(f"▾ {title}")
        header_btn.setFlat(True)
        header_btn.setStyleSheet("text-align: left; padding: 4px 0;")
        header_layout.addWidget(header_btn)
        header_layout.addStretch()

        count_label = QLabel(f"({enabled_count}/{total})")
        self._section_labels[key] = count_label
        header_layout.addWidget(count_label)

        self._content_layout.addWidget(header_widget)

        # Section content frame
        section_frame = QFrame()
        section_frame.setFrameShape(QFrame.StyledPanel)
        section_layout = QVBoxLayout(section_frame)
        section_layout.setContentsMargins(4, 4, 4, 4)
        section_layout.setSpacing(2)

        for tool in tools:
            self._add_tool_row(section_layout, tool, key, show_server)

        self._content_layout.addWidget(section_frame)

        # Toggle collapse on header click
        def toggle(checked=False, btn=header_btn, frame=section_frame, t=title):
            visible = frame.isVisible()
            frame.setVisible(not visible)
            btn.setText(f"{'▾' if not visible else '▸'} {t}")
            self._reposition()

        header_btn.clicked.connect(toggle)

    def _add_tool_row(self, layout, tool: dict, section_key: str, show_server: bool):
        """Add a single tool row with checkbox and description."""
        row = QWidget()
        row_layout = QVBoxLayout(row)
        row_layout.setContentsMargins(4, 2, 4, 2)
        row_layout.setSpacing(1)

        # Top row: checkbox with tool name (+ server badge for MCP)
        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)

        cb = QCheckBox(tool.get("name", ""))
        cb.setChecked(tool.get("enabled", False))
        top.addWidget(cb)

        if show_server:
            server_name = tool.get("server_name", "")
            if server_name:
                badge = QLabel(f"[{server_name}]")
                badge.setStyleSheet("font-size: 10px; color: gray;")
                top.addWidget(badge)

        top.addStretch()
        row_layout.addLayout(top)

        # Description label
        desc = tool.get("description", "")
        if desc:
            desc_label = QLabel(desc)
            desc_label.setWordWrap(True)
            desc_label.setStyleSheet("font-size: 11px; color: gray; padding-left: 20px;")
            row_layout.addWidget(desc_label)

        layout.addWidget(row)

        # Connect checkbox toggle
        tool_type = tool.get("type", section_key)
        tool_id = tool.get("id", "")
        server_id = tool.get("server_id", None)

        def on_toggled(checked, t=tool_type, i=tool_id, s=server_id, k=section_key):
            self._update_preference(t, i, checked, s, k)

        cb.toggled.connect(on_toggled)

    def _update_preference(
        self,
        tool_type: str,
        tool_id: str,
        enabled: bool,
        server_id,
        section_key: str,
    ):
        """Fire off a worker to update a tool preference."""
        worker = UpdateToolPreferenceWorker(
            client=self._client,
            tool_type=tool_type,
            tool_id=tool_id,
            enabled=enabled,
            server_id=server_id,
            parent=self,
        )

        def on_updated(t, i, e, k=section_key):
            self._update_section_count(k, e)

        worker.preference_updated.connect(on_updated)
        worker.error_occurred.connect(self._on_update_error)
        worker.finished.connect(worker.safe_delete)
        self._workers.append(worker)
        worker.start()

    def _update_section_count(self, section_key: str, enabled: bool):
        """Increment or decrement the enabled count label for a section."""
        if section_key not in self._section_counts:
            return
        counts = self._section_counts[section_key]
        delta = 1 if enabled else -1
        counts[0] = max(0, min(counts[0] + delta, counts[1]))
        if section_key in self._section_labels:
            self._section_labels[section_key].setText(f"({counts[0]}/{counts[1]})")

    @Slot(str)
    def _on_load_error(self, error: str):
        logger.error(f"ToolsPopup load error: {error}")

    @Slot(str)
    def _on_update_error(self, error: str):
        logger.error(f"ToolsPopup update error: {error}")

    def _reposition(self):
        """Reposition the popup above the anchor widget."""
        self.adjustSize()
        anchor_global = self._anchor.mapToGlobal(self._anchor.rect().topLeft())
        x = anchor_global.x()
        y = anchor_global.y() - self.height()
        self.move(x, y)

    def showEvent(self, event):
        super().showEvent(event)
        self._reposition()
