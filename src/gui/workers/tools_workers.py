"""
Tools workers - Background threads for tool listing and preference updates.

Handles listing all tools and updating individual tool preferences
in background threads to keep the UI responsive.
"""

from typing import Optional

from PySide6.QtCore import QThread, Signal

from gui.api.tools_client import ToolsClient


class ListAllToolsWorker(QThread):
    """
    Worker thread for listing all tools.

    Signals:
        tools_loaded: Emitted with tools envelope dict on success.
        error_occurred: Emitted with error message on failure.
    """

    tools_loaded = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, client: ToolsClient, parent=None):
        super().__init__(parent)
        self._client = client

    def run(self):
        try:
            result = self._client.list_tools()
            self.tools_loaded.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()


class UpdateToolPreferenceWorker(QThread):
    """
    Worker thread for updating a single tool preference.

    Signals:
        preference_updated: Emitted with (type, id, enabled) on success.
        error_occurred: Emitted with error message on failure.
    """

    preference_updated = Signal(str, str, bool)
    error_occurred = Signal(str)

    def __init__(
        self,
        client: ToolsClient,
        tool_type: str,
        tool_id: str,
        enabled: bool,
        server_id: Optional[str] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._client = client
        self._tool_type = tool_type
        self._tool_id = tool_id
        self._enabled = enabled
        self._server_id = server_id

    def run(self):
        try:
            self._client.update_preference(
                type=self._tool_type,
                id=self._tool_id,
                enabled=self._enabled,
                server_id=self._server_id,
            )
            self.preference_updated.emit(self._tool_type, self._tool_id, self._enabled)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()
