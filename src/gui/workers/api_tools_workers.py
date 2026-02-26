"""
API Tools workers - Background threads for API tool CRUD operations.

Handles listing, creating, updating, deleting tools and retrieving
secrets in background threads to keep the UI responsive.
"""

from PySide6.QtCore import QThread, Signal

from gui.api.api_tools_client import ApiToolsClient


class ListToolsWorker(QThread):
    """
    Worker thread for listing API tools.

    Signals:
        tools_loaded: Emitted with tools envelope dict on success.
        error_occurred: Emitted with error message on failure.
    """

    tools_loaded = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, client: ApiToolsClient, parent=None):
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


class CreateToolWorker(QThread):
    """
    Worker thread for creating an API tool.

    Signals:
        tool_created: Emitted with created tool dict on success.
        error_occurred: Emitted with error message on failure.
    """

    tool_created = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, client: ApiToolsClient, data: dict, parent=None):
        super().__init__(parent)
        self._client = client
        self._data = data

    def run(self):
        try:
            result = self._client.create_tool(self._data)
            self.tool_created.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()


class UpdateToolWorker(QThread):
    """
    Worker thread for updating an API tool.

    Signals:
        tool_updated: Emitted with updated tool dict on success.
        error_occurred: Emitted with error message on failure.
    """

    tool_updated = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, client: ApiToolsClient, tool_id: str, data: dict, parent=None):
        super().__init__(parent)
        self._client = client
        self._tool_id = tool_id
        self._data = data

    def run(self):
        try:
            result = self._client.update_tool(self._tool_id, self._data)
            self.tool_updated.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()


class DeleteToolWorker(QThread):
    """
    Worker thread for deleting an API tool.

    Signals:
        tool_deleted: Emitted with tool_id on success.
        error_occurred: Emitted with error message on failure.
    """

    tool_deleted = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, client: ApiToolsClient, tool_id: str, parent=None):
        super().__init__(parent)
        self._client = client
        self._tool_id = tool_id

    def run(self):
        try:
            success = self._client.delete_tool(self._tool_id)
            if success:
                self.tool_deleted.emit(self._tool_id)
            else:
                self.error_occurred.emit("Failed to delete tool")
        except Exception as e:
            self.error_occurred.emit(str(e))

    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()


class GetSecretsWorker(QThread):
    """
    Worker thread for retrieving API tool secrets.

    Signals:
        secrets_loaded: Emitted with secrets dict on success.
        error_occurred: Emitted with error message on failure.
    """

    secrets_loaded = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, client: ApiToolsClient, tool_id: str, parent=None):
        super().__init__(parent)
        self._client = client
        self._tool_id = tool_id

    def run(self):
        try:
            result = self._client.get_secrets(self._tool_id)
            self.secrets_loaded.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()
