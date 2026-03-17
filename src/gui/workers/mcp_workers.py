"""
MCP workers - Background threads for MCP server CRUD operations.

Handles listing, creating, updating, and deleting servers
in background threads to keep the UI responsive.
"""

from PySide6.QtCore import QThread, Signal

from gui.api.mcp_client import McpClient


class ListServersWorker(QThread):
    """
    Worker thread for listing MCP servers.

    Signals:
        servers_loaded: Emitted with servers envelope dict on success.
        error_occurred: Emitted with error message on failure.
    """

    servers_loaded = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, client: McpClient, parent=None):
        super().__init__(parent)
        self._client = client

    def run(self):
        try:
            result = self._client.list_servers()
            self.servers_loaded.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()


class CreateServerWorker(QThread):
    """
    Worker thread for creating an MCP server.

    Signals:
        server_created: Emitted with created server dict on success.
        error_occurred: Emitted with error message on failure.
    """

    server_created = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, client: McpClient, data: dict, parent=None):
        super().__init__(parent)
        self._client = client
        self._data = data

    def run(self):
        try:
            result = self._client.create_server(self._data)
            self.server_created.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()


class UpdateServerWorker(QThread):
    """
    Worker thread for updating an MCP server.

    Signals:
        server_updated: Emitted with updated server dict on success.
        error_occurred: Emitted with error message on failure.
    """

    server_updated = Signal(dict)
    error_occurred = Signal(str)

    def __init__(self, client: McpClient, server_id: str, data: dict, parent=None):
        super().__init__(parent)
        self._client = client
        self._server_id = server_id
        self._data = data

    def run(self):
        try:
            result = self._client.update_server(self._server_id, self._data)
            self.server_updated.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))

    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()


class DeleteServerWorker(QThread):
    """
    Worker thread for deleting an MCP server.

    Signals:
        server_deleted: Emitted with server_id on success.
        error_occurred: Emitted with error message on failure.
    """

    server_deleted = Signal(str)
    error_occurred = Signal(str)

    def __init__(self, client: McpClient, server_id: str, parent=None):
        super().__init__(parent)
        self._client = client
        self._server_id = server_id

    def run(self):
        try:
            success = self._client.delete_server(self._server_id)
            if success:
                self.server_deleted.emit(self._server_id)
            else:
                self.error_occurred.emit("Failed to delete server")
        except Exception as e:
            self.error_occurred.emit(str(e))

    def safe_delete(self):
        """Safely delete worker - wait if still running."""
        if self.isRunning():
            self.wait(2000)
        self.deleteLater()
