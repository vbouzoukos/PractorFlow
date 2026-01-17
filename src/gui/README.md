# PractorFlow GUI

Desktop chat client for PractorFlow - A native application built with PySide6 providing an intuitive interface for interacting with local LLM services.

## Overview

PractorFlow GUI is a Qt-based desktop application that connects to the PractorFlow API server, offering a modern chat interface with advanced features like session management, document handling, and dual operation modes (Chat and Agent).

**Key Use Cases:**
- Desktop interface for private AI interactions
- Document-based conversations with RAG support
- Agent task execution with step-by-step verification
- Session history management and search
- Multi-file document uploads

## Features

- **Modern Chat Interface** with Markdown rendering and syntax highlighting
- **Dual Operation Modes:**
  - **Chat Mode**: Real-time streaming conversations
  - **Agent Mode**: Task execution with verification steps
- **Document Management:**
  - Drag-and-drop file upload
  - Support for PDF, DOCX, PPTX, XLSX, images, and more
  - Per-session document tracking and deletion
- **Session Management:**
  - Collapsible history sidebar
  - Session search functionality
  - Resume previous conversations
  - Delete sessions and documents
- **Message Features:**
  - Edit and resend user messages
  - Message truncation for conversation branching
  - Live streaming with token count updates
- **UI/UX:**
  - Automatic dark/light theme support
  - Responsive layout with resizable panels
  - Background worker threads for smooth performance
  - Keyboard shortcuts for common actions

## Quick Start

### Prerequisites

- Python 3.10 or higher
- PractorFlow API server running (see [API Documentation](../api/README.md))
- Qt6 libraries (installed automatically with PySide6)

### Installation

```bash
# Install from the gui directory
cd src/gui
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Configuration

Set the API server URL via environment variable:

```bash
# Linux/macOS
export PRACTORFLOW_API_URL=http://localhost:8000

# Windows (CMD)
set PRACTORFLOW_API_URL=http://localhost:8000

# Windows (PowerShell)
$env:PRACTORFLOW_API_URL="http://localhost:8000"
```

Or create a `.env` file in the project root:

```env
PRACTORFLOW_API_URL=http://localhost:8000
```

### Run the Application

```bash
# Using the installed entry point
practorflow-gui

# Or run directly
cd src/gui
python main.py
```

## Installation Guide

### Option 1: Install as Package

```bash
cd src/gui
pip install -e .
```

This installs:
- `PySide6` - Qt6 bindings for Python
- `markdown` - Markdown to HTML conversion
- `pygments` - Syntax highlighting for code blocks
- `requests` - HTTP client for API communication
- `python-dotenv` - Environment variable management

The GUI automatically depends on the PractorFlow API client interfaces but does not require the core PractorFlow library or API server installation.

### Option 2: Install from Requirements

```bash
cd src/gui
pip install -r requirements.txt
```

### Development Installation

For development with testing tools:

```bash
cd src/gui
pip install -e ".[dev]"
```

This adds:
- `pytest` - Testing framework
- `pytest-qt` - Qt testing utilities
- `black` - Code formatter
- `isort` - Import sorter
- `mypy` - Type checker

## User Interface Guide

### Main Window Layout

```
┌─────────────────────────────────────────────────────────┐
│ [New Session] [🔌 Reconnect] [📄 Documents] [☐ Agent]  │ ← Toolbar
├───────────┬─────────────────────────────────────────────┤
│           │                                             │
│  Session  │                                             │
│  History  │         Chat Display Area                   │ ← Main Area
│           │         (Messages with Markdown)            │
│  (Search) │                                             │
│           │                                             │
├───────────┼─────────────────────────────────────────────┤
│           │  📎 [Attach Files]                          │
│           │  ┌────────────────────────────────────────┐ │
│           │  │ Type your message here...               │ │ ← Input Area
│           │  └────────────────────────────────────────┘ │
│           │                          [Send] or [Ctrl+↵] │
├───────────┴─────────────────────────────────────────────┤
│ Status: API: http://localhost:8000          | Tokens: 0 │ ← Status Bar
└─────────────────────────────────────────────────────────┘
```

### Session History Panel (Left)

- **Collapsible sidebar** - Click the edge to show/hide
- **Search bar** - Filter sessions by content
- **Session list** - Shows recent conversations with:
  - Session title (from first message)
  - Message count
  - Document count
  - Last updated timestamp
- **Click to resume** - Load any previous session
- **Right-click menu** - Delete session option

### Documents Panel (Foldable)

- **Click "Documents" button** to show/hide
- **Document list** shows:
  - Filename
  - File type
  - Delete button per document
- **Auto-refreshes** when files are uploaded

### Chat Display

- **User messages** - Right-aligned, light blue background
- **Assistant messages** - Left-aligned, Markdown rendered
- **System messages** - Center-aligned, italic style
- **Edit feature** - Click "Edit" on user messages to modify and resend
- **Auto-scroll** - Follows latest message during streaming

### Input Widget

- **Text area** - Multi-line input with auto-resize
- **File attachment** - Click 📎 or drag-and-drop files
- **Send button** - Click or press `Ctrl+Enter`
- **Character counter** - Shows input length

## Usage

### Starting a New Session

1. Click **"New Session"** button
2. Wait for confirmation message
3. Start chatting

Or simply type a message - the app creates a session automatically on first message if none exists.

### Chat Mode (Default)

**Basic Conversation:**
```
1. Type your message in the input area
2. Press Ctrl+Enter or click Send
3. Watch the response stream in real-time
4. Token count updates live in status bar
```

**With Documents:**
```
1. Click 📎 or drag files into the input area
2. Selected files appear as chips below input
3. Type your message referencing the documents
4. Click Send
5. LLM uses RAG to answer from document content
```

**Example:**
```
[Upload: sales_report_q4.pdf]
Message: "What were the top 3 products by revenue in Q4?"
```

### Agent Mode

Agent mode uses a multi-step verification workflow for complex tasks.

**Enable Agent Mode:**
1. Check the **"Agent Mode"** checkbox in toolbar
2. Status bar shows "Agent mode enabled"

**Execute Tasks:**
```
1. Type your task description
2. Click Send
3. Agent plans the task breakdown
4. Agent executes each step with verification
5. Final result shown when complete
```

**Example:**
```
Message: "Analyze the sales data and create a summary report with trends"

Agent Response:
[Step 1] Parsing uploaded documents... ✓
[Step 2] Extracting sales metrics... ✓
[Step 3] Calculating trends... ✓
[Step 4] Generating summary report... ✓

Final Summary:
[Report content with trends and insights]
```

**Note:** Agent mode shows a progress message while processing. Execution happens asynchronously - you can continue using the UI.

### Editing Messages

**Edit and Resend:**
1. Find the user message you want to edit
2. Click the **"Edit"** button
3. Message becomes editable
4. Modify the text
5. Click **"Save"** to resend with new content

**What Happens:**
- All messages from the edited point onward are removed
- Conversation branches from the edited message
- New response is generated from edited content
- This enables "what if" exploration of conversations

### Managing Documents

**View Session Documents:**
1. Click **"Documents"** button
2. Panel shows all uploaded files for current session
3. Each file has filename, type, and delete button

**Delete a Document:**
1. Click the delete (trash) icon next to the document
2. Confirm deletion
3. Document removed from knowledge base
4. Future messages won't use this document

### Session Management

**Resume Previous Session:**
1. Open history panel (left sidebar)
2. Browse or search for the session
3. Click to load
4. Full message history appears
5. Continue the conversation

**Search Sessions:**
1. Type search term in history panel search box
2. Sessions filter in real-time
3. Searches through message content

**Delete a Session:**
1. Right-click on session in history panel
2. Select "Delete"
3. Confirm deletion
4. Session and all associated documents removed

### Reconnecting to API

If the API server goes down or restarts:

1. Click **"Reconnect"** button
2. App attempts to resume current session
3. Success: Continue where you left off
4. Failure: Create new session

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Ctrl+Enter` | Send message |
| `Ctrl+N` | New session |
| `Ctrl+O` | Toggle documents panel |
| `Ctrl+H` | Toggle history panel |
| `Ctrl+L` | Focus message input |
| `Ctrl+Q` | Quit application |
| `Esc` | Cancel edit mode |

## Configuration

### Environment Variables

**`PRACTORFLOW_API_URL`**
- API server base URL
- Default: `http://localhost:8000`
- Example: `http://192.168.1.100:8000`

### Runtime Configuration

No additional configuration files needed. The GUI reads all settings from the API server.

## Architecture

### Component Overview

```
gui/
├── main.py                  # Application entry point
├── chat_window.py           # Main window (QMainWindow)
├── logger.py                # Logging configuration
├── widgets/                 # UI components
│   ├── chat_display.py      # Message history display
│   ├── input_widget.py      # Message input area
│   ├── message_widget.py    # Single message display
│   ├── history_panel.py     # Session history sidebar
│   └── documents_panel.py   # Document management panel
├── workers/                 # Background threads
│   ├── stream_worker.py     # Chat streaming handler
│   ├── agent_worker.py      # Agent task handler
│   ├── session_worker.py    # Session management
│   ├── history_worker.py    # History loading
│   └── document_worker.py   # Document operations
└── api/                     # API client layer
    ├── chat_client.py       # Chat endpoint client
    ├── agent_client.py      # Agent endpoint client
    ├── session_client.py    # Session endpoint client
    └── client_data.py       # Data models
```

### Thread Architecture

The GUI uses Qt's threading model to keep the UI responsive:

- **Main Thread** - UI rendering and event handling
- **Worker Threads** - All API calls and network I/O
  - `StreamWorker` - Chat message streaming
  - `AgentTaskWorker` - Agent execution
  - `SessionWorker` - Session creation/deletion
  - `HistoryWorker` - Load session history
  - `DocumentWorker` - Document operations

**Benefits:**
- UI never blocks during network operations
- Streaming responses update UI in real-time
- Multiple operations can run concurrently
- Graceful handling of network failures

### Markdown Rendering

Assistant messages are rendered as HTML using:

1. **Markdown** library - Converts Markdown to HTML
2. **Pygments** - Syntax highlighting for code blocks
3. **QTextBrowser** - Renders HTML in Qt widget

**Supported Markdown:**
- Headers (`#`, `##`, `###`)
- Bold (`**text**`), Italic (`*text*`)
- Code blocks with syntax highlighting
- Inline code (`` `code` ``)
- Lists (ordered and unordered)
- Links (rendered but not clickable by default)
- Tables

### Dark Mode Support

The GUI automatically adapts to the system theme:

- **Light Mode** - Light backgrounds, dark text
- **Dark Mode** - Dark backgrounds, light text
- **Accent Colors** - Derived from system palette
- **Message Bubbles** - Themed for readability

Uses Qt's `QPalette` system for automatic color adaptation.

## Development

### Local Setup

```bash
# Clone repository
git clone https://github.com/vbouzoukos/PractorFlow.git
cd PractorFlow/src/gui

# Install in development mode
pip install -e ".[dev]"

# Run application
python main.py
```

### Running Tests

```bash
cd src/gui

# Run all tests
pytest

# Run with coverage
pytest --cov=gui --cov-report=html

# Run specific test
pytest tests/test_chat_window.py -v
```

### Code Quality

```bash
# Format code
black gui/

# Sort imports
isort gui/

# Type checking
mypy gui/
```

### Creating Widgets

Example custom widget:

```python
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Signal

class MyWidget(QWidget):
    # Define signals
    value_changed = Signal(str)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()
    
    def _setup_ui(self):
        layout = QVBoxLayout(self)
        label = QLabel("Hello World")
        layout.addWidget(label)
    
    def set_value(self, value: str):
        """Update widget value."""
        self.value_changed.emit(value)
```

### Background Workers

Example worker thread:

```python
from PySide6.QtCore import QThread, Signal

class MyWorker(QThread):
    # Signals for communication
    finished = Signal(str)
    error_occurred = Signal(str)
    
    def __init__(self, data: str, parent=None):
        super().__init__(parent)
        self._data = data
    
    def run(self):
        """Execute in background thread."""
        try:
            result = self._do_work()
            self.finished.emit(result)
        except Exception as e:
            self.error_occurred.emit(str(e))
    
    def _do_work(self):
        # Your async operation here
        return f"Processed: {self._data}"
```

## Troubleshooting

### Application Won't Start

**Problem:** GUI crashes on startup

**Solutions:**
- Check PySide6 installation: `python -c "import PySide6; print(PySide6.__version__)"`
- Verify Qt6 libraries are installed
- Try reinstalling: `pip uninstall PySide6 && pip install PySide6`
- Check for conflicting Qt installations (PyQt5/PyQt6)

**Linux-specific:**
```bash
# Install Qt6 dependencies
sudo apt-get install qt6-base-dev libgl1-mesa-dev
```

---

### Connection Refused

**Problem:** Cannot connect to API server

**Solutions:**
- Verify API server is running: `curl http://localhost:8000/health`
- Check `PRACTORFLOW_API_URL` environment variable
- Verify no firewall blocking the connection
- Check API server logs for errors
- Try clicking "Reconnect" button

**Test Connection:**
```bash
# Check API health
curl http://localhost:8000/health

# Should return: {"status": "healthy"}
```

---

### Authentication Failed

**Problem:** 401 Unauthorized errors

**Solutions:**
- Check API server authentication mode in `config/api/options/config.env`
- For local mode: Verify `APP_SECRET` is configured
- For OIDC mode: Ensure identity token is valid
- The GUI uses hardcoded username "practorFlowClient" - ensure this is allowed
- Check API server logs for authentication details

---

### Session Not Loading

**Problem:** Previous sessions don't appear in history

**Solutions:**
- Verify session storage in `data/session/session.json` exists
- Check file permissions on session storage
- API server must be running same session backend
- Try restarting API server
- Check API logs for session loading errors

---

### File Upload Fails

**Problem:** Documents not uploading or processing

**Solutions:**
- Check file size (default limit: varies by API configuration)
- Verify file format is supported: PDF, DOCX, PPTX, XLSX, images
- Check API server logs for document processing errors
- Ensure ChromaDB is running properly
- Verify sufficient disk space for document storage

---

### Streaming Stops or Hangs

**Problem:** Chat response stops mid-stream

**Solutions:**
- Check API server is still running
- Verify model is loaded: Check API startup logs
- Ensure sufficient GPU/CPU memory
- Check network connection stability
- Look for errors in API server logs
- Try shorter messages or smaller context

---

### Dark Mode Issues

**Problem:** Incorrect colors or unreadable text

**Solutions:**
- Restart application to reload system theme
- Check OS theme settings
- Try forcing theme: `QApplication.setStyle("Fusion")`
- Update Qt6: `pip install --upgrade PySide6`

**Windows Dark Mode:**
```bash
# Enable dark mode support
python -m pip install darkdetect
```

---

### Message Editing Not Working

**Problem:** Cannot edit or resend messages

**Solutions:**
- Ensure session is active (check status bar)
- Verify you're editing a user message (not assistant/system)
- Check API connection is stable
- Try creating a new session
- Check browser console for JavaScript errors (if using web view)

---

### High Memory Usage

**Problem:** GUI consuming excessive RAM

**Solutions:**
- Close old sessions with large histories
- Delete unused documents from sessions
- Limit session history display (modify `ChatDisplay.add_message`)
- Clear message history periodically
- Restart application to free memory

---

### Agent Mode Not Responding

**Problem:** Agent tasks never complete

**Solutions:**
- Check API server agent endpoint: `/agent/{session_id}/execute`
- Verify background job processing is enabled
- Check job status with job_id manually via API
- Look for errors in API server logs
- Ensure agent service is initialized properly
- Try shorter, simpler tasks first

---

### Keyboard Shortcuts Not Working

**Problem:** Ctrl+Enter or other shortcuts don't respond

**Solutions:**
- Ensure input widget has focus
- Check for conflicting shortcuts from other apps
- Verify Qt key event handling
- Try default shortcuts on different OS
- Check `chat_window.py` keyPressEvent implementation

---

### Panel Resize Issues

**Problem:** Panels won't resize or stay resized

**Solutions:**
- Use splitter handles to drag panel sizes
- Check `QSplitter` configuration in `chat_window.py`
- Reset layout by restarting application
- Verify no layout conflicts in parent widgets

---

## Platform-Specific Notes

### Windows

- **Dark Mode**: Requires Windows 10 1809+ for proper theme detection
- **Installation**: May require Visual C++ Redistributable
- **Shortcuts**: Use standard `Ctrl+` combinations

### macOS

- **Dark Mode**: Automatically follows system theme
- **Shortcuts**: May use `Cmd+` instead of `Ctrl+` on some macOS versions
- **Retina Display**: High-DPI support automatic

### Linux

- **Dark Mode**: Depends on desktop environment (GNOME, KDE, etc.)
- **Dependencies**: May need to install Qt6 platform plugins
```bash
sudo apt-get install qt6-gtk-platformtheme
```
- **Wayland**: Use `QT_QPA_PLATFORM=wayland` for native Wayland support

## Advanced Usage

### Custom API Endpoints

Connect to a remote API server:

```bash
# Production server
export PRACTORFLOW_API_URL=https://ai.company.com/api

# Development server with custom port
export PRACTORFLOW_API_URL=http://192.168.1.100:9000
```

### Batch Document Upload

Upload multiple documents at once:

1. Click 📎 to open file dialog
2. Select multiple files (Ctrl+Click or Shift+Click)
3. Or drag multiple files into input area
4. All files upload before sending message
5. LLM can reference all documents in response

### Session Export/Import

**Export Session** (Manual):
```bash
# Sessions stored in data/session/session.json
cp data/session/session.json session_backup.json
```

**Import Session** (Manual):
```bash
# Restore from backup
cp session_backup.json data/session/session.json
# Restart API server
```

### Custom Themes

Modify `message_widget.py` to customize colors:

```python
# Change user message background
user_bg = palette.color(QPalette.Highlight)

# Change assistant message background
assistant_bg = palette.color(QPalette.Base)
```

## Performance Tips

1. **Session Cleanup** - Delete old sessions regularly to reduce memory
2. **Document Management** - Remove unused documents from sessions
3. **Network Latency** - Run API server on local network for best performance
4. **Message History** - Long conversations slow rendering; consider truncating
5. **File Sizes** - Large PDFs take time to process; split if possible
