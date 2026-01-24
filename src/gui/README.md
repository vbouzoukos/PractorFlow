# PractorFlow GUI

Desktop chat client for PractorFlow - A native Qt application providing an intuitive interface for interacting with local LLM services through the PractorFlow API.

## Overview

PractorFlow GUI is a Qt-based desktop application built with PySide6 that connects to the PractorFlow API server. It offers a modern chat interface with session management, document handling, and dual operation modes (Chat and Agent).

## Features

- Modern chat interface with Markdown rendering and syntax highlighting
- Dual operation modes: Chat (streaming) and Agent (task execution)
- Document management with drag-and-drop upload
- Session history with search and resume functionality
- Message editing and conversation branching
- Automatic dark/light/auto theme support
- Background threading for responsive UI

## Quick Start

### Installation

```bash
cd src/gui
pip install -e .
```

### Run

```bash
practorflow-gui
```

### First Launch

Configure settings on first launch:
- API URL (default: `http://localhost:8000`)
- Username for authentication
- Theme preference (auto/dark/light)

## Usage

### Basic Workflow

1. Click "Connect" to start a new session
2. Toggle "Agent Mode" for task execution (optional)
3. Type messages and press Ctrl+Enter to send
4. Attach documents via 📎 button or drag-and-drop
5. Browse past sessions in the collapsible sidebar
6. View/delete documents via the documents panel

### Chat Mode

Send messages and receive streaming responses. Attach documents for RAG-based answers.

### Agent Mode

Enable for multi-step task execution with planning and verification.

### Message Editing

Click "Edit" on user messages to modify and resend. The conversation branches from the edited point.

### Session Management

- Resume previous sessions by clicking them in the sidebar
- Search sessions by content using the search box
- Delete sessions via right-click context menu

## Architecture

### Design Overview

The application uses a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────┐
│         Presentation Layer              │
│  (Views, Widgets, UI Components)        │
└──────────────┬──────────────────────────┘
               │ Signals/Slots
┌──────────────▼──────────────────────────┐
│       Coordination Layer                │
│  (Event Handling, State Management)     │
└──────────────┬──────────────────────────┘
               │ Worker Delegation
┌──────────────▼──────────────────────────┐
│       Business Logic Layer              │
│  (Workers, API Clients, Data Models)    │
└──────────────┬──────────────────────────┘
               │ HTTP/SSE
┌──────────────▼──────────────────────────┐
│          Data Layer                     │
│  (API Server, Settings Storage)         │
└─────────────────────────────────────────┘
```

### Key Architectural Patterns

**Model-View-Controller (MVC)**
- Views handle presentation and user input
- Controllers coordinate between views and business logic
- Models represent data structures and API responses

**Observer Pattern**
- Qt signals/slots for loose coupling between components
- Widgets observe state changes and update accordingly
- Workers emit progress updates for UI consumption

**Thread-per-Task**
- Each blocking operation runs in a dedicated worker thread
- Main thread remains responsive for UI rendering
- Workers communicate results via signals

**Repository Pattern**
- API clients abstract communication with backend
- Consistent interface for chat, agent, and session operations
- Data models define request/response contracts

### Threading Model

**Main Thread**: UI rendering and event handling only

**Worker Threads**: All blocking operations
- API communication (HTTP requests, SSE streaming)
- File uploads and downloads
- Session and document management
- All workers use Qt signals to communicate with main thread

This ensures the UI never blocks during network operations.

### Communication Flow

**User Input → Response**
1. User submits message via input widget
2. Widget emits signal to coordinator
3. Coordinator creates worker with message data
4. Worker performs API call in background
5. Worker emits chunks/results via signals
6. UI updates incrementally as data arrives

**Session Loading**
1. User selects session from history panel
2. Panel emits session selection signal
3. Coordinator clears current display
4. Worker loads session history from API
5. Messages populate display on completion

### State Management

**Application State**
- Settings stored in `settings.json` (API URL, username, theme)
- Theme applied on startup and settings change

**Session State**
- Active session ID tracked by coordinator
- Connection status determines UI enabled/disabled state
- Operation mode (chat/agent) toggles execution path

**UI State**
- Global flags coordinate widget behavior during operations
- Panels maintain collapsed/expanded state
- Message display tracks editing mode

### API Integration

Three client types handle server communication:
- Chat client: Streaming conversations, file uploads
- Agent client: Task execution, result polling
- Session client: CRUD operations, history retrieval, document management

All clients accept base URL and username, construct endpoints, and handle HTTP/SSE protocols.

### Rendering Pipeline

**Markdown Processing**
1. Assistant messages received as plain text
2. Markdown library converts to HTML
3. Pygments adds syntax highlighting to code blocks
4. QTextBrowser renders final HTML

**Theme Application**
1. qdarktheme detects system theme (auto mode)
2. Applies consistent styling across all widgets
3. Updates on theme change without restart

## Packaging

### Prerequisites

Install PyInstaller in your environment:

```bash

```

Navigate to the GUI source directory:

```bash
cd src/gui
```

---

### Windows

```powershell
pip install .
pyinstaller --onefile --windowed --name practorflow-gui `
  --hidden-import=PySide6.QtSvg `
  --hidden-import=PySide6.QtXml `
  --hidden-import=httpx `
  --hidden-import=httpx_sse `
  --hidden-import=qdarktheme `
  --hidden-import=platformdirs `
  --collect-data=qdarktheme `
  main.py
```

Output: `dist\practorflow-gui.exe`

---

### Linux

```bash
pip install .
pyinstaller --onefile --windowed --name practorflow-gui \
  --hidden-import=PySide6.QtSvg \
  --hidden-import=PySide6.QtXml \
  --hidden-import=httpx \
  --hidden-import=httpx_sse \
  --hidden-import=qdarktheme \
  --hidden-import=platformdirs \
  --collect-data=qdarktheme \
  main.py
```

Output: `dist/practorflow-gui`

---

### macOS

```bash
pip install .
pyinstaller --onefile --windowed --name practorflow-gui \
  --hidden-import=PySide6.QtSvg \
  --hidden-import=PySide6.QtXml \
  --hidden-import=httpx \
  --hidden-import=httpx_sse \
  --hidden-import=qdarktheme \
  --hidden-import=platformdirs \
  --collect-data=qdarktheme \
  --osx-bundle-identifier=com.practorflow.gui \
  main.py
```

Output: `dist/practorflow-gui.app`

---

### Notes

- Build must be performed on the target OS (PyInstaller does not support cross-compilation)
- The `--collect-data=qdarktheme` flag ensures theme assets are bundled
- Use `--onedir` instead of `--onefile` for faster startup and easier debugging
- Add `--icon=path/to/icon.ico` (Windows) or `--icon=path/to/icon.icns` (macOS) for custom icons