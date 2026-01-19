# PractorFlow API

FastAPI server for PractorFlow - Private AI Service providing local LLM inference with RAG support, authentication, and session management.

## Overview

PractorFlow API is a production-ready RESTful service that exposes the core PractorFlow library's capabilities through HTTP endpoints. It enables web applications, mobile apps, and microservices to leverage local LLM inference without direct Python integration.

**Key Use Cases:**
- Web-based chat interfaces with streaming responses
- Agent task execution with background processing
- Multi-user session management with isolation
- Document upload and RAG-enabled knowledge retrieval
- Enterprise SSO integration via OIDC

## Features

- **RESTful API** with OpenAPI/Swagger documentation
- **Three Authentication Modes:**
  - Open Mode (no credentials - development/trusted networks)
  - Local Mode (app secret-based authentication)
  - OIDC Mode (enterprise SSO with Keycloak, Auth0, Okta, etc.)
- **JWT-based Security** with configurable token expiration
- **Server-Sent Events (SSE)** streaming for real-time chat responses
- **Multi-file Upload** with automatic knowledge base integration
- **Session Management** with history, search, and document tracking
- **Agent Workflows** with asynchronous task execution and job monitoring
- **Background Cleanup** scheduler for orphaned session maintenance
- **CORS Support** for cross-origin requests

## Quick Start

### Prerequisites

- Python 3.10 or higher
- PractorFlow core library installed
- LLM model configured (see main PractorFlow README)

### Installation

```bash
# Install from the api directory
cd src/api
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Basic Configuration

Create configuration files in `config/api/`:

**`config/api/options/config.env`**
```env
# Authentication Mode: open, local, or oidc
AUTH_MODE=local

# JWT Configuration
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=1440

# Cleanup Scheduler (optional)
CLEANUP_INTERVAL_MINUTES=60
```

**`config/api/secrets/config.env`** (for sensitive values)
```env
# Local Mode
APP_SECRET=your-app-secret

# OIDC Mode (if using)
OIDC_CLIENT_ID=your-client-id
OIDC_CLIENT_SECRET=your-client-secret
```

### Run the Server

```bash
# Development mode with auto-reload
practorflow-api-debug

# Production mode
practorflow-api
```

The API will be available at:
- **API**: http://localhost:8000
- **Swagger UI**: http://localhost:8000/docs
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## Configuration

### API Configuration Files

Configuration is loaded from `config/api/` with two directories:

**`config/api/options/config.env`** - Non-sensitive settings
```env
# Authentication
AUTH_MODE=local                    # open | local | oidc
JWT_ALGORITHM=HS256               # HS256 | RS256
JWT_EXPIRATION_MINUTES=1440       # Token lifetime (default: 24 hours)

# OIDC (if using oidc mode)
OIDC_ISSUER_URL=https://your-oidc-provider.com
OIDC_AUDIENCE=your-audience

# Cleanup Scheduler
CLEANUP_INTERVAL_MINUTES=60       # 0 to disable
```

**`config/api/secrets/config.env`** - Sensitive credentials
```env
# JWT Secret (required for local/oidc modes)
JWT_SECRET_KEY=generate-a-secure-random-key

# Local Mode Authentication
APP_SECRET=your-application-secret

# OIDC Authentication
OIDC_CLIENT_ID=your-oidc-client-id
OIDC_CLIENT_SECRET=your-oidc-client-secret
```

### LLM Configuration

The API uses the core PractorFlow configuration from `config/llm/`:

- `config/llm/options/model.env` - Model settings (model path, context size, GPU layers)
- `config/llm/options/knowledge.env` - RAG/ChromaDB settings
- `config/llm/options/session.env` - Session storage settings
- `config/llm/options/api.env` - API endpoint settings (if using remote models)

See the main [PractorFlow Configuration Guide](../../README.md#configuration) for details.

## Authentication

### Open Mode (No Authentication)

No credentials required - suitable for development or trusted networks.

**Configuration:**
```env
AUTH_MODE=open
```

**Usage:**
```bash
# No token needed
curl http://localhost:8000/chat
```

### Local Mode (App Secret)

Simple shared secret authentication.

**Configuration:**
```env
AUTH_MODE=local
APP_SECRET=my-secure-secret
JWT_SECRET_KEY=your-jwt-secret
JWT_EXPIRATION_MINUTES=1440
```

**Obtain Token:**
```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"app_secret": "my-secure-secret", "username": "user@example.com"}'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

**Use Token:**
```bash
curl http://localhost:8000/chat \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

### OIDC Mode (Enterprise SSO)

Integrate with enterprise identity providers like Keycloak, Auth0, or Okta.

**Configuration:**
```env
AUTH_MODE=oidc
OIDC_ISSUER_URL=https://keycloak.example.com/realms/myrealm
OIDC_CLIENT_ID=practorflow-api
OIDC_CLIENT_SECRET=your-client-secret
OIDC_AUDIENCE=practorflow-api
JWT_SECRET_KEY=your-jwt-secret
JWT_EXPIRATION_MINUTES=1440
```

**Workflow:**
1. User authenticates with OIDC provider (handled by your frontend)
2. Frontend receives `id_token` from provider
3. Exchange `id_token` for PractorFlow JWT:

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"identity_token": "provider-id-token-here"}'
```

**Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### Check Authentication Status

```bash
# Get current user info
curl http://localhost:8000/auth/me \
  -H "Authorization: Bearer <token>"

# Get auth configuration
curl http://localhost:8000/auth/status
```

## Response Modes

### Chat Mode - SSE Streaming

The `/chat/{session_id}` endpoint uses **Server-Sent Events (SSE)** for real-time streaming responses. This provides:
- Immediate feedback as tokens are generated
- Progressive display of responses
- Real-time user experience

**Use Cases:**
- Interactive conversations
- Real-time chat interfaces
- Progressive content generation

### Agent Mode - Async Job Execution

The `/agent/{session_id}/execute` endpoint uses **asynchronous job execution** with polling. This provides:
- Background task processing
- Job status tracking
- Reliable completion handling for long-running tasks

**Use Cases:**
- Document analysis and processing
- Multi-step task workflows
- Complex data transformations
- Tasks requiring verification or approval steps

---

## API Endpoints

### Authentication (`/auth`)

#### `POST /auth/token`
Obtain a JWT access token.

**Request:**
```json
{
  "app_secret": "string",      // Local mode
  "identity_token": "string",  // OIDC mode
  "username": "string"         // Optional for local mode
}
```

**Response (200):**
```json
{
  "access_token": "string",
  "token_type": "bearer",
  "expires_in": 86400
}
```

**Errors:**
- `401` - Authentication failed
- `500` - Server error

---

#### `GET /auth/me`
Get current authenticated user information.

**Response (200):**
```json
{
  "user_id": "user@example.com",
  "is_authenticated": true
}
```

**Errors:**
- `401` - Not authenticated

---

#### `GET /auth/status`
Get authentication configuration details.

**Response (200):**
```json
{
  "mode": "local",
  "requires_authentication": true
}
```

---

### Chat (`/chat`)

The chat endpoint provides **streaming responses** using Server-Sent Events (SSE) for real-time interaction.

#### `GET /chat`
Start a new chat session.

**Response (200):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Session created successfully"
}
```

---

#### `POST /chat/{session_id}`
Send a message and receive streaming response via SSE.

**Request (multipart/form-data):**
- `message` (required): User message text
- `files` (optional): Multiple file uploads

**cURL Example:**
```bash
curl -X POST http://localhost:8000/chat/550e8400-e29b-41d4-a716-446655440000 \
  -H "Authorization: Bearer <token>" \
  -F "message=What is machine learning?" \
  -F "files=@document.pdf"
```

**Response (SSE Stream):**
```
data: {"text": "Machine", "finished": false, "finish_reason": null, "usage": null}

data: {"text": " learning", "finished": false, "finish_reason": null, "usage": null}

data: {"text": " is...", "finished": true, "finish_reason": "stop", "usage": {"prompt_tokens": 20, "completion_tokens": 150}}

data: [DONE]
```

**Stream Format:**

Each SSE event contains JSON data:
```json
{
  "text": "chunk of text",
  "finished": false,
  "finish_reason": null,
  "usage": null
}
```

**Fields:**
- `text`: Text chunk from the LLM
- `finished`: Boolean indicating if generation is complete
- `finish_reason`: Reason for completion (`"stop"`, `"length"`, etc.) when `finished=true`
- `usage`: Token usage statistics (only in final chunk)

**Final Event:**

The stream ends with:
```
data: [DONE]
```

**Errors:**
- `404` - Session not found
- `500` - Internal server error

---

### Agent (`/agent`)

The agent endpoint provides **asynchronous task execution** with job status polling. Unlike chat, agent tasks do **not use streaming**.

#### `GET /agent`
Start a new agent task session.

**Response (200):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "message": "Agent session created successfully"
}
```

---

#### `POST /agent/{session_id}/execute`
Execute an agent task asynchronously with optional file uploads.

**Request (multipart/form-data):**
- `task` (required): Task description
- `files` (optional): Multiple file uploads

**cURL Example:**
```bash
curl -X POST http://localhost:8000/agent/550e8400-e29b-41d4-a716-446655440000/execute \
  -H "Authorization: Bearer <token>" \
  -F "task=Analyze the sales data and create a summary report" \
  -F "files=@sales_q4.xlsx"
```

**Response (200):**
```json
{
  "job_id": "job_abc123",
  "status": "scheduled"
}
```

**Errors:**
- `500` - Failed to schedule task

---

#### `GET /agent/jobs/{job_id}`
Get the status and result of an agent job.

**Response (200):**
```json
{
  "job_id": "job_abc123",
  "status": "completed",
  "result": {
    "success": true,
    "output": "Task completed successfully",
    "artifacts": []
  },
  "error": null
}
```

**Status Values:**
- `scheduled` - Job queued for execution
- `running` - Currently executing
- `completed` - Finished successfully
- `failed` - Execution failed

**Errors:**
- `403` - Access denied (not your job)
- `404` - Job not found

---

### Sessions (`/sessions`)

#### `GET /sessions`
List all sessions for the authenticated user.

**Response (200):**
```json
[
  {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "user": "user@example.com",
    "title": "Machine Learning Discussion",
    "message_count": 12,
    "document_count": 2,
    "created_at": "2025-01-17T10:30:00Z",
    "updated_at": "2025-01-17T11:45:00Z"
  }
]
```

---

#### `GET /sessions/search?term={query}`
Search sessions by content.

**Parameters:**
- `term` (optional): Search query string

**Response (200):**
```json
[
  {
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
    "user": "user@example.com",
    "title": "Python Tutorial",
    "message_count": 8,
    "document_count": 1,
    "created_at": "2025-01-17T09:00:00Z",
    "updated_at": "2025-01-17T09:30:00Z"
  }
]
```

---

#### `GET /sessions/{session_id}/history`
Get complete message history for a session.

**Response (200):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "user": "user@example.com",
  "instructions": "You are a helpful assistant",
  "messages": [
    {
      "id": "msg_001",
      "role": "user",
      "content": "What is Python?",
      "timestamp": "2025-01-17T10:30:00Z"
    },
    {
      "id": "msg_002",
      "role": "assistant",
      "content": "Python is a high-level programming language...",
      "timestamp": "2025-01-17T10:30:15Z"
    }
  ],
  "document_count": 2,
  "created_at": "2025-01-17T10:30:00Z",
  "updated_at": "2025-01-17T11:45:00Z"
}
```

---

#### `DELETE /sessions/{session_id}`
Delete a session and all associated documents.

**Response (200):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "deleted": true,
  "message": "Session deleted successfully"
}
```

**Errors:**
- `403` - Access denied
- `404` - Session not found

---

#### `PUT /sessions/{session_id}/truncate`
Remove messages from a specific index onwards (edit-and-regenerate).

**Request:**
```json
{
  "from_index": 5
}
```

**Response (200):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "truncated_count": 3,
  "message": "Messages truncated successfully"
}
```

---

#### `GET /sessions/{session_id}/documents`
List all documents in a session.

**Response (200):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "documents": [
    {
      "id": "doc_001",
      "filename": "report.pdf",
      "file_type": "pdf"
    },
    {
      "id": "doc_002",
      "filename": "data.xlsx",
      "file_type": "xlsx"
    }
  ],
  "count": 2
}
```

---

#### `DELETE /sessions/{session_id}/documents/{document_id}`
Delete a specific document from the session.

**Response (200):**
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "document_id": "doc_001",
  "deleted": true,
  "message": "Document deleted successfully"
}
```

---

### Health (`/health`)

#### `GET /health`
Health check endpoint.

**Response (200):**
```json
{
  "status": "healthy"
}
```