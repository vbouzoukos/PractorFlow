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

**Errors:**
- `404` - Session not found
- `500` - Internal server error

---

### Agent (`/agent`)

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

---

## SSE Streaming

The chat endpoint uses Server-Sent Events (SSE) for real-time streaming responses.

### Stream Format

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

### Final Event

The stream ends with:
```
data: [DONE]
```

### JavaScript Client Example

```javascript
const eventSource = new EventSource(
  `http://localhost:8000/chat/${sessionId}?message=${encodeURIComponent(message)}`,
  {
    headers: {
      'Authorization': `Bearer ${token}`
    }
  }
);

eventSource.onmessage = (event) => {
  if (event.data === '[DONE]') {
    eventSource.close();
    return;
  }
  
  const chunk = JSON.parse(event.data);
  console.log(chunk.text);
  
  if (chunk.finished) {
    console.log('Finish reason:', chunk.finish_reason);
    console.log('Usage:', chunk.usage);
  }
};

eventSource.onerror = (error) => {
  console.error('SSE Error:', error);
  eventSource.close();
};
```

### Python Client Example

```python
import requests
import json

url = f"http://localhost:8000/chat/{session_id}"
headers = {"Authorization": f"Bearer {token}"}
data = {"message": "What is Python?"}

response = requests.post(url, headers=headers, data=data, stream=True)

for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data = line_str[6:]  # Remove 'data: ' prefix
            if data == '[DONE]':
                break
            chunk = json.loads(data)
            print(chunk['text'], end='', flush=True)
```

---

## Deployment

### Production Configuration

**Environment Variables:**
```env
# API Configuration
AUTH_MODE=oidc
JWT_SECRET_KEY=<generate-strong-random-key>
JWT_EXPIRATION_MINUTES=1440

# OIDC
OIDC_ISSUER_URL=https://your-sso.company.com/realms/production
OIDC_CLIENT_ID=practorflow-production
OIDC_CLIENT_SECRET=<from-your-idp>
OIDC_AUDIENCE=practorflow-api

# Cleanup
CLEANUP_INTERVAL_MINUTES=60

# Logging
LOG_LEVEL=INFO
```

### Running with Uvicorn

```bash
# Basic
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Production with workers
uvicorn api.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info \
  --access-log
```

### Systemd Service

Create `/etc/systemd/system/practorflow-api.service`:

```ini
[Unit]
Description=PractorFlow API Service
After=network.target

[Service]
Type=simple
User=practorflow
WorkingDirectory=/opt/practorflow
Environment="PATH=/opt/practorflow/venv/bin"
ExecStart=/opt/practorflow/venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable practorflow-api
sudo systemctl start practorflow-api
sudo systemctl status practorflow-api
```

### Docker Deployment

**Dockerfile:**
```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY . .
RUN pip install -e . && pip install -e ./src/api

# Expose port
EXPOSE 8000

# Run server
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**docker-compose.yml:**
```yaml
version: '3.8'

services:
  practorflow-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - AUTH_MODE=local
      - JWT_SECRET_KEY=${JWT_SECRET_KEY}
      - APP_SECRET=${APP_SECRET}
    restart: unless-stopped
```

### Reverse Proxy (Nginx)

```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # SSE specific
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400s;
        proxy_http_version 1.1;
        proxy_set_header Connection "";
    }
}
```

---

## Development

### Local Setup

```bash
# Clone repository
git clone https://github.com/vbouzoukos/PractorFlow.git
cd PractorFlow

# Install in development mode
pip install -e .
pip install -e "./src/api[dev]"

# Run with auto-reload
practorflow-api-debug
```

### Running Tests

```bash
cd src/api

# Run all tests
pytest

# Run with coverage
pytest --cov=api --cov-report=html

# Run specific test file
pytest tests/test_auth.py -v
```

### Code Quality

```bash
# Format code
black api/

# Sort imports
isort api/

# Type checking
mypy api/
```

### API Documentation

Access interactive documentation:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## Troubleshooting

### Authentication Issues

**Problem:** `401 Unauthorized` errors

**Solutions:**
- Verify `AUTH_MODE` matches your configuration
- Check token hasn't expired (`JWT_EXPIRATION_MINUTES`)
- Ensure `Authorization: Bearer <token>` header is set
- For OIDC, verify `identity_token` is valid and not expired

---

### Token Generation Failed

**Problem:** Cannot obtain JWT token

**Solutions:**
- **Local Mode**: Verify `APP_SECRET` in request matches config
- **OIDC Mode**: Check `OIDC_ISSUER_URL`, `OIDC_CLIENT_ID`, and `OIDC_CLIENT_SECRET`
- Verify `JWT_SECRET_KEY` is set
- Check logs for detailed error messages

---

### SSE Stream Hangs

**Problem:** Chat streaming stops or times out

**Solutions:**
- Check model is loaded: Look for "Model preloaded successfully" in logs
- Verify sufficient GPU/CPU memory for model
- Increase proxy timeout if using Nginx/Apache
- Check firewall isn't blocking long-lived connections

---

### Session Not Found

**Problem:** `404 Session not found` errors

**Solutions:**
- Verify session was created with `GET /chat` or `GET /agent`
- Check session hasn't been deleted
- Ensure you're using correct `session_id` from creation response
- Session IDs are user-specific - can't access other users' sessions

---

### File Upload Errors

**Problem:** File upload fails or documents not indexed

**Solutions:**
- Check file size doesn't exceed server limits
- Verify file format is supported (PDF, DOCX, TXT, images)
- Ensure ChromaDB is properly configured
- Check logs for document processing errors

---

### Model Loading Issues

**Problem:** API starts but model fails to load

**Solutions:**
- Verify model path in `config/llm/options/model.env`
- Check model file exists and is accessible
- Ensure sufficient VRAM/RAM for model
- Review `LLM_GPU_LAYERS` setting
- Check logs for GGUF compatibility issues

---

### OIDC Integration Issues

**Problem:** OIDC token validation fails

**Solutions:**
- Verify `OIDC_ISSUER_URL` matches provider's issuer exactly
- Check `OIDC_AUDIENCE` matches expected audience claim
- Ensure provider's JWKS endpoint is accessible
- Verify client credentials (`OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET`)
- Check provider-specific requirements (Keycloak, Auth0, etc.)

---

### Background Cleanup Not Running

**Problem:** Orphaned sessions accumulating

**Solutions:**
- Verify `CLEANUP_INTERVAL_MINUTES > 0`
- Check logs for "Cleanup scheduler started" message
- Ensure no exceptions during cleanup execution
- Manual cleanup: Restart API server

---

### High Memory Usage

**Problem:** API consuming excessive memory

**Solutions:**
- Reduce `max_models` in model pool (currently hardcoded to 1)
- Lower `LLM_N_CTX` (context window size)
- Use quantized models (4-bit or 8-bit)
- Increase `CLEANUP_INTERVAL_MINUTES` frequency
- Monitor session and document counts

---

### CORS Errors

**Problem:** Browser blocks requests from frontend

**Current Behavior:** API allows all origins (`allow_origins=["*"]`)

**Solutions for Production:**
- Modify `api/main.py` to restrict origins:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```
