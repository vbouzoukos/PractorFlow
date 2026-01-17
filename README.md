# PractorFlow

**Private, Self-Hosted AI for Organizations That Can't Share Their Data**

PractorFlow is an open-source AI service that organizations can deploy inside their own infrastructure with full control. Built for regulated industries, consulting firms, and enterprises handling confidential data—where privacy is not a feature, but a requirement.

## 🎯 The Problem

Many companies want the productivity gains of tools like ChatGPT or Claude but legally, contractually, or ethically cannot send sensitive data to third-party AI APIs. For AI to be adopted responsibly in these environments, it must run where the data already lives.

## 💡 The Solution

PractorFlow is a production-ready, self-hosted AI service designed for real business workflows. Think of it as a private internal AI assistant that never leaves your environment.

### What This Enables

- **AI assistants that never leave your environment** - All inference and data processing happens on your infrastructure
- **Full ownership of data, prompts, and outputs** - Complete control over your AI interactions
- **Support for agentic workflows** - AI that reasons across documents and tasks
- **No vendor lock-in** - Built on open standards and open-source models

## 🌟 Key Features

- **Multiple Backend Support**: Run GGUF models via llama.cpp or HuggingFace models via transformers
- **RAG with ChromaDB**: Persistent document storage with Small-to-Big chunking strategy for secure document ingestion and retrieval
- **Async Model Pooling**: Efficient resource management with LRU eviction for production deployments
- **Session Management**: Persistent conversation history and document context management
- **Pydantic AI Integration**: Drop-in compatibility with modern AI orchestration and agent frameworks
- **Web Search Tools**: Built-in DuckDuckGo (free) and SerpAPI (premium) support
- **Native Function Calling**: Automatic detection and support for models with native tool calling
- **Streaming Generation**: Real-time token streaming with async support
- **Document Processing**: Support for PDF, DOCX, PPTX, XLSX, images, and more via Docling
- **Comprehensive Logging**: Configurable logging levels per component for production monitoring

## 📋 Requirements

- Python 3.10+
- CUDA-capable GPU (optional, for GPU acceleration)

## 🚀 Installation

### Core Library (practorflow)

#### Option 1: Install as Package

```bash
# Clone the repository
git clone https://github.com/vbouzoukos/PractorFlow.git
cd PractorFlow

# Install the package
pip install -e .
```

This installs all dependencies defined in `pyproject.toml` and makes the `practorflow` package available system-wide.

#### Option 2: Install from Requirements

```bash
# Clone the repository
git clone https://github.com/vbouzoukos/PractorFlow.git
cd PractorFlow

# Install dependencies
pip install -r requirements.txt
```

The requirements.txt includes llama-cpp-python with CUDA 12.1 support. If you need a different CUDA version or CPU-only installation, modify the llama-cpp-python line accordingly.

#### Development Installation

For development with additional tools (pytest, black, mypy, etc.):

```bash
CMAKE_ARGS="-DGGML_OPENMP=OFF" pip install -e ".[dev]"
```

### API Server (practorflow-api)

Install the FastAPI server as a separate package:

```bash
pip install -e ./src/api
```

This automatically installs the core `practorflow` library as a dependency.

#### Development Installation

For development with additional tools (pytest, black, mypy, etc.):

```bash
pip install -e ".[dev]" ./src/api
```

### Enabling Qwen3 and Mistral3 Support

PractorFlow supports the latest Qwen3VL and Mistral3 models, which require a specialized build of llama-cpp-python.

#### Why Special Support is Needed

- The official llama-cpp-python releases do not yet include support for Qwen3VL and Mistral3 architectures
- These newer models use updated attention mechanisms and vision encoders
- Attempting to load these models with standard llama-cpp-python will result in unsupported architecture errors

#### Installation Options

**Option 1: Prebuilt Wheels (Recommended)**

For Python 3.12 (CUDA 12.8, Linux/WSL2):

```bash
# Edit requirements.txt - comment out the standard llama-cpp-python line:
# llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Uncomment this line:
llama-cpp-python @ https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.18-cu128-AVX2-linux-20251220/llama_cpp_python-0.3.18-cp312-cp312-linux_x86_64.whl

# Then install:
pip install -r requirements.txt
```

For Python 3.10 (CUDA 12.8, Linux/WSL2):

```bash
# Use the Python 3.10 wheel instead:
llama-cpp-python @ https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.18-cu128-AVX2-linux-20251220/llama_cpp_python-0.3.18-cp310-cp310-linux_x86_64.whl
```

**Option 2: Prebuilt Wheels with pyproject.toml**

The standard `pip install .` installs llama-cpp-python from PyPI, which does **NOT** support Qwen3VL and Mistral3.

You need to remove llama-cpp-python from pyproject.toml and install it separately:

```bash
# Step 1: Clone the repository
git clone https://github.com/vbouzoukos/PractorFlow.git
cd PractorFlow

# Step 2: Edit pyproject.toml - remove "llama-cpp-python" from the dependencies list

# Step 3: Install custom llama-cpp-python

# For Python 3.12 (CUDA 12.8, Linux/WSL2):
pip install https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.18-cu128-AVX2-linux-20251220/llama_cpp_python-0.3.18-cp312-cp312-linux_x86_64.whl

# For Python 3.10 (CUDA 12.8, Linux/WSL2):
pip install https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.18-cu128-AVX2-linux-20251220/llama_cpp_python-0.3.18-cp310-cp310-linux_x86_64.whl

# Step 4: Install PractorFlow
pip install .
```

**Build from Source with pyproject.toml:**

```bash
# Steps 1-2: Same as above (clone, remove "llama-cpp-python" from pyproject.toml)

# Step 3: Build and install custom llama-cpp-python (replace '89' with your GPU compute capability)
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=$(which nvcc)" \
pip install git+https://github.com/JamePeng/llama-cpp-python.git --no-cache-dir

# Step 4: Install PractorFlow
pip install .
```

**If you already ran `pip install .` without Qwen3/Mistral3 support:**

```bash
pip uninstall llama-cpp-python
pip install https://github.com/JamePeng/llama-cpp-python/releases/download/v0.3.18-cu128-AVX2-linux-20251220/llama_cpp_python-0.3.18-cp312-cp312-linux_x86_64.whl
```

**Option 3: Build from Source (Advanced)**

If prebuilt wheels don't work for your system:

1. Check your CUDA version:
```bash
nvcc --version
```

2. Determine your GPU compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

3. Build and install (replace '89' with your GPU's compute capability):
```bash
CMAKE_ARGS="-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CUDA_COMPILER=$(which nvcc)" \
pip install git+https://github.com/JamePeng/llama-cpp-python.git --no-cache-dir
```

**Common GPU Compute Capabilities:**
- RTX 4090/4080/4070/4060: 89
- RTX 3090/3080/3070/3060: 86
- RTX 2080 Ti/2080: 75
- GTX 1080 Ti/1080: 61

#### Verification

After installation, verify the custom build:

```bash
# Check llama-cpp-python version
python -c "import llama_cpp; print(f'llama-cpp-python version: {llama_cpp.__version__}')"

# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Test model loading (replace with your model path)
python -c "
from llama_cpp import Llama
model = Llama(model_path='path/to/Qwen3VL-8B-Instruct-Q4_K_M.gguf', n_gpu_layers=-1)
print('Model loaded successfully!')
"
```

#### Troubleshooting

**Error: "Unsupported architecture"**
- Solution: Ensure you're using the custom llama-cpp-python wheel, not the standard version

**Error: "CUDA out of memory"**
- Reduce context window size (n_ctx) in your model configuration
- Reduce GPU layers (n_gpu_layers) or use a smaller quantization
- Close other GPU-intensive applications

**Error: "Model loading takes too long"**
- First load is always slower due to model download
- Subsequent loads use cached models from `models` directory
- Consider using `pool.preload()` at server startup

**Error: "ImportError: cannot import name 'Llama'"**
- Reinstall llama-cpp-python:
```bash
pip uninstall llama-cpp-python
pip install --no-cache-dir <wheel-url-or-build-command>
```

**Error: "CMAKE_CUDA_COMPILER not found"**
- Install CUDA toolkit:
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit

# Verify installation
nvcc --version
```

#### Performance Tips

1. **Quantization Selection:**
   - Q4_K_M: Best balance of quality/speed (recommended)
   - Q5_K_M: Slightly better quality, slower
   - Q3_K_M: Faster, lower quality (good for testing)

2. **Context Window:**
   - Use smaller contexts (8192-16384) if you don't need full 32K
   - Larger contexts consume more VRAM and slow inference

3. **Batch Size:**
   - Increase n_batch for faster prompt processing
   - Decrease if you hit VRAM limits

4. **GPU Layers:**
   - Start with -1 (all layers on GPU)
   - If VRAM limited, try 30-40 layers
   - Monitor with `nvidia-smi` to find optimal balance

#### Model Sources

**Qwen3VL Models:**
- [Qwen/Qwen3-VL-8B-Instruct-GGUF](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF)
- Multimodal (text + vision) capabilities
- 8B parameters, fits on 24GB GPU with Q4 quantization

**Mistral3 Models:**
- [mistralai/Ministral-3-8B-Instruct-2512-GGUF](https://huggingface.co/mistralai/Ministral-3-8B-Instruct-2512-GGUF)
- Text-only, improved architecture over Mistral 7B
- 8B parameters, excellent quality/efficiency ratio

#### Switching Back to Standard Models

To revert to standard llama-cpp-python for other models:

**For requirements.txt users:**

```bash
# Edit requirements.txt - comment out custom wheel
# Uncomment standard version:
llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Reinstall:
pip install --force-reinstall -r requirements.txt
```

**For pyproject.toml users:**

```bash
# Uninstall custom wheel
pip uninstall llama-cpp-python

# Install standard version FIRST
pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121

# Then reinstall PractorFlow (pip will skip llama-cpp-python since it's already installed)
pip install .
```

#### Additional Resources

- [llama.cpp GitHub](https://github.com/ggerganov/llama.cpp)
- [Custom llama-cpp-python builds](https://github.com/JamePeng/llama-cpp-python)
- [GGUF Model Hub](https://huggingface.co/models?search=gguf)
- [Qwen Documentation](https://qwen.readthedocs.io/)
- [Mistral Documentation](https://docs.mistral.ai/)

**Note:** The custom llama-cpp-python builds are community-maintained until official support is merged. Always verify the source and use official releases when available for production deployments.

## ⚙️ Configuration

PractorFlow uses environment variables for configuration, managed by `app_settings.py`. Configuration is loaded from three `.env` files in the `config/llm/options/` directory.

### Loading Configuration

Configuration is automatically loaded when you import the module. You can also specify a custom config path:

```python
from practorflow.settings.app_settings import load_configuration, appConfiguration

# Load from default path (config/llm/options/)
load_configuration()

# Or load from custom path
load_configuration(config_path="/path/to/your/config")

# Access configuration
config = appConfiguration.ModelConfiguration
```

### Configuration Files

#### 1. Logger Configuration (`config/llm/options/logger.env`)

Controls logging verbosity for different components. Useful for debugging specific parts of the system or reducing log noise in production.

| Variable | Description | Valid Values | Default |
|----------|-------------|--------------|---------|
| `LOG_RUNNER_LEVEL` | LLM runner inference logging (generation, streaming) | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |
| `LOG_DOC_LEVEL` | Document processing logging (parsing, chunking) | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |
| `LOG_KNOWLEDGE_LEVEL` | Knowledge store operations (indexing, search) | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |
| `LOG_MODEL_POOL_LEVEL` | Model pool lifecycle (loading, caching, eviction) | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |
| `LOG_TOOL_LEVEL` | Tool execution logging (search, web tools) | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |
| `LOG_AGENT_LEVEL` | Pydantic AI agent logging (tool calls, responses) | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` | `INFO` |

**Example:**

```bash
# Production: minimal logging
LOG_RUNNER_LEVEL=WARNING
LOG_DOC_LEVEL=INFO
LOG_KNOWLEDGE_LEVEL=INFO
LOG_MODEL_POOL_LEVEL=INFO
LOG_TOOL_LEVEL=WARNING
LOG_AGENT_LEVEL=INFO

# Development: verbose logging for debugging
LOG_RUNNER_LEVEL=DEBUG
LOG_DOC_LEVEL=DEBUG
LOG_KNOWLEDGE_LEVEL=DEBUG
LOG_MODEL_POOL_LEVEL=DEBUG
LOG_TOOL_LEVEL=DEBUG
LOG_AGENT_LEVEL=DEBUG
```

---

#### 2. Model Configuration (`config/llm/options/model.env`)

Configures the LLM model, backend, and inference parameters.

##### Model Selection

| Variable | Description | Valid Values | Default |
|----------|-------------|--------------|---------|
| `LLM_MODEL` | Model identifier. For GGUF: `repo_id/filename.gguf`. For transformers: HuggingFace model name | Any valid model path or HF model ID | `Qwen/Qwen2-1.5B-Instruct-GGUF/qwen2-1_5b-instruct-q4_k_m.gguf` |
| `LLM_BACKEND` | Inference backend to use | `llama_cpp` (for GGUF models), `transformers` (for HF models) | `llama_cpp` |
| `LLM_DEVICE` | Device for model execution | `auto` (auto-detect), `cuda`, `cpu`, `cuda:0`, `cuda:1`, etc. | `auto` |
| `LLM_DTYPE` | Data type for model weights (transformers only) | `auto`, `float32`, `float16`, `bfloat16` | `auto` |
| `LLM_MODELS_DIR` | Directory where models are downloaded and cached | Any valid path | `../models` |

##### Generation Parameters

| Variable | Description | Valid Values | Default |
|----------|-------------|--------------|---------|
| `LLM_MAX_NEW_TOKENS` | Maximum number of tokens to generate per response | `1` - `32768` (depends on model) | `2048` |
| `LLM_TEMPERATURE` | Controls randomness. Lower = more deterministic, higher = more creative | `0.0` - `2.0` | `0.7` |
| `LLM_TOP_P` | Nucleus sampling: only consider tokens with cumulative probability ≤ top_p | `0.0` - `1.0` | `0.9` |
| `LLM_STOP_TOKENS` | Comma-separated list of tokens that stop generation | e.g., `</s>,<\|endoftext\|>` | None |
| `LLM_MAX_SEARCH_RESULTS` | Default number of results for knowledge search tool | `1` - `20` | `5` |

##### Llama.cpp Backend Settings

These settings only apply when `LLM_BACKEND=llama_cpp`:

| Variable | Description | Valid Values | Default |
|----------|-------------|--------------|---------|
| `LLM_GPU_LAYERS` | Number of model layers to offload to GPU. `-1` = all layers (maximum GPU usage) | `-1` to number of model layers | `-1` |
| `LLM_N_CTX` | Context window size in tokens. Larger = more memory, can handle longer conversations | `512` - `131072` (model dependent) | `32768` |
| `LLM_N_BATCH` | Batch size for prompt processing. Larger = faster prompt processing, more VRAM | `1` - `n_ctx` | `2048` |

##### Transformers Backend Settings

These settings only apply when `LLM_BACKEND=transformers`:

| Variable | Description | Valid Values | Default |
|----------|-------------|--------------|---------|
| `LLM_QUANTIZATION` | Enable quantization to reduce memory usage | `4bit`, `8bit`, or unset for none | None |
| `LLM_USE_TORCH_COMPILE` | Enable torch.compile() for faster inference (PyTorch 2.0+) | `true`, `false` | `true` |
| `LLM_COMPILE_MODE` | Optimization mode for torch.compile() | `default` (fast compile), `reduce-overhead` (balanced), `max-autotune` (slowest compile, fastest runtime) | `reduce-overhead` |
| `LLM_WARMUP_ON_LOAD` | Run warmup inference after loading to trigger JIT compilation | `true`, `false` | `true` |

**Example - GGUF model with llama.cpp:**

In this example we use as working directory the src folder

```bash
cd src
```

```bash
LLM_MODEL=bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf
LLM_BACKEND=llama_cpp
LLM_DEVICE=auto
LLM_MODELS_DIR=../models
LLM_MAX_NEW_TOKENS=2048
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9
LLM_GPU_LAYERS=-1
LLM_N_CTX=32768
LLM_N_BATCH=2048
LLM_MAX_SEARCH_RESULTS=5
```

**Example - HuggingFace model with transformers:**

In this example we use as working directory the src folder

```bash
cd src
```

```bash
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
LLM_BACKEND=transformers
LLM_DEVICE=auto
LLM_DTYPE=auto
LLM_MODELS_DIR=../models
LLM_MAX_NEW_TOKENS=2048
LLM_TEMPERATURE=0.7
LLM_TOP_P=0.9
LLM_QUANTIZATION=4bit
LLM_USE_TORCH_COMPILE=true
LLM_COMPILE_MODE=reduce-overhead
LLM_WARMUP_ON_LOAD=true
```

---

#### 3. Knowledge Database Configuration (`config/llm/options/knowledge.env`)

Configures the ChromaDB-based knowledge store for RAG (Retrieval-Augmented Generation).

##### Storage Settings

| Variable | Description | Valid Values | Default |
|----------|-------------|--------------|---------|
| `KB_TYPE` | Knowledge store backend type | `chromadb` | `chromadb` |
| `KB_CHROMA_PERSIST_DIRECTORY` | Directory where ChromaDB stores its data persistently | Any valid path | `../chroma_db` |
| `KB_CHROMA_BATCH_SIZE` | Number of chunks to process in a single batch operation | `1` - `1000` | `100` |

##### Collection Names

ChromaDB uses separate collections for different purposes:

| Variable | Description | Default |
|----------|-------------|---------|
| `KB_CHROMA_RETRIEVE_COLLECTION` | Collection for small retrieval chunks (used for similarity search) | `knowledge_retrieval` |
| `KB_CHROMA_CONTEXT_COLLECTION` | Collection for large context chunks (returned to LLM) | `knowledge_context` |
| `KB_CHROMA_DOCUMENT_COLLECTION` | Collection for document metadata and full content | `knowledge_documents` |

##### Embedding Model

| Variable | Description | Valid Values | Default |
|----------|-------------|--------------|---------|
| `KB_CHROMA_EMBEDDING_MODEL` | SentenceTransformer model for generating embeddings | Any model from [sentence-transformers](https://www.sbert.net/docs/pretrained_models.html) | `all-MiniLM-L6-v2` |
| `KB_CHROMA_EMBEDDING_MODEL_DIR` | Directory to cache embedding model files | Any valid path | `../models` |

**Popular embedding models:**
- `all-MiniLM-L6-v2` - Fast, 384 dimensions, good quality (recommended)
- `all-mpnet-base-v2` - Slower, 768 dimensions, better quality
- `bge-small-en-v1.5` - Fast, 384 dimensions, excellent quality

##### Small-to-Big Chunking Strategy

PractorFlow uses a two-tier chunking approach for optimal RAG performance:

| Variable | Description | Valid Values | Default |
|----------|-------------|--------------|---------|
| `KB_CHROMA_RETRIEVAL_CHUNK_SIZE` | Size of small chunks (in characters) used for embedding similarity search. Smaller = more precise matching | `50` - `500` | `128` |
| `KB_CHROMA_RETRIEVAL_CHUNK_OVERLAP` | Overlap between retrieval chunks to avoid cutting sentences | `0` - `chunk_size/2` | `20` |
| `KB_CHROMA_CONTEXT_CHUNK_SIZE` | Size of large parent chunks (in characters) returned to the LLM. Larger = more context | `256` - `4096` | `1024` |
| `KB_CHROMA_CONTEXT_CHUNK_OVERLAP` | Overlap between context chunks | `0` - `chunk_size/2` | `100` |

**How Small-to-Big works:**
1. Documents are split into large "context chunks" (1024 chars by default)
2. Each context chunk is further split into small "retrieval chunks" (128 chars)
3. Retrieval chunks are embedded and searched for similarity
4. When a match is found, the parent context chunk is returned to the LLM
5. This provides precise search with rich context

**Example:**

In this example we use as working directory the src folder

```bash
cd src
```

```bash
# Knowledge store type
KB_TYPE=chromadb

# Storage location (relative to src/ directory)
KB_CHROMA_PERSIST_DIRECTORY=../chroma_db

# Collection names
KB_CHROMA_RETRIEVE_COLLECTION=knowledge_retrieval
KB_CHROMA_CONTEXT_COLLECTION=knowledge_context
KB_CHROMA_DOCUMENT_COLLECTION=knowledge_documents

# Performance
KB_CHROMA_BATCH_SIZE=100

# Embedding model
KB_CHROMA_EMBEDDING_MODEL=all-MiniLM-L6-v2
KB_CHROMA_EMBEDDING_MODEL_DIR=../models

# Chunking strategy
KB_CHROMA_RETRIEVAL_CHUNK_SIZE=128
KB_CHROMA_RETRIEVAL_CHUNK_OVERLAP=20
KB_CHROMA_CONTEXT_CHUNK_SIZE=1024
KB_CHROMA_CONTEXT_CHUNK_OVERLAP=100
```

#### 4. Session Storage Configuration (`session.env`)

The application supports persistent session storage via TinyDB. Configure the following environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `STORE_SESSION` | Storage backend type. Use `LOCAL` for TinyDB file-based persistence or `MEMORY` for in-memory storage. | `LOCAL` |
| `STORE_SESSION_DB_PATH` | File path for the TinyDB database when using `LOCAL` storage. | `data/session/session.json` |

**Example configuration:**
In this example we use as working directory the src folder

```bash
cd src
```

```env
STORE_SESSION=LOCAL
STORE_SESSION_DB_PATH=../data/session/session.json
```

#### 5. Path Notes:

Note the following parameters should be relative to execution path

```
LLM_MODELS_DIR
KB_CHROMA_PERSIST_DIRECTORY
KB_CHROMA_EMBEDDING_MODEL_DIR
STORE_SESSION_DB_PATH
```

---

### Path Configuration for Different Execution Modes

The configuration examples above assume execution from the `src/` directory, which is why paths use the `../` prefix. Depending on how you install and run PractorFlow, you'll need different path configurations:

#### Running from `src/` Directory (Development Mode)

When running examples or scripts from the `src/` directory:

```bash
cd src
python examples/chat.py
```

**Configuration paths** (with `../` to reach project root):
```bash
LLM_MODELS_DIR=../models
KB_CHROMA_PERSIST_DIRECTORY=../chroma_db
KB_CHROMA_EMBEDDING_MODEL_DIR=../models
STORE_SESSION_DB_PATH=../data/session/session.json
```

#### Installed via pyproject.toml (Production Mode)

When you install PractorFlow using `pip install .` and run from project root or any directory:

```bash
# From project root
python your_script.py
```

**Configuration paths** (relative to project root, no `../`):
```bash
LLM_MODELS_DIR=models
KB_CHROMA_PERSIST_DIRECTORY=chroma_db
KB_CHROMA_EMBEDDING_MODEL_DIR=models
STORE_SESSION_DB_PATH=data/session/session.json
```

**Best Practice:** Maintain separate configuration files for development and production, or use absolute paths to avoid confusion between execution contexts.

---

### Sample Model Configurations

Sample configurations for various models are provided in `config/llm/samples/models/`. Copy these to your `config/llm/options` directory as `model.env`:

| File | Model | Backend | Notes |
|------|-------|---------|-------|
| `model-Qwen2-1.5B-Instruct-GGUF.env` | Qwen2-1.5B | llama_cpp | Small, fast, for testing |
| `model-Qwen2.5-7B.env` | Qwen2.5-7B | transformers | Balanced quality/performance |
| `model-Qwen2.5-7B-Instruct-GGUF.env` | Qwen2.5-7B | llama_cpp | GGUF version |
| `model-Qwen3-VL-8B-Instruct-GGUF.env` | Qwen3-VL-8B | llama_cpp | Multimodal (requires custom build) |
| `model-Ministral-3-8B-Instruct-2512-GGUF.env` | Ministral-3-8B | llama_cpp | Latest Mistral (requires custom build) |
| `model-gpt-oss-20b.env` | GPT-OSS-20B | transformers | Large model, requires significant resources |

**Using Sample Configurations:**

```bash
# Copy a sample configuration to use
cp config/llm/samples/models/model-Qwen2.5-7B-Instruct-GGUF.env config/llm/options/model.env
```

## 🎯 Quick Start

### Basic Usage

```python
import asyncio
from practorflow.llm import ModelPool, create_runner
from practorflow.settings.app_settings import appConfiguration

async def main():
    config = appConfiguration.ModelConfiguration
    pool = ModelPool(max_models=1)
    
    async with pool.acquire_context(config) as handle:
        runner = create_runner(handle)
        result = await runner.generate(prompt="What is Python?")
        print(result['reply'])

asyncio.run(main())
```

### Streaming Generation

```python
async with pool.acquire_context(config) as handle:
    runner = create_runner(handle)
    
    async for chunk in runner.generate_stream(prompt="Explain recursion"):
        if not chunk.finished:
            print(chunk.text, end="", flush=True)
        else:
            print(f"\n\nLatency: {chunk.latency_seconds:.2f}s")
```

### Secure Document Ingestion with RAG

```python
from practorflow.llm.knowledge.chroma_knowledge_store import ChromaKnowledgeStore
from practorflow.settings.app_settings import appConfiguration

# Initialize knowledge store (all data stays local)
kb_config = appConfiguration.KnowledgeChromaConfiguration
knowledge_store = ChromaKnowledgeStore(kb_config)

# Add documents - data never leaves your infrastructure
doc_info = knowledge_store.add_document_from_file("confidential_report.pdf")

# Create runner with knowledge store
async with pool.acquire_context(config) as handle:
    runner = create_runner(handle, knowledge_store=knowledge_store)
    
    # Set document scope for session-specific context
    runner.set_document_scope({doc_info['id']})
    
    # Search and generate - entirely within your environment
    search_result = runner.search("What are the key findings?")
    result = await runner.generate(prompt="Summarize the recommendations")
    print(result['reply'])
```

### Agentic Workflows with Pydantic AI

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext
from practorflow.llm.pyai import LocalLLMModel, KnowledgeDeps, search_knowledge

@dataclass
class Deps:
    knowledge_store: ChromaKnowledgeStore
    document_scope: set[str] | None = None

async with pool.acquire_context(config) as handle:
    runner = create_runner(handle)
    model = LocalLLMModel(runner)
    
    # Create agent that can reason across documents
    agent = Agent(
        model=model, 
        deps_type=KnowledgeDeps,
        system_prompt="You are a secure AI assistant that helps analyze internal documents."
    )
    agent.tool(search_knowledge)
    
    # Run with dependencies - all processing stays local
    deps = KnowledgeDeps(knowledge_store=knowledge_store)
    result = await agent.run(
        "What patterns do you see across our quarterly reports?",
        deps=deps
    )
    print(result.output)
```

### Web Search Integration (Optional)

```python
from practorflow.llm.tools import DuckDuckGoSearchTool

@dataclass
class WebSearchDeps:
    web_search_tool: DuckDuckGoSearchTool

agent = Agent(model=model, deps_type=WebSearchDeps)

@agent.tool
async def search_web(ctx: RunContext[WebSearchDeps], query: str) -> str:
    """Search the web for current information."""
    result = ctx.deps.web_search_tool.execute(query=query, max_results=5)
    return result.data if result.success else "No results found"

deps = WebSearchDeps(web_search_tool=DuckDuckGoSearchTool())
result = await agent.run(
    "What are industry best practices for data governance?",
    deps=deps
)
```

## 📚 Examples

Comprehensive examples are provided in the `src/examples/` directory:

- `src/examples/llm.py` - Basic usage, streaming, RAG workflows
- `src/examples/pyai.py` - Pydantic AI integration patterns
- `src/examples/chat.py` - Chat interface examples

Run examples:

```bash
cd src

# Basic examples
python examples/llm.py

# With document
python examples/llm.py path/to/document.pdf

# Pydantic AI examples
python examples/pyai.py

# Chat examples
python examples/chat.py
```

## 🗃️ Architecture

PractorFlow is built with a modular, production-ready architecture designed for secure, private AI deployments.

### Core Components

#### 1. LLM Runners
**Abstract base with multiple backend implementations:**
- `LlamaCppRunner` - GGUF models with llama.cpp (optimized for inference)
- `TransformersRunner` - HuggingFace models with transformers library
- Both support streaming, native function calling detection, and thinking trace extraction
- Unified interface: `generate()`, `generate_stream()`, `search()`

#### 2. Model Pool
**Production-ready async model management:**
- LRU eviction with reference counting for safe concurrent access
- Configuration-based caching (same config = reused model)
- Async context managers for automatic acquire/release
- Preload support for warm starts
- Thread-pool execution to avoid blocking event loop
- Supports multiple concurrent models with configurable limits

#### 3. Knowledge Store (RAG)
**Small-to-Big chunking strategy with ChromaDB:**
- **Retrieval chunks** (128 chars): Small, embedded chunks for precise similarity search
- **Context chunks** (1024 chars): Large parent chunks returned to LLM for rich context
- **Document metadata**: Full document tracking with IDs
- **Three collections**: `retrieval`, `context`, `documents`
- **Document scoping**: Isolate searches to specific documents per session/user
- Batch processing, persistent storage, and SentenceTransformer embeddings

#### 4. Session Management
**Dual-mode session persistence:**
- `InMemorySessionStore` - Fast, ephemeral (development/testing)
- `TinyDBSessionStore` - JSON file persistence (production)
- Session history with message tracking and document references
- Factory pattern for easy switching via configuration
- Search by title, user filtering, and sorted retrieval

#### 5. Services Layer

**Chat Service:**
- Session-based conversations with history management
- Context window management with intelligent truncation
- Document upload and scoping per session
- Streaming support with async generators

**Agent Service:**
- Multi-agent pipeline: Plan → Execute → Synthesize → Verify
- Tool integration with registry and OpenAI-format function definitions
- Session persistence with execution logs
- Retry logic and heuristic verification

**History Service:**
- Token estimation and context window management
- Message truncation with summarization
- Session title generation
- Memory-efficient history preparation

#### 6. Tool System
**Extensible tool registry with built-in tools:**
- `Calculator` - Math expressions with safe evaluation
- `KnowledgeSearchTool` - RAG queries with document scoping
- `DuckDuckGoSearchTool` - Free web search (no API key)
- `SerpAPISearchTool` - Premium web search with structured results
- `WebFetchTool` - URL content retrieval
- `TextSummarizerTool` - Extractive summarization
- `JSONTransformTool` - JSON manipulation
- OpenAI-compatible tool definitions for native function calling

#### 7. Pydantic AI Integration
**Drop-in model implementation:**
- `LocalLLMModel` - Implements Pydantic AI's `Model` protocol
- `LocalStreamedResponse` - Streaming with tool call extraction
- Message conversion between Pydantic AI and internal formats
- Tool call parsing from LLM responses (JSON extraction)
- Seamless integration with Pydantic AI agents

### Architecture Patterns

**Async-First Design:**
- All I/O operations are async (model loading, generation, document processing)
- Thread pool execution for CPU-bound operations (inference, embeddings)
- Async context managers for resource management
- Non-blocking streaming with async generators

**Configuration Management:**
- Environment-based configuration with `.env` files
- Singleton pattern with lazy initialization
- Validation and type safety with dataclasses
- Support for custom configuration paths (Docker/container deployments)

**Dependency Injection:**
- Service container for FastAPI dependency injection
- Factory pattern for session stores and runners
- Clean separation between business logic and infrastructure

**Privacy by Design:**
- All processing happens locally - no external API calls (except optional web search)
- Document scoping prevents cross-contamination between sessions
- Session isolation with user-based filtering
- Local model execution with full data control

### Data Flow

**Simple Generation:**
```
User → Runner → Model Pool (acquire) → Backend (llama.cpp/transformers) → Response
```

**RAG-Enhanced Generation:**
```
User → Runner.search() → Knowledge Store → ChromaDB (similarity search)
     → Runner.generate() → Context injection → Model → Response
```

**Agent Workflow:**
```
User Task → Planner (decompose) → Executor (tools + LLM reasoning)
         → Synthesizer (combine outputs) → Verifier (validate) → Final Answer
```

**Session-Based Chat:**
```
User → Chat Service → Session Store (load history) → History Manager (truncate)
    → Runner.generate() → Session Store (save) → Response Stream
```

### Small-to-Big Chunking Strategy

PractorFlow implements an efficient RAG strategy for secure document retrieval:

1. **Small Chunks (Retrieval)**: Documents are split into small chunks (128 chars) with embeddings for precise similarity search
2. **Large Chunks (Context)**: Parent chunks (1024 chars) provide rich context for the LLM
3. **Parent-Child Linking**: Small chunks reference their parent chunks
4. **Search Flow**: Search small chunks → Deduplicate by parent → Return parent chunks

This approach balances embedding quality with context richness while maintaining data locality.

## 🔧 Advanced Usage

### Custom Backend Selection

```python
# Use transformers backend with quantization
from practorflow.llm import LLMConfig

config = LLMConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    backend="transformers",
    device="auto",
    dtype="auto",
    quantization="4bit"  # Optional 4-bit/8-bit quantization
)
```

### Model Pool for Production Deployments

```python
from practorflow.llm import ModelPool

# Initialize pool with multiple models for concurrent requests
pool = ModelPool(max_models=3)

# Preload models at startup
await pool.preload(config1)
await pool.preload(config2)

# Acquire models concurrently
async with pool.acquire_context(config) as handle:
    # Model is automatically released on exit
    runner = create_runner(handle, knowledge_store)
    result = await runner.generate(prompt="Hello")
```

### Session-Based Document Scope

```python
# Add multiple documents for a specific session/user
session_doc_ids = set()
for filepath in ["contract.pdf", "policy.docx", "memo.txt"]:
    doc_info = knowledge_store.add_document_from_file(filepath)
    session_doc_ids.add(doc_info['id'])

# Scope search to session-specific documents
runner.set_document_scope(session_doc_ids)

# All searches are isolated to this session's documents
result = runner.search("What are the terms?")
```

### Native Function Calling

The library automatically detects and uses native function calling when supported by the model:

```python
# Check if model supports native function calling
if runner.supports_function_calling():
    print("Model has native tool calling support")

# Define tools in OpenAI format
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_data",
            "description": "Retrieve customer information from internal database",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer ID"}
                },
                "required": ["customer_id"]
            }
        }
    }
]

# Generate with tools
result = await runner.generate(
    prompt="What's the status of customer ABC123?",
    tools=tools
)

# Check for tool calls
if 'tool_calls' in result:
    for tc in result['tool_calls']:
        print(f"Tool: {tc['tool_name']}, Args: {tc['args']}")
```

### Custom Configuration Path

```python
from practorflow.settings.app_settings import load_configuration

# Load configuration from a custom directory
load_configuration(config_path="/etc/practorflow/config")

# Or for Docker/container deployments
load_configuration(config_path="/app/config")
```

## 🧪 Testing

```bash
cd src

# Run basic examples
python examples/llm.py

# Run Pydantic AI examples
python examples/pyai.py

# Test with document
python examples/llm.py path/to/test.pdf
```

## Additional Components

### API Server

PractorFlow API is a production-ready FastAPI server that exposes the core library's capabilities through RESTful endpoints, enabling web-based access to LLM inference with authentication, session management, and agent workflows.

**Key Features:**
- RESTful API with chat, agent, and session endpoints
- Three authentication modes: Open (no credentials), Local (app secret), OIDC (enterprise SSO)
- Server-Sent Events (SSE) streaming for real-time responses
- JWT-based authentication with configurable providers
- Multi-file upload with automatic knowledge base integration
- Background cleanup scheduler for maintenance tasks
- OpenAPI documentation with Swagger UI

**Use Cases:**
Ideal for integrating LLM capabilities into web applications, mobile apps, or microservices without direct Python integration. Enables multiple clients to share model resources efficiently while maintaining session isolation and security.

For detailed documentation including endpoint specifications, authentication setup, deployment guides, and configuration options, see **[API Server Documentation](src/api/README.md)**.

---

### GUI Client

PractorFlow GUI is a desktop application built with PySide6 that provides a user-friendly chat interface for interacting with the LLM service through a native desktop experience.

**Key Features:**
- Modern chat interface with markdown rendering and syntax highlighting
- Dual-mode operation: Chat mode and Agent mode (task execution with verification)
- Document upload via drag-and-drop (PDF, DOCX, images, etc.)
- Session history panel with search and resume functionality
- Real-time streaming with live token updates
- Message editing and resend capability
- Automatic dark/light theme support
- Background worker threads for responsive UI

**User Experience:**
Native desktop application with collapsible history sidebar, foldable documents panel, and seamless switching between chat and agent modes. All API communication handled in background threads to maintain UI responsiveness.

For detailed documentation including installation instructions, usage guide, keyboard shortcuts, and troubleshooting, see **[GUI Client Documentation](src/gui/README.md)**.

## 📊 Performance Tips

1. **GPU Acceleration**: Set `LLM_GPU_LAYERS=-1` to offload all layers to GPU
2. **Context Size**: Adjust `LLM_N_CTX` based on your model and GPU memory
3. **Batch Size**: Tune `LLM_N_BATCH` for optimal throughput
4. **Quantization**: Use 4-bit quantization for large models on limited hardware
5. **Model Pooling**: Set appropriate `max_models` based on available VRAM
6. **Embedding Model**: Use `all-MiniLM-L6-v2` for speed or `all-mpnet-base-v2` for quality

## 🎯 Who This Is For

- **Regulated Industries**: Healthcare, finance, legal—where data privacy is mandated
- **Consulting Firms**: Working with confidential client data
- **Enterprises**: Organizations with strict data governance requirements
- **Government Agencies**: Public sector with security clearance needs
- **Research Organizations**: Academic institutions with sensitive research data

## � Contributing

Contributions are welcome! We appreciate your help in building a better private AI service for organizations.

Please read our [CONTRIBUTING.md](CONTRIBUTING.md) for:
- Development setup and guidelines
- Code style and architecture principles
- Privacy considerations
- How to submit pull requests
- Areas where we need help

For major changes, please open an issue first to discuss what you would like to change.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE.txt](LICENSE.txt) file for details.

Apache 2.0 allows you to:
- ✅ Use commercially
- ✅ Modify and distribute
- ✅ Use privately
- ✅ Use patents granted by contributors

With the requirement to:
- 📄 Include license and copyright notice
- 📝 State significant changes made

## 🙏 Acknowledgments

- **llama.cpp** - Efficient GGUF model inference
- **Transformers** - HuggingFace model support
- **ChromaDB** - Vector database for RAG
- **Docling** - Advanced document parsing
- **Pydantic AI** - Type-safe agent framework
- **sentence-transformers** - High-quality embeddings

Built on modern open-source models including Qwen, Mistral, and others.

## 🔧 Support

For questions, issues, or feature requests, please [open an issue](https://github.com/vbouzoukos/PractorFlow/issues) on GitHub.

---

**Privacy by Design**: PractorFlow is built for organizations where AI must run where the data already lives. All inference, document processing, and reasoning happens entirely within your infrastructure. No data ever leaves your environment.