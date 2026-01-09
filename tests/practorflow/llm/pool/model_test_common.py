from datetime import datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock

from practorflow.llm.llm_config import LLMConfig
from practorflow.llm.pool.model_handle import ModelHandle


def create_mock_llm_config(
    *,
    model_name: str = "test-model",
    backend: str = "llama_cpp",
    device: str = "cpu",
    dtype: str = "float16",
    quantization: Optional[str] = None,
    n_gpu_layers: int = 0,
    n_ctx: int = 4096,
    n_batch: int = 512,
    use_torch_compile: bool = False,
    compile_mode: str = "default",
    warmup_on_load: bool = True,
):
    """
    Create a deterministic mock LLMConfig suitable for ModelHandle
    and ModelPool hashing logic.
    """
    config = MagicMock(spec=LLMConfig)
    config.model_name = model_name
    config.backend = backend
    config.device = device
    config.dtype = dtype
    config.quantization = quantization
    config.n_gpu_layers = n_gpu_layers
    config.n_ctx = n_ctx
    config.n_batch = n_batch
    config.use_torch_compile = use_torch_compile
    config.compile_mode = compile_mode
    config.warmup_on_load = warmup_on_load

    # Required by other pool paths
    config.temperature = 0.7
    config.top_p = 0.9
    config.max_new_tokens = 512
    config.models_dir = "/tmp/models"

    return config


def create_mock_model_handle(
    *,
    config=None,
    backend: str = "llama_cpp",
    ref_count: int = 0,
    last_used_at: Optional[datetime] = None,
    with_tokenizer: bool = False,
    config_hash: str = "deadbeefcafebabe",
) -> ModelHandle:
    """
    Create a real ModelHandle with controlled internal state.
    """
    if config is None:
        config = create_mock_llm_config(backend=backend)

    handle = ModelHandle(
        config=config,
        backend=backend,
        model=MagicMock(),
        tokenizer=MagicMock() if with_tokenizer else None,
        config_hash=config_hash,
    )

    handle.ref_count = ref_count

    if last_used_at is not None:
        handle.last_used_at = last_used_at

    return handle


def make_handle_in_use(handle: ModelHandle, refs: int = 1) -> None:
    """
    Increment ref_count using real acquire() calls.
    """
    for _ in range(refs):
        handle.acquire()


def advance_last_used(handle: ModelHandle, seconds: int) -> None:
    """
    Move last_used_at backwards in time for LRU eviction tests.
    """
    handle.last_used_at = handle.last_used_at - timedelta(seconds=seconds)

