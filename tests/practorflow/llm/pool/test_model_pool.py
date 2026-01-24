import asyncio
import os
from unittest.mock import MagicMock, patch

import pytest
import torch

from practorflow.llm.pool.model_pool import ModelPool, validate_model_path
from tests.practorflow.llm.pool.model_test_common import (
    create_mock_llm_config,
    create_mock_model_handle,
    make_handle_in_use,
    advance_last_used,
)

# ===================== validate_model_path =====================


def test_validate_model_path_missing_logs_warning():
    fake_logger = MagicMock()
    with (
        patch("practorflow.llm.pool.model_pool.logger", fake_logger),
        patch("os.path.exists", return_value=False),
    ):
        validate_model_path("/missing")
    fake_logger.warning.assert_called_once()


# ===================== singleton =====================


def test_singleton_get_and_reset():
    ModelPool.reset_instance()
    p1 = ModelPool.get_instance(max_models=1)
    p2 = ModelPool.get_instance(max_models=2)
    assert p1 is p2
    ModelPool.reset_instance()
    assert ModelPool._instance is None


# ===================== config hash =====================


def test_compute_config_hash_deterministic_and_missing_attrs():
    pool = ModelPool()
    cfg = create_mock_llm_config()

    h1 = pool._compute_config_hash(cfg)
    h2 = pool._compute_config_hash(cfg)
    assert h1 == h2

    del cfg.use_torch_compile
    del cfg.compile_mode
    h3 = pool._compute_config_hash(cfg)
    assert isinstance(h3, str)


# ===================== loading lock =====================


@pytest.mark.asyncio
async def test_get_loading_lock_create_and_reuse():
    pool = ModelPool()
    l1 = await pool._get_loading_lock("x")
    l2 = await pool._get_loading_lock("x")
    assert l1 is l2


# ===================== eviction =====================


@pytest.mark.asyncio
async def test_evict_lru_success_and_tokenizer_deletion():
    pool = ModelPool()
    h1 = create_mock_model_handle()
    h2 = create_mock_model_handle(with_tokenizer=True)

    advance_last_used(h2, 10)
    pool._models = {"a": h1, "b": h2}

    assert await pool._evict_lru() is True
    assert len(pool._models) == 1
    assert h2.tokenizer is None


@pytest.mark.asyncio
async def test_evict_lru_all_in_use():
    pool = ModelPool()
    h = create_mock_model_handle()
    make_handle_in_use(h)
    pool._models = {"x": h}

    assert await pool._evict_lru() is False


# ===================== llama.cpp loading =====================


def test_load_llama_cpp_invalid_name_raises(tmp_path):
    pool = ModelPool()
    cfg = create_mock_llm_config()
    cfg.backend = "llama_cpp"
    cfg.models_dir = str(tmp_path)
    cfg.model_name = "invalidname"

    with pytest.raises(ValueError, match="model_name must be"):
        pool._load_llama_cpp_model(cfg)


def test_load_llama_cpp_hf_download_not_file(tmp_path):
    pool = ModelPool()
    cfg = create_mock_llm_config()
    cfg.backend = "llama_cpp"
    cfg.models_dir = str(tmp_path)
    cfg.model_name = "repo/model.gguf"

    with (
        patch("os.walk", return_value=[]),
        patch("huggingface_hub.hf_hub_download", return_value=str(tmp_path / "x.gguf")),
        patch("os.path.isfile", return_value=False),
    ):
        with pytest.raises(ValueError, match="not a file"):
            pool._load_llama_cpp_model(cfg)


def test_load_llama_cpp_invalid_gguf_magic(tmp_path):
    pool = ModelPool()
    cfg = create_mock_llm_config()
    cfg.backend = "llama_cpp"
    cfg.models_dir = str(tmp_path)
    cfg.model_name = "repo/model.gguf"

    bad = tmp_path / "model.gguf"
    bad.write_bytes(b"BAD!")

    with patch("os.walk", return_value=[(str(tmp_path), [], ["model.gguf"])]):
        with pytest.raises(ValueError, match="not a valid GGUF"):
            pool._load_llama_cpp_model(cfg)


def test_load_llama_cpp_success(tmp_path):
    pool = ModelPool()
    cfg = create_mock_llm_config()
    cfg.backend = "llama_cpp"
    cfg.models_dir = str(tmp_path)
    cfg.model_name = "repo/model.gguf"

    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"GGUFxxxx")

    fake_llama = MagicMock()
    fake_llama.n_ctx.return_value = 4096

    with (
        patch("llama_cpp.Llama", return_value=fake_llama),
        patch("os.walk", return_value=[(str(tmp_path), [], ["model.gguf"])]),
    ):
        handle = pool._load_llama_cpp_model(cfg)
        assert handle.backend == "llama_cpp"


# ===================== compile model =====================


def test_compile_model_disabled():
    pool = ModelPool()
    model = MagicMock()
    cfg = create_mock_llm_config(use_torch_compile=False)
    assert pool._compile_model(model, cfg) is model


def test_compile_model_torch_too_old():
    pool = ModelPool()
    model = MagicMock()
    cfg = create_mock_llm_config(use_torch_compile=True)

    with patch("torch.__version__", "1.13.0"):
        assert pool._compile_model(model, cfg) is model


def test_compile_model_no_cuda():
    pool = ModelPool()
    model = MagicMock()
    cfg = create_mock_llm_config(use_torch_compile=True)

    with (
        patch("torch.__version__", "2.1.0"),
        patch("torch.cuda.is_available", return_value=False),
    ):
        assert pool._compile_model(model, cfg) is model


def test_compile_model_success_and_exception():
    pool = ModelPool()
    model = MagicMock()
    cfg = create_mock_llm_config(use_torch_compile=True)

    with (
        patch("torch.__version__", "2.1.0"),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.compile", return_value="compiled"),
    ):
        assert pool._compile_model(model, cfg) == "compiled"

    with (
        patch("torch.__version__", "2.1.0"),
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.compile", side_effect=RuntimeError("boom")),
    ):
        assert pool._compile_model(model, cfg) is model


# ===================== warmup =====================


def test_warmup_model_success_and_exception():
    pool = ModelPool()
    model = MagicMock()
    tokenizer = MagicMock()
    tokenizer.return_value.to.return_value = {}
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 1
    cfg = create_mock_llm_config()

    pool._warmup_model(model, tokenizer, "cpu", cfg)

    tokenizer.side_effect = RuntimeError("boom")
    pool._warmup_model(model, tokenizer, "cpu", cfg)


# ===================== transformers loading =====================


def test_load_transformers_all_branches(tmp_path):
    pool = ModelPool()
    cfg = create_mock_llm_config(backend="transformers")
    cfg.models_dir = str(tmp_path)
    cfg.dtype = "float16"
    cfg.device = "cuda"
    cfg.quantization = "8bit"
    cfg.warmup_on_load = False

    model = MagicMock()
    model.config.max_position_embeddings = 1024
    model.device = "cpu"

    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
        patch("transformers.BitsAndBytesConfig", MagicMock()),
        patch("practorflow.llm.pool.model_pool.validate_model_path"),
        patch.object(pool, "_compile_model", return_value=model),
    ):
        handle = pool._load_transformers_model(cfg)
        assert handle.backend == "transformers"


# ===================== acquire / release / context =====================


@pytest.mark.asyncio
async def test_acquire_cache_hit_and_release():
    pool = ModelPool()
    cfg = create_mock_llm_config()
    h = create_mock_model_handle()
    pool._models[pool._compute_config_hash(cfg)] = h

    handle = await pool.acquire(cfg)
    assert handle.ref_count == 1
    await pool.release(handle)
    assert handle.ref_count == 0


@pytest.mark.asyncio
async def test_acquire_new_context_and_eviction_failure():
    pool = ModelPool(max_models=1)
    cfg1 = create_mock_llm_config(model_name="a")
    cfg2 = create_mock_llm_config(model_name="b")

    with patch.object(pool, "_load_model", return_value=create_mock_model_handle()):
        async with pool.acquire_context(cfg1):
            pass

    h = create_mock_model_handle()
    make_handle_in_use(h)
    pool._models[pool._compute_config_hash(cfg1)] = h

    with pytest.raises(RuntimeError):
        await pool.acquire(cfg2)


# ===================== preload =====================


@pytest.mark.asyncio
async def test_preload():
    pool = ModelPool()
    cfg = create_mock_llm_config()

    with patch.object(pool, "_load_model", return_value=create_mock_model_handle()):
        await pool.preload(cfg)

    assert len(pool._models) == 1


# ===================== unload =====================


@pytest.mark.asyncio
async def test_unload_missing_in_use_and_success():
    pool = ModelPool()
    cfg = create_mock_llm_config()

    assert await pool.unload(cfg) is False

    h = create_mock_model_handle()
    make_handle_in_use(h)
    pool._models[pool._compute_config_hash(cfg)] = h
    assert await pool.unload(cfg) is False

    h2 = create_mock_model_handle(with_tokenizer=True)
    pool._models[pool._compute_config_hash(cfg)] = h2

    assert await pool.unload(cfg) is True
    assert h2.tokenizer is None


# ===================== unload all =====================


@pytest.mark.asyncio
async def test_unload_all_skip_and_force():
    pool = ModelPool()

    h1 = create_mock_model_handle()
    h2 = create_mock_model_handle(with_tokenizer=True)

    make_handle_in_use(h2)

    pool._models = {"1": h1, "2": h2}

    # skip in-use model
    assert await pool.unload_all(force=False) == 1

    # force unload remaining model (executes tokenizer deletion)
    assert await pool.unload_all(force=True) == 1
    assert h2.tokenizer is None


# ===================== stats and repr =====================


def test_get_stats_and_repr():
    pool = ModelPool(max_models=2)
    h = create_mock_model_handle()
    pool._models["x"] = h

    stats = pool.get_stats()
    assert stats["loaded_models"] == 1
    assert "created_at" in stats["models"][0]

    r = repr(pool)
    assert "ModelPool" in r


@pytest.mark.asyncio
async def test_evict_lru_deletes_tokenizer_branch():
    pool = ModelPool()

    h_old = create_mock_model_handle(with_tokenizer=True)
    h_new = create_mock_model_handle()

    advance_last_used(h_old, 100)

    pool._models = {
        "old": h_old,
        "new": h_new,
    }

    evicted = await pool._evict_lru()

    assert evicted is True
    assert h_old.tokenizer is None


def test_load_transformers_sets_dtype_auto_and_device_map_auto(tmp_path):
    pool = ModelPool()
    cfg = create_mock_llm_config(backend="transformers")
    cfg.models_dir = str(tmp_path)
    cfg.dtype = "auto"
    cfg.device = "auto"
    cfg.quantization = None
    cfg.warmup_on_load = False

    model = MagicMock()
    model.config.max_position_embeddings = 1024
    model.device = "cpu"

    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained", return_value=model
        ) as mock_from_pretrained,
        patch("practorflow.llm.pool.model_pool.validate_model_path"),
        patch.object(pool, "_compile_model", return_value=model),
    ):
        handle = pool._load_transformers_model(cfg)
        assert handle.backend == "transformers"

        _, kwargs = mock_from_pretrained.call_args
        assert kwargs["dtype"] == "auto"
        assert kwargs["device_map"] == "auto"


def test_transformers_quantization_4bit_kwargs(tmp_path):
    pool = ModelPool()

    cfg = create_mock_llm_config(backend="transformers")
    cfg.models_dir = str(tmp_path)
    cfg.quantization = "4bit"
    cfg.dtype = "auto"
    cfg.device = "cpu"
    cfg.warmup_on_load = False

    model = MagicMock()
    model.config.max_position_embeddings = 512
    model.device = "cpu"

    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"

    fake_bnb_instance = MagicMock()
    fake_bnb_cls = MagicMock(return_value=fake_bnb_instance)

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch(
            "transformers.AutoModelForCausalLM.from_pretrained"
        ) as mock_from_pretrained,
        patch("transformers.BitsAndBytesConfig", fake_bnb_cls),
        patch("practorflow.llm.pool.model_pool.validate_model_path"),
        patch.object(pool, "_compile_model", return_value=model),
    ):

        mock_from_pretrained.return_value = model

        pool._load_transformers_model(cfg)

        fake_bnb_cls.assert_called_once_with(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )


def test_transformers_quantization_import_error_only(tmp_path):
    pool = ModelPool()

    cfg = create_mock_llm_config(backend="transformers")
    cfg.models_dir = str(tmp_path)
    cfg.quantization = "4bit"
    cfg.dtype = "auto"
    cfg.device = "cpu"
    cfg.warmup_on_load = False

    model = MagicMock()
    model.config.max_position_embeddings = 512
    model.device = "cpu"

    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
        patch("transformers.BitsAndBytesConfig", side_effect=ImportError),
        patch("practorflow.llm.pool.model_pool.validate_model_path"),
        patch.object(pool, "_compile_model", return_value=model),
    ):

        pool._load_transformers_model(cfg)


def test_load_transformers_uses_parameters_device_when_model_has_no_device(tmp_path):
    pool = ModelPool()

    cfg = create_mock_llm_config(backend="transformers")
    cfg.models_dir = str(tmp_path)
    cfg.dtype = "auto"
    cfg.device = "cpu"
    cfg.quantization = None
    cfg.warmup_on_load = False

    fake_param = MagicMock()
    fake_param.device = "cpu"

    model = MagicMock(spec=["parameters", "to", "eval", "config"])
    model.parameters.return_value = iter([fake_param])
    model.to.return_value = model
    model.eval.return_value = None
    model.config.max_position_embeddings = 256

    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
        patch("practorflow.llm.pool.model_pool.validate_model_path"),
        patch.object(pool, "_compile_model", return_value=model),
    ):

        handle = pool._load_transformers_model(cfg)
        assert handle.backend == "transformers"


def test_load_transformers_calls_warmup_when_enabled(tmp_path):
    pool = ModelPool()

    cfg = create_mock_llm_config(backend="transformers")
    cfg.models_dir = str(tmp_path)
    cfg.dtype = "auto"
    cfg.device = "cpu"
    cfg.quantization = None
    cfg.warmup_on_load = True
    cfg.use_torch_compile = True

    fake_param = MagicMock()
    fake_param.device = "cpu"

    model = MagicMock(spec=["parameters", "to", "eval", "config"])
    model.parameters.return_value = iter([fake_param])
    model.to.return_value = model
    model.eval.return_value = None
    model.config.max_position_embeddings = 256

    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
        patch("practorflow.llm.pool.model_pool.validate_model_path"),
        patch.object(pool, "_compile_model", return_value=model),
        patch.object(pool, "_warmup_model") as warmup_mock,
    ):

        pool._load_transformers_model(cfg)

        warmup_mock.assert_called_once()


def test_load_transformers_uses_config_n_ctx_when_model_config_missing(tmp_path):
    pool = ModelPool()

    cfg = create_mock_llm_config(backend="transformers")
    cfg.models_dir = str(tmp_path)
    cfg.dtype = "auto"
    cfg.device = "cpu"
    cfg.quantization = None
    cfg.warmup_on_load = False
    cfg.n_ctx = 4096

    fake_param = MagicMock()
    fake_param.device = "cpu"

    model = MagicMock(spec=["parameters", "to", "eval"])
    model.parameters.return_value = iter([fake_param])
    model.to.return_value = model
    model.eval.return_value = None
    # IMPORTANT: no model.config attribute at all

    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<eos>"

    with (
        patch("transformers.AutoTokenizer.from_pretrained", return_value=tokenizer),
        patch("transformers.AutoModelForCausalLM.from_pretrained", return_value=model),
        patch("practorflow.llm.pool.model_pool.validate_model_path"),
        patch.object(pool, "_compile_model", return_value=model),
    ):

        handle = pool._load_transformers_model(cfg)

        assert handle.max_context_length == cfg.n_ctx


@pytest.mark.asyncio
async def test_load_model_llama_cpp_branch():
    pool = ModelPool()

    cfg = create_mock_llm_config()
    cfg.backend = "llama_cpp"

    fake_handle = create_mock_model_handle()

    async def fake_executor(_, fn, config):
        assert fn.__func__ is ModelPool._load_llama_cpp_model
        assert fn.__self__ is pool
        return fake_handle

    with patch("asyncio.get_running_loop") as loop_mock:
        loop = MagicMock()
        loop.run_in_executor.side_effect = fake_executor
        loop_mock.return_value = loop

        handle = await pool._load_model(cfg)
        assert handle is fake_handle


@pytest.mark.asyncio
async def test_load_model_transformers_branch():
    pool = ModelPool()

    cfg = create_mock_llm_config(backend="transformers")

    fake_handle = create_mock_model_handle(backend="transformers")

    async def fake_executor(_, fn, config):
        assert fn.__func__ is ModelPool._load_transformers_model
        assert fn.__self__ is pool
        return fake_handle

    with patch("asyncio.get_running_loop") as loop_mock:
        loop = MagicMock()
        loop.run_in_executor.side_effect = fake_executor
        loop_mock.return_value = loop

        handle = await pool._load_model(cfg)
        assert handle is fake_handle


@pytest.mark.asyncio
async def test_load_model_unsupported_backend_raises():
    pool = ModelPool()

    cfg = create_mock_llm_config()
    cfg.backend = "unknown"

    with pytest.raises(ValueError):
        await pool._load_model(cfg)


@pytest.mark.asyncio
async def test_acquire_hits_cache_inside_loading_lock():
    pool = ModelPool()

    cfg = create_mock_llm_config()
    config_hash = pool._compute_config_hash(cfg)
    handle = create_mock_model_handle()

    class FakeLoadingLock:
        async def __aenter__(self):
            # Insert model AFTER first cache check
            pool._models[config_hash] = handle
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

    async def fake_get_loading_lock(_):
        return FakeLoadingLock()

    with patch.object(pool, "_get_loading_lock", side_effect=fake_get_loading_lock):
        acquired = await pool.acquire(cfg)

        assert acquired is handle
        assert handle.ref_count == 1
