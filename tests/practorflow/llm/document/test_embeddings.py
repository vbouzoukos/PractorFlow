import numpy as np
import pytest
from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# SentenceTransformerEmbeddingModel (fully mocked import)
# ---------------------------------------------------------------------------

def test_sentence_transformer_embedding_model_init_and_methods(monkeypatch):
    from practorflow.llm.document import embeddings

    fake_model = MagicMock()
    fake_model.get_sentence_embedding_dimension.return_value = 384
    fake_model.encode.return_value = np.ones((1, 384), dtype=np.float32)

    fake_sentence_transformer = MagicMock(return_value=fake_model)

    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        MagicMock(SentenceTransformer=fake_sentence_transformer),
    )

    model = embeddings.SentenceTransformerEmbeddingModel(
        model_name="fake-model",
        device="cpu",
        cache_dir="/tmp",
    )

    assert model.embedding_dimension == 384
    assert model.embed("hello").shape == (1, 384)
    assert model.embed_batch(["a", "b"]).shape == (1, 384)


# ---------------------------------------------------------------------------
# LLMEmbeddingModel – transformers backend
# ---------------------------------------------------------------------------

def test_llm_embedding_transformers_dimension_from_hidden_size():
    from practorflow.llm.document.embeddings import LLMEmbeddingModel

    mock_model = MagicMock()
    mock_model.config.hidden_size = 1024

    model = LLMEmbeddingModel(
        llm_model=mock_model,
        backend="transformers",
        tokenizer=MagicMock(),
        device="cpu",
    )

    assert model.embedding_dimension == 1024


def test_llm_embedding_transformers_dimension_default_warning(monkeypatch):
    from practorflow.llm.document import embeddings

    # config without hidden_size
    mock_config = MagicMock(spec=[])
    mock_model = MagicMock()
    mock_model.config = mock_config

    warning_hit = False

    def fake_warning(*args, **kwargs):
        nonlocal warning_hit
        warning_hit = True

    monkeypatch.setattr(embeddings.logger, "warning", fake_warning)

    model = embeddings.LLMEmbeddingModel(
        llm_model=mock_model,
        backend="transformers",
        tokenizer=MagicMock(),
        device="cpu",
    )

    assert model.embedding_dimension == 768
    assert warning_hit is True


def test_llm_embedding_transformers_embed_and_embed_batch(monkeypatch):
    from practorflow.llm.document.embeddings import LLMEmbeddingModel

    mock_model = MagicMock()
    mock_model.config.hidden_size = 4

    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }

    monkeypatch.setitem(
        __import__("sys").modules,
        "torch",
        MagicMock(
            no_grad=MagicMock(),
            nn=MagicMock(functional=MagicMock(normalize=lambda x, p, dim: x)),
        ),
    )

    embeddings_np = np.ones((2, 4), dtype=np.float32)

    model = LLMEmbeddingModel(
        llm_model=mock_model,
        backend="transformers",
        tokenizer=tokenizer,
        device="cpu",
    )

    monkeypatch.setattr(
        model,
        "_embed_transformers_batch",
        lambda texts: embeddings_np[: len(texts)],
    )

    assert model.embed("hello").shape == (4,)
    assert model.embed_batch(["a", "b"], batch_size=1).shape == (2, 4)


def test_llm_embedding_transformers_tokenizer_missing():
    from practorflow.llm.document.embeddings import LLMEmbeddingModel

    mock_model = MagicMock()
    mock_model.config.hidden_size = 10

    model = LLMEmbeddingModel(
        llm_model=mock_model,
        backend="transformers",
        tokenizer=None,
        device="cpu",
    )

    with pytest.raises(ValueError, match="Tokenizer required"):
        model._embed_transformers_batch(["test"])


# ---------------------------------------------------------------------------
# LLMEmbeddingModel – llama_cpp backend
# ---------------------------------------------------------------------------

def test_llm_embedding_llama_cpp_dimension_success():
    from practorflow.llm.document.embeddings import LLMEmbeddingModel

    mock_llama = MagicMock()
    mock_llama.create_embedding.return_value = {
        "data": [{"embedding": [1.0, 2.0, 3.0]}]
    }

    model = LLMEmbeddingModel(mock_llama, backend="llama_cpp")
    assert model.embedding_dimension == 3


def test_llm_embedding_llama_cpp_dimension_error(monkeypatch):
    from practorflow.llm.document import embeddings

    mock_llama = MagicMock()
    mock_llama.create_embedding.side_effect = RuntimeError("boom")

    error_hit = False

    def fake_error(*args, **kwargs):
        nonlocal error_hit
        error_hit = True

    monkeypatch.setattr(embeddings.logger, "error", fake_error)

    with pytest.raises(RuntimeError):
        embeddings.LLMEmbeddingModel(mock_llama, backend="llama_cpp")

    assert error_hit is True


def test_llm_embedding_llama_cpp_embed_and_batch():
    from practorflow.llm.document.embeddings import LLMEmbeddingModel

    mock_llama = MagicMock()
    mock_llama.create_embedding.return_value = {
        "data": [{"embedding": [3.0, 4.0]}]
    }

    model = LLMEmbeddingModel(mock_llama, backend="llama_cpp")

    assert model.embed("hello").shape == (2,)
    assert model.embed_batch(["a", "b"], batch_size=1).shape == (2, 2)


# ---------------------------------------------------------------------------
# Unsupported backend
# ---------------------------------------------------------------------------

def test_llm_embedding_unsupported_backend():
    from practorflow.llm.document.embeddings import LLMEmbeddingModel

    model = LLMEmbeddingModel(MagicMock(), backend="invalid_backend")

    with pytest.raises(ValueError, match="Unsupported backend"):
        model.embed("test")

def test_llm_embedding_embed_returns_embeddings_for_list_input():
    import numpy as np
    from unittest.mock import MagicMock
    from practorflow.llm.document.embeddings import LLMEmbeddingModel

    mock_model = MagicMock()
    mock_model.config.hidden_size = 3

    expected = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, 1.0, 0.0]],
        dtype=np.float32,
    )

    model = LLMEmbeddingModel(
        llm_model=mock_model,
        backend="transformers",
        tokenizer=MagicMock(),
        device="cpu",
    )

    # Force the branch without invoking torch
    model._embed_transformers_batch = MagicMock(return_value=expected)

    result = model.embed(["text one", "text two"])

    assert result is expected
    assert result.shape == (2, 3)

def test_embed_transformers_batch_full_flow(monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    from unittest.mock import MagicMock
    from practorflow.llm.document.embeddings import LLMEmbeddingModel

    class FakeTensor:
        def __init__(self, value):
            self.value = np.array(value, dtype=np.float32)

        def to(self, device):
            return self

        def unsqueeze(self, dim):
            return FakeTensor(np.expand_dims(self.value, axis=dim))

        def sum(self, dim):
            return FakeTensor(self.value.sum(axis=dim))

        def __mul__(self, other):
            return FakeTensor(self.value * other.value)

        def __truediv__(self, other):
            return FakeTensor(self.value / other.value)

        def cpu(self):
            return self

        def numpy(self):
            return self.value

    fake_torch = MagicMock()
    fake_torch.no_grad.return_value.__enter__.return_value = None
    fake_torch.no_grad.return_value.__exit__.return_value = None
    fake_torch.nn.functional.normalize.side_effect = lambda x, p, dim: x

    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

    attention_mask = FakeTensor([[1, 1, 0], [1, 1, 1]])
    input_ids = FakeTensor([[1, 2, 3], [4, 5, 6]])

    tokenizer = MagicMock(return_value={
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    })

    hidden_state = FakeTensor(
        [
            [[1, 0], [0, 1], [0, 0]],
            [[1, 1], [1, 1], [1, 1]],
        ]
    )

    outputs = SimpleNamespace(hidden_states=[hidden_state])

    model = MagicMock(return_value=outputs)
    model.config.hidden_size = 2

    embedding_model = LLMEmbeddingModel(
        llm_model=model,
        backend="transformers",
        tokenizer=tokenizer,
        device="cpu",
    )

    result = embedding_model._embed_transformers_batch(
        ["hello", "world"]
    )

    assert isinstance(result, np.ndarray)
    assert result.shape == (2, 2)

def test_llm_embedding_embed_batch_unsupported_backend():
    from unittest.mock import MagicMock
    from practorflow.llm.document.embeddings import LLMEmbeddingModel

    model = LLMEmbeddingModel(
        llm_model=MagicMock(),
        backend="invalid_backend",
    )

    with pytest.raises(ValueError, match="Unsupported backend"):
        model.embed_batch(["a", "b"], batch_size=1)


def test_llm_embedding_embed_batch_show_progress_logs(monkeypatch):
    from unittest.mock import MagicMock
    from practorflow.llm.document import embeddings
    from practorflow.llm.document.embeddings import LLMEmbeddingModel
    import numpy as np

    debug_hit = False

    def fake_debug(*args, **kwargs):
        nonlocal debug_hit
        debug_hit = True

    monkeypatch.setattr(embeddings.logger, "debug", fake_debug)

    mock_model = MagicMock()
    mock_model.create_embedding.return_value = {
        "data": [{"embedding": [1.0, 0.0]}]
    }

    model = LLMEmbeddingModel(
        llm_model=mock_model,
        backend="llama_cpp",
    )

    result = model.embed_batch(
        ["a", "b", "c"],
        batch_size=2,
        show_progress=True,
    )

    assert isinstance(result, np.ndarray)
    assert debug_hit is True


def test_llm_embedding_llama_cpp_nested_embedding_list_handling():
    import numpy as np
    from unittest.mock import MagicMock
    from practorflow.llm.document.embeddings import LLMEmbeddingModel

    mock_llama = MagicMock()
    mock_llama.create_embedding.return_value = {
        "data": [{"embedding": [[1.0, 2.0, 3.0]]}]
    }

    model = LLMEmbeddingModel(
        llm_model=mock_llama,
        backend="llama_cpp",
    )

    embedding = model.embed("test")

    expected = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    expected = expected / np.linalg.norm(expected)

    assert embedding.shape == (3,)
    assert np.allclose(embedding, expected)

def test_sentence_transformer_embedding_model_uses_local_files_only_when_model_cached(
    monkeypatch, tmp_path
):
    from practorflow.llm.document import embeddings

    # Arrange: fake cached model directory
    model_name = "fake-model"
    cache_dir = tmp_path
    model_path = cache_dir / model_name
    model_path.mkdir()

    fake_model = MagicMock()
    fake_model.get_sentence_embedding_dimension.return_value = 123

    captured_kwargs = {}

    def fake_sentence_transformer(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return fake_model

    monkeypatch.setitem(
        __import__("sys").modules,
        "sentence_transformers",
        MagicMock(SentenceTransformer=fake_sentence_transformer),
    )

    model = embeddings.SentenceTransformerEmbeddingModel(
        model_name=model_name,
        device="cpu",
        cache_dir=str(cache_dir),
    )

    assert captured_kwargs["local_files_only"] is True
    assert model.embedding_dimension == 123
