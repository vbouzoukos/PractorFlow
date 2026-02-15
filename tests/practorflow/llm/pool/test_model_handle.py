import time
from datetime import datetime
from unittest.mock import MagicMock, PropertyMock

import pytest

from practorflow.llm.pool.model_handle import ModelHandle
from practorflow.llm.llm_config import LLMConfig

from tests.practorflow.llm.pool.model_test_common import (
    create_mock_llm_config,
    create_mock_model_handle,
)


class TestModelHandleLifecycle:
    """Tests for ModelHandle reference counting and lifecycle metadata."""

    @pytest.fixture
    def handle(self):
        return create_mock_model_handle()

    def test_acquire_increments_ref_count_and_updates_last_used(self, handle):
        initial_last_used = handle.last_used_at
        time.sleep(0.01)

        handle.acquire()

        assert handle.ref_count == 1
        assert handle.last_used_at > initial_last_used

    def test_release_decrements_ref_count(self, handle):
        handle.acquire()
        handle.acquire()
        assert handle.ref_count == 2

        handle.release()
        assert handle.ref_count == 1

    def test_release_does_not_go_below_zero(self, handle):
        handle.release()
        assert handle.ref_count == 0

    def test_is_in_use(self, handle):
        assert handle.is_in_use is False
        handle.acquire()
        assert handle.is_in_use is True
        handle.release()
        assert handle.is_in_use is False


class TestModelHandleBackendFlags:
    """Tests backend identification helpers."""

    def test_llama_cpp_flags(self):
        handle = create_mock_model_handle(backend="llama_cpp")
        assert handle.is_llama_cpp is True
        assert handle.is_transformers is False

    def test_transformers_flags(self):
        handle = create_mock_model_handle(
            backend="transformers",
            with_tokenizer=True,
        )
        assert handle.is_transformers is True
        assert handle.is_llama_cpp is False

    def test_unknown_backend_flags(self):
        handle = create_mock_model_handle(backend="unknown")
        assert handle.is_llama_cpp is False
        assert handle.is_transformers is False


class TestModelHandleChatTemplate:
    """Tests chat template resolution paths."""

    def test_llama_cpp_template_from_metadata(self):
        model = MagicMock()
        model.metadata = {"tokenizer.chat_template": "llama-template"}

        handle = ModelHandle(
            config=create_mock_llm_config(),
            backend="llama_cpp",
            model=model,
        )

        assert handle.get_chat_template() == "llama-template"

    def test_llama_cpp_no_template(self):
        model = MagicMock()
        model.metadata = {}

        handle = ModelHandle(
            config=create_mock_llm_config(),
            backend="llama_cpp",
            model=model,
        )

        assert handle.get_chat_template() is None

    def test_llama_cpp_metadata_exception(self):
        model = MagicMock()
        type(model).metadata = PropertyMock(side_effect=Exception("boom"))

        handle = ModelHandle(
            config=create_mock_llm_config(),
            backend="llama_cpp",
            model=model,
        )

        assert handle.get_chat_template() is None

    def test_transformers_template_from_tokenizer(self):
        tokenizer = MagicMock()
        tokenizer.chat_template = "transformers-template"

        handle = ModelHandle(
            config=create_mock_llm_config(backend="transformers"),
            backend="transformers",
            model=MagicMock(),
            tokenizer=tokenizer,
        )

        assert handle.get_chat_template() == "transformers-template"

    def test_transformers_no_tokenizer(self):
        handle = ModelHandle(
            config=create_mock_llm_config(backend="transformers"),
            backend="transformers",
            model=MagicMock(),
            tokenizer=None,
        )

        assert handle.get_chat_template() is None

    def test_transformers_tokenizer_exception(self):
        tokenizer = MagicMock()
        type(tokenizer).chat_template = PropertyMock(side_effect=Exception("boom"))

        handle = ModelHandle(
            config=create_mock_llm_config(backend="transformers"),
            backend="transformers",
            model=MagicMock(),
            tokenizer=tokenizer,
        )

        assert handle.get_chat_template() is None


class TestModelHandleTimestamps:
    """Tests timestamp behavior."""

    def test_created_and_last_used_set_on_init(self):
        handle = create_mock_model_handle()

        assert isinstance(handle.created_at, datetime)
        assert isinstance(handle.last_used_at, datetime)

    def test_touch_updates_last_used_only(self):
        handle = create_mock_model_handle()

        created_at = handle.created_at
        last_used = handle.last_used_at
        time.sleep(0.01)

        handle.touch()

        assert handle.created_at == created_at
        assert handle.last_used_at > last_used
