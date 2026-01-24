import torch
import pytest

from practorflow.converters.torch_dtype_convertor import dtype_from_config


def test_dtype_from_config_none():
    """None input should return None"""
    assert dtype_from_config(None) is None


def test_dtype_from_config_auto():
    """'auto' should be passed through unchanged"""
    assert dtype_from_config("auto") == "auto"


@pytest.mark.parametrize(
    "dtype_name,expected",
    [
        ("float16", torch.float16),
        ("float32", torch.float32),
        ("float64", torch.float64),
        ("bfloat16", torch.bfloat16),
        ("int8", torch.int8),
        ("int16", torch.int16),
        ("int32", torch.int32),
        ("int64", torch.int64),
        ("bool", torch.bool),
    ],
)
def test_dtype_from_config_valid_torch_dtypes(dtype_name, expected):
    """Valid torch dtype names should resolve to torch.dtype objects"""
    result = dtype_from_config(dtype_name)
    assert result is expected
    assert isinstance(result, torch.dtype)


def test_dtype_from_config_invalid_string():
    """Unknown dtype strings should return None"""
    assert dtype_from_config("not_a_dtype") is None


def test_dtype_from_config_case_sensitive():
    """Torch dtype lookup is case-sensitive"""
    assert dtype_from_config("Float32") is None
    assert dtype_from_config("FLOAT32") is None


def test_dtype_from_config_empty_string():
    """Empty string should be treated as invalid"""
    assert dtype_from_config("") is None


def test_dtype_from_config_whitespace():
    """Whitespace is not stripped and should fail lookup"""
    assert dtype_from_config(" float32 ") is None
