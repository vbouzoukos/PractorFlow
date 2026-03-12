"""Tests for encryption module."""

import base64

import pytest

import practorflow.llm.tools.api.encryption as _enc_module
from practorflow.llm.tools.api.encryption import (
    EncryptionError,
    EncryptionNotInitializedError,
    EncryptionService,
    get_encryption_service,
    initialize_encryption,
    is_encryption_initialized,
)


@pytest.fixture(autouse=True)
def reset_singleton():
    original = _enc_module._encryption_service
    _enc_module._encryption_service = None
    yield
    _enc_module._encryption_service = original


# EncryptionError

def test_encryption_error_default_message():
    err = EncryptionError()
    assert "Encryption operation failed" in str(err)


def test_encryption_error_custom_message():
    err = EncryptionError("custom error")
    assert "custom error" in str(err)


# EncryptionNotInitializedError

def test_encryption_not_initialized_error_default_message():
    err = EncryptionNotInitializedError()
    assert "not initialized" in str(err)


def test_encryption_not_initialized_error_custom_message():
    err = EncryptionNotInitializedError("must init first")
    assert "must init first" in str(err)


# EncryptionService.__init__

def test_encryption_service_init_valid_key():
    svc = EncryptionService("a" * 16)
    assert svc._master_key == ("a" * 16).encode("utf-8")


def test_encryption_service_init_empty_key_raises():
    with pytest.raises(ValueError, match="at least 16 characters"):
        EncryptionService("")


def test_encryption_service_init_short_key_raises():
    with pytest.raises(ValueError, match="at least 16 characters"):
        EncryptionService("short_key")


# encrypt

def test_encrypt_empty_string_raises():
    svc = EncryptionService("a_valid_key_16ch")
    with pytest.raises(EncryptionError, match="empty"):
        svc.encrypt("")


def test_encrypt_returns_base64_string():
    svc = EncryptionService("a_valid_key_16ch")
    result = svc.encrypt("hello world")
    assert isinstance(result, str)
    assert len(result) > 0


def test_encrypt_different_calls_produce_different_ciphertext():
    svc = EncryptionService("a_valid_key_16ch")
    enc1 = svc.encrypt("same plaintext")
    enc2 = svc.encrypt("same plaintext")
    assert enc1 != enc2  # Random nonce/salt


# decrypt

def test_decrypt_empty_string_raises():
    svc = EncryptionService("a_valid_key_16ch")
    with pytest.raises(EncryptionError, match="empty"):
        svc.decrypt("")


def test_encrypt_decrypt_roundtrip():
    svc = EncryptionService("my_secret_key_abc")
    plaintext = "my secret api key"
    encrypted = svc.encrypt(plaintext)
    decrypted = svc.decrypt(encrypted)
    assert decrypted == plaintext


def test_decrypt_too_short_data_raises():
    svc = EncryptionService("a_valid_key_16ch")
    short_data = base64.b64encode(b"x" * 10).decode("utf-8")
    with pytest.raises(EncryptionError):
        svc.decrypt(short_data)


def test_decrypt_tampered_data_raises():
    svc = EncryptionService("a_valid_key_16ch")
    encrypted = svc.encrypt("secret value")
    data = bytearray(base64.b64decode(encrypted))
    data[-1] ^= 0xFF
    tampered = base64.b64encode(bytes(data)).decode("utf-8")
    with pytest.raises(EncryptionError):
        svc.decrypt(tampered)


def test_decrypt_wrong_key_raises():
    svc1 = EncryptionService("a_valid_key_16ch")
    svc2 = EncryptionService("different_key_ab")
    encrypted = svc1.encrypt("secret")
    with pytest.raises(EncryptionError):
        svc2.decrypt(encrypted)


# Global functions

def test_is_encryption_initialized_false_when_not_initialized():
    assert is_encryption_initialized() is False


def test_is_encryption_initialized_true_after_init():
    initialize_encryption("a_valid_key_16ch")
    assert is_encryption_initialized() is True


def test_initialize_encryption_sets_service():
    initialize_encryption("a_valid_key_16ch")
    assert _enc_module._encryption_service is not None


def test_get_encryption_service_raises_when_not_initialized():
    with pytest.raises(EncryptionNotInitializedError):
        get_encryption_service()


def test_get_encryption_service_returns_service_when_initialized():
    initialize_encryption("a_valid_key_16ch")
    svc = get_encryption_service()
    assert isinstance(svc, EncryptionService)
