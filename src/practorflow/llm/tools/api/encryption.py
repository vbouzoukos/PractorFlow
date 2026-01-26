"""
Secret encryption module for API Tools.

Provides AES-GCM authenticated encryption for storing secrets securely.
Uses a singleton pattern - must be initialized with a key before use.
"""

import base64
import hashlib
from typing import Optional

from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes


class EncryptionError(Exception):
    """Raised when encryption or decryption fails."""
    
    def __init__(self, message: str = "Encryption operation failed"):
        super().__init__(message)


class EncryptionNotInitializedError(Exception):
    """Raised when encryption service is used before initialization."""
    
    def __init__(self, message: str = "Encryption service not initialized. Call initialize_encryption() first."):
        super().__init__(message)


class EncryptionService:
    """
    AES-GCM encryption service for secret storage.
    
    Uses 256-bit AES in GCM mode for authenticated encryption.
    Key is derived from the provided secret using PBKDF2.
    """
    
    _KEY_LENGTH = 32  # 256 bits
    _NONCE_LENGTH = 12  # 96 bits (recommended for GCM)
    _TAG_LENGTH = 16  # 128 bits
    _SALT_LENGTH = 16  # 128 bits
    _PBKDF2_ITERATIONS = 100_000
    
    def __init__(self, key: str):
        """
        Initialize encryption service with a secret key.
        
        Args:
            key: Secret key string (e.g., JWT secret key).
                 Will be derived into a 256-bit AES key.
        
        Raises:
            ValueError: If key is empty or too short.
        """
        if not key or len(key) < 16:
            raise ValueError("Encryption key must be at least 16 characters")
        self._master_key = key.encode("utf-8")
    
    def _derive_key(self, salt: bytes) -> bytes:
        """
        Derive a 256-bit AES key from the master key using PBKDF2.
        
        Args:
            salt: Random salt for key derivation.
        
        Returns:
            Derived 256-bit key.
        """
        return hashlib.pbkdf2_hmac(
            "sha256",
            self._master_key,
            salt,
            self._PBKDF2_ITERATIONS,
            dklen=self._KEY_LENGTH,
        )
    
    def encrypt(self, plaintext: str) -> str:
        """
        Encrypt a plaintext string.
        
        Format: base64(salt + nonce + tag + ciphertext)
        
        Args:
            plaintext: String to encrypt.
        
        Returns:
            Base64-encoded encrypted string.
        
        Raises:
            EncryptionError: If encryption fails.
        """
        if not plaintext:
            raise EncryptionError("Cannot encrypt empty string")
        
        try:
            salt = get_random_bytes(self._SALT_LENGTH)
            nonce = get_random_bytes(self._NONCE_LENGTH)
            key = self._derive_key(salt)
            
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            ciphertext, tag = cipher.encrypt_and_digest(plaintext.encode("utf-8"))
            
            encrypted_data = salt + nonce + tag + ciphertext
            return base64.b64encode(encrypted_data).decode("utf-8")
        except Exception as e:
            raise EncryptionError(f"Encryption failed: {e}") from e
    
    def decrypt(self, encrypted: str) -> str:
        """
        Decrypt an encrypted string.
        
        Args:
            encrypted: Base64-encoded encrypted string.
        
        Returns:
            Decrypted plaintext string.
        
        Raises:
            EncryptionError: If decryption fails or data is tampered.
        """
        if not encrypted:
            raise EncryptionError("Cannot decrypt empty string")
        
        try:
            encrypted_data = base64.b64decode(encrypted.encode("utf-8"))
            
            min_length = self._SALT_LENGTH + self._NONCE_LENGTH + self._TAG_LENGTH + 1
            if len(encrypted_data) < min_length:
                raise EncryptionError("Invalid encrypted data: too short")
            
            salt = encrypted_data[:self._SALT_LENGTH]
            nonce = encrypted_data[self._SALT_LENGTH:self._SALT_LENGTH + self._NONCE_LENGTH]
            tag = encrypted_data[self._SALT_LENGTH + self._NONCE_LENGTH:self._SALT_LENGTH + self._NONCE_LENGTH + self._TAG_LENGTH]
            ciphertext = encrypted_data[self._SALT_LENGTH + self._NONCE_LENGTH + self._TAG_LENGTH:]
            
            key = self._derive_key(salt)
            
            cipher = AES.new(key, AES.MODE_GCM, nonce=nonce)
            plaintext = cipher.decrypt_and_verify(ciphertext, tag)
            
            return plaintext.decode("utf-8")
        except EncryptionError:
            raise
        except Exception as e:
            raise EncryptionError(f"Decryption failed: {e}") from e


# Singleton instance
_encryption_service: Optional[EncryptionService] = None


def initialize_encryption(key: str) -> None:
    """
    Initialize the global encryption service singleton.
    
    Must be called before using get_encryption_service().
    Typically called at application startup with the JWT secret key.
    
    Args:
        key: Secret key for encryption (e.g., JWT secret key).
    
    Raises:
        ValueError: If key is invalid.
    """
    global _encryption_service
    _encryption_service = EncryptionService(key)


def get_encryption_service() -> EncryptionService:
    """
    Get the global encryption service singleton.
    
    Returns:
        Initialized EncryptionService instance.
    
    Raises:
        EncryptionNotInitializedError: If initialize_encryption() was not called.
    """
    if _encryption_service is None:
        raise EncryptionNotInitializedError(
            "Encryption service not initialized. Call initialize_encryption() first."
        )
    return _encryption_service


def is_encryption_initialized() -> bool:
    """
    Check if the encryption service has been initialized.
    
    Returns:
        True if initialized, False otherwise.
    """
    return _encryption_service is not None