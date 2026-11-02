"""Versioned, authenticated encryption for provider credentials."""

import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

MAGIC = b"AFGCM2\x00"
MAX_PLAINTEXT_BYTES = 60 * 1024


class AESCipher:
    def __init__(self, key: bytes, *, context: str = "provider.credentials"):
        if len(key) != 32:
            raise ValueError("A 32-byte data-encryption key is required")
        derived = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b"autoflow/provider-credentials/v2",
        ).derive(key)
        self._cipher = AESGCM(derived)
        self._aad = MAGIC + context.encode("utf-8")

    def encrypt(self, plain_text: str) -> bytes:
        data = plain_text.encode("utf-8")
        if len(data) > MAX_PLAINTEXT_BYTES:
            raise ValueError("Credential value is too large")
        nonce = os.urandom(12)
        return MAGIC + nonce + self._cipher.encrypt(nonce, data, self._aad)

    def decrypt(self, encrypted_text: bytes) -> str:
        data = bytes(encrypted_text)
        if not data.startswith(MAGIC):
            raise ValueError(
                "Legacy provider credentials require the explicit migration tool"
            )
        if not len(MAGIC) + 28 <= len(data) <= len(MAGIC) + 28 + MAX_PLAINTEXT_BYTES:
            raise ValueError("Invalid encrypted credential")
        try:
            offset = len(MAGIC)
            return self._cipher.decrypt(
                data[offset : offset + 12], data[offset + 12 :], self._aad
            ).decode("utf-8")
        except (InvalidTag, UnicodeDecodeError, ValueError):
            raise ValueError("Credential authentication failed") from None
