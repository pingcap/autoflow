"""Legacy decoding for the explicit offline migration; never used by request paths."""

from cryptography.hazmat.decrepit.ciphers.modes import CFB
from cryptography.hazmat.primitives import padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms


def decrypt_legacy_credentials(data: bytes, signing_secret: bytes) -> str:
    if len(signing_secret) < 32 or len(data) < 32 or len(data) % 16:
        raise ValueError("Invalid legacy credential or key")
    try:
        decryptor = Cipher(
            algorithms.AES(signing_secret[:32]), CFB(data[:16])
        ).decryptor()
        padded = decryptor.update(data[16:]) + decryptor.finalize()
        unpadder = padding.PKCS7(128).unpadder()
        return (unpadder.update(padded) + unpadder.finalize()).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        raise ValueError("Legacy credential could not be decrypted") from None
