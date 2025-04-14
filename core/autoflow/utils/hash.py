import hashlib
from typing import Optional


def sha256(text: Optional[str]) -> Optional[str]:
    return hashlib.sha256(text.encode("utf-8")).hexdigest() if text else None
