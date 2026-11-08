import os
from pathlib import Path
from typing import IO

from app.file_storage.base import FileStorage
from app.core.config import settings


class LocalFileStorage(FileStorage):
    def path(self, name: str) -> str:
        root = Path(settings.LOCAL_FILE_STORAGE_PATH).resolve()
        path = (root / name).resolve()
        if Path(name).is_absolute() or path == root or not path.is_relative_to(root):
            raise ValueError("Invalid storage path")
        return str(path)

    def open(self, name: str, mode: str = "rb") -> IO:
        return open(self.path(name), mode)

    def save(self, name: str, content: IO) -> None:
        path = self.path(name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(content.read())

    def delete(self, name: str) -> None:
        os.remove(self.path(name))

    def exists(self, name: str) -> bool:
        return os.path.exists(self.path(name))

    def size(self, name: str) -> int:
        return os.path.getsize(self.path(name))
