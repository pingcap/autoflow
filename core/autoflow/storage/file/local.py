import os
from typing import IO

from autoflow.storage.file import FileStorage


class LocalFileStorage(FileStorage):
    def path(self, name: str) -> str:
        return os.path.join("", name)

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
