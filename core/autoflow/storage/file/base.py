from typing import IO

from abc import ABC, abstractmethod


class FileStorage(ABC):
    @abstractmethod
    def open(self, name: str, mode: str = "rb") -> IO:
        raise NotImplementedError

    @abstractmethod
    def save(self, name: str, content: IO) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, name: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def exists(self, name: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def size(self, name: str) -> int:
        raise NotImplementedError
