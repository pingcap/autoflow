from .base import FileStorage
from .local import LocalFileStorage


def get_file_storage() -> FileStorage:
    return LocalFileStorage()


default_file_storage = get_file_storage()
