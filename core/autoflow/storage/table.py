from typing import Type

from sqlalchemy import Engine
from sqlmodel import SQLModel


class Table:
    def __init__(self, engine: Engine, schema: SQLModel):
        self._engine = engine
        self._model = schema

    def get(self, id: int):
        raise NotImplementedError

    def update(self, obj: Type[SQLModel], update: dict):
        raise NotImplementedError

    def search(self):
        raise NotImplementedError

    def delete(self):
        raise NotImplementedError
