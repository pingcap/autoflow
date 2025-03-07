from pydantic import PrivateAttr
from sqlalchemy.orm import registry
from sqlmodel import SQLModel

default_registry = registry()

Base = default_registry.generate_base()


class TiDBModel(SQLModel):
    # Generated on query
    _similarity_score: float = PrivateAttr(None)
    _score: float = PrivateAttr(None)

    @property
    def similarity_score(self) -> float:
        return self._similarity_score

    @similarity_score.setter
    def similarity_score(self, val: float):
        self._similarity_score = val

    @property
    def score(self) -> float:
        return self._score

    @score.setter
    def score(self, val: float):
        self._score = val
