import uuid
from typing import List, Dict

from sqlmodel import Field
from tidb_vector.sqlalchemy import VectorType
from autoflow.storage.tidb.base import TiDBModel


class Memory(TiDBModel, table=True):
    __tablename__ = "agent_memory"
    id: uuid.UUID = Field(None, primary_key=True)
    vector: List[float] = Field(None, sa_type=VectorType)
    payload: Dict = Field(None)
