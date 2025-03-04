import os
import uuid

from typing import Dict, Any, Optional
from sqlalchemy import Text, JSON, create_engine
from sqlmodel import SQLModel, Field, Session


class TestDocument(SQLModel, table=True):
    __tablename__ = "test_documents"

    id: int = Field(primary_key=True)
    name: str = Field(nullable=False)


class TestBaseChunk(SQLModel):
    # __table_args__ = {"extend_existing": True}
    __abstract__ = True

    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    hash: Optional[str] = Field(max_length=64)
    text: str = Field(sa_type=Text)
    meta: Optional[Dict[str, Any]] = Field(default_factory=dict, sa_type=JSON)
    # embedding: Optional[Any] = Field(
    #     default_factory=None, sa_column=Column(VectorType())
    # )
    document_id: Optional[int] = Field(default=None, foreign_key="documents.id")


def create_chunk_model(table_name: str):
    # raise error: <class 'dict'> has no matching SQLAlchemy type
    class TestChunk(TestBaseChunk, table=True):
        __tablename__ = table_name

    return TestChunk


def test_extend_a_base_sqlmodel():
    engine = create_engine(
        os.getenv("DATABASE_URL"),
        pool_size=20,
        max_overflow=40,
        pool_recycle=300,
        pool_pre_ping=True,
    )
    chunk_model_2 = create_chunk_model("chunk_test_2")
    chunk_model_1 = create_chunk_model("chunk_test_1")

    SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)

    with Session(engine) as session:
        chunk_1 = chunk_model_1(id=uuid.uuid4(), text="foo")
        chunk_2 = chunk_model_2(id=uuid.uuid4(), text="bar")
        session.add(chunk_1)
        session.add(chunk_2)
        session.commit()
        session.refresh(chunk_1)
        session.refresh(chunk_2)

        print(chunk_1.model_dump())
        print(chunk_2.model_dump())


test_extend_a_base_sqlmodel()
