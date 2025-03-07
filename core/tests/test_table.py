import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from sqlalchemy import JSON, Integer, Column, Text, VARCHAR
from sqlmodel import Field
from tidb_vector.sqlalchemy import VectorType
from autoflow.storage.tidb import TiDBClient, Base
from autoflow.storage.tidb.base import TiDBModel
from autoflow.storage.tidb.constants import DistanceMetric
from autoflow.storage.tidb.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def db() -> TiDBClient:
    return TiDBClient.connect(os.getenv("DATABASE_URL"))


# CRUD


def test_table_crud(db):
    table_name = "test_get_data"
    db.drop_table(table_name)

    class Chunk(Base):
        __tablename__ = table_name
        id = Column(Integer, primary_key=True)
        text = Column(VARCHAR(10))
        text_vec = Column(VectorType(dim=3))

    tbl = db.create_table(schema=Chunk)

    # CREATE
    tbl.insert(Chunk(id=1, text="foo", text_vec=[1, 2, 3]))
    tbl.insert(Chunk(id=2, text="bar", text_vec=[4, 5, 6]))
    tbl.insert(Chunk(id=3, text="biz", text_vec=[7, 8, 9]))

    # RETRIEVE
    c = tbl.get(1)
    assert np.array_equal(c.text_vec, [1, 2, 3])

    # UPDATE
    tbl.update(
        values={
            "text": "fooooooo",
            "text_vec": [3, 6, 9],
        },
        filters={"text": "foo"},
    )
    c = tbl.get(1)
    assert c.text == "fooooooo"
    assert np.array_equal(c.text_vec, [3, 6, 9])

    # DELETE
    tbl.delete(filters={"id": {"$in": [1, 2]}})
    assert tbl.rows() == 1


# Test filters


@pytest.fixture(scope="module")
def table_for_test_filters(db):
    table_name = "test_query_data"
    db.drop_table(table_name)

    class Chunk(Base):
        __tablename__ = table_name
        id = Column(Integer, primary_key=True)
        text = Column(Text)
        document_id = Column(Integer)
        meta = Column(JSON)

    tbl = db.create_table(schema=Chunk)
    Base.metadata.create_all(db.db_engine)

    test_data = [
        Chunk(
            id=1,
            text="foo",
            document_id=1,
            meta={"f": 0.2, "s": "apple", "a": [1, 2, 3]},
        ),
        Chunk(
            id=2,
            text="bar",
            document_id=2,
            meta={"f": 0.5, "s": "banana", "a": [4, 5, 6]},
        ),
        Chunk(
            id=3,
            text="biz",
            document_id=2,
            meta={"f": 0.7, "s": "cherry", "a": [7, 8, 9]},
        ),
    ]

    tbl.bulk_insert(test_data)
    yield tbl
    db.drop_table(table_name)


filter_test_data = [
    pytest.param({"document_id": 1}, ["foo"], id="implicit $eq operator"),
    pytest.param({"document_id": {"$eq": 1}}, ["foo"], id="explicit $eq operator"),
    pytest.param({"id": {"$in": [2, 3]}}, ["bar", "biz"], id="$in operator"),
    pytest.param({"id": {"$nin": [2, 3]}}, ["foo"], id="$nin operator"),
    pytest.param({"id": {"$gte": 2}}, ["bar", "biz"], id="$gte operator"),
    pytest.param({"id": {"$lt": 2}}, ["foo"], id="$lt operator"),
    pytest.param(
        {"$and": [{"document_id": 2}, {"id": {"$gt": 2}}]}, ["biz"], id="$and operator"
    ),
    pytest.param(
        {"$or": [{"document_id": 1}, {"id": 3}]}, ["foo", "biz"], id="$or operator"
    ),
    pytest.param(
        {"meta.f": {"$gte": 0.5}}, ["bar", "biz"], id="json column: $gt operator"
    ),
    pytest.param({"meta.s": {"$eq": "apple"}}, ["foo"], id="json column: $eq operator"),
]


@pytest.mark.parametrize(
    "filters,expected",
    filter_test_data,
)
def test_filters(table_for_test_filters, filters: Dict[str, Any], expected: List[str]):
    tbl = table_for_test_filters
    result = tbl.query(filters)

    actual = [r.text for r in result]
    assert actual == expected


# Vector Search


def test_vector_search(db: TiDBClient):
    class Chunk(TiDBModel, table=True):
        __tablename__ = "test_vector_search"
        id: int = Field(None, primary_key=True)
        text: str = Field(None)
        text_vec: Any = Field(sa_column=Column(VectorType(3)))
        user_id: int = Field(None)

    tbl = db.create_table(schema=Chunk)
    tbl.truncate()
    tbl.bulk_insert(
        [
            Chunk(id=1, text="foo", text_vec=[4, 5, 6], user_id=1),
            Chunk(id=2, text="bar", text_vec=[1, 2, 3], user_id=2),
            Chunk(id=3, text="biz", text_vec=[7, 8, 9], user_id=3),
        ]
    )
    results = (
        tbl.search([1, 2, 3])
        .distance_metric(metric=DistanceMetric.COSINE)
        .num_candidate(20)
        .filter({"user_id": 2})
        .limit(2)
        .to_pydantic()
    )
    assert len(results) == 1
    assert results[0].text == "bar"
    assert results[0].similarity_score == 1
    assert results[0].score == 1
    assert results[0].user_id == 2

    for r in results:
        logger.info(f"{r.id} {r.text} {r.similarity_score} {r.score}")


# Auto embedding


def test_auto_embedding(db: TiDBClient):
    embed_model = EmbeddingModel("openai/text-embedding-3-small")

    class Chunk(TiDBModel, table=True):
        __tablename__ = "test_auto_embedding"
        id: int = Field(None, primary_key=True)
        text: str = embed_model.SourceField()
        text_vec: Optional[Any] = embed_model.VectorField()
        user_id: int = Field(None)

    tbl = db.create_table(schema=Chunk, embed_model=embed_model)
    tbl.truncate()
    tbl.bulk_insert(
        [
            Chunk(id=1, text="foo", user_id=1),
            Chunk(id=2, text="bar", user_id=2),
            Chunk(id=3, text="biz", user_id=3),
        ]
    )
    results = tbl.search("bar").limit(2).to_pydantic()
    assert len(results) == 1
    assert results[0].text == "bar"
