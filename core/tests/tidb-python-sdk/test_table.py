import hashlib
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pytest
from pydantic import BaseModel

from autoflow.storage.tidb import TiDBClient
from autoflow.storage.tidb.schema import DistanceMetric, TableModel
from autoflow.storage.tidb.base import Base
from autoflow.storage.tidb.search import SIMILARITY_SCORE_LABEL

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def db() -> TiDBClient:
    return TiDBClient.connect(
        host=os.getenv("TIDB_HOST"),
        port=int(os.getenv("TIDB_PORT", "4000")),
        username=os.getenv("TIDB_USERNAME"),
        password=os.getenv("TIDB_PASSWORD"),
    )


def test_raw_sql(db: TiDBClient):
    result = db.execute("DROP TABLE IF EXISTS test_raw_sql;")
    assert result.success
    assert result.rowcount == 0

    result = db.execute("CREATE TABLE IF NOT EXISTS test_raw_sql(id int);")
    assert result.success
    assert result.rowcount == 0

    result = db.execute("CREATE TABLE test_raw_sql(id int);")
    assert not result.success
    assert result.rowcount == 0
    assert result.message is not None

    result = db.execute("INSERT INTO test_raw_sql VALUES (1), (2), (3);")
    assert result.success
    assert result.rowcount == 3

    result = db.query("SELECT id FROM test_raw_sql;")
    df = result.to_pandas()
    assert df.size == 3

    result = db.query("SELECT id FROM test_raw_sql;")
    rows = result.to_rows()
    ids = sorted([r[0] for r in rows])
    assert ids == [1, 2, 3]

    result = db.query("SELECT id FROM test_raw_sql;")
    list = result.to_list()
    ids = sorted([item["id"] for item in list])
    assert ids == [1, 2, 3]

    class Record(BaseModel):
        id: int

    result = db.query("SELECT id FROM test_raw_sql;")
    records = result.to_pydantic(Record)
    ids = sorted([r.id for r in records])
    assert ids == [1, 2, 3]

    result = db.query("SELECT COUNT(*) FROM test_raw_sql;")
    n = result.scalar()
    assert n == 3


# CRUD


def test_table_crud(db):
    from sqlalchemy import Integer, Column, VARCHAR
    from tidb_vector.sqlalchemy import VectorType

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
    from sqlalchemy import JSON, Integer, Column, Text

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
    from sqlalchemy import Column
    from tidb_vector.sqlalchemy import VectorType
    from sqlmodel import Field

    class Chunk(TableModel, table=True):
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

    # to_pydantic()
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

    # to_pandas()
    results = tbl.search([1, 2, 3]).limit(2).to_pandas()
    assert results.size > 0

    # to_list()
    results = tbl.search([1, 2, 3]).limit(2).to_list()
    assert len(results) > 0
    assert results[0]["text"] == "bar"
    assert results[0][SIMILARITY_SCORE_LABEL] == 1
    assert results[0]["score"] == 1
    assert results[0]["user_id"] == 2


# Auto embedding


def test_auto_embedding(db: TiDBClient):
    from sqlmodel import Field
    from autoflow.storage.tidb.embeddings.litellm import LiteLLMEmbeddingFunction

    text_embed_small = LiteLLMEmbeddingFunction("openai/text-embedding-3-small")

    class Chunk(TableModel, table=True):
        __tablename__ = "test_auto_embedding"
        id: int = Field(primary_key=True)
        text: str = Field()
        # FIXME: if using list[float], sqlmodel will report an error
        text_vec: Optional[Any] = text_embed_small.VectorField(source_field="text")
        user_id: int = Field()

    tbl = db.create_table(schema=Chunk)
    tbl.truncate()
    tbl.insert(Chunk(id=1, text="foo", user_id=1))
    tbl.bulk_insert(
        [
            Chunk(id=2, text="bar", user_id=2),
            Chunk(id=3, text="baz", user_id=3),
            Chunk(id=4, text="qux", user_id=4),
        ]
    )
    chunks = tbl.query(
        filters={
            "user_id": 3,
        }
    )
    assert len(chunks) == 1
    assert chunks[0].text == "baz"
    assert len(chunks[0].text_vec) == 1536

    results = tbl.search("bar").limit(1).to_pydantic(with_score=True)
    assert len(results) == 1
    assert results[0].id == 2
    assert results[0].text == "bar"
    assert results[0].similarity_score >= 0.9

    # Update,
    chunk = tbl.get(1)
    assert chunk.text_vec is not None
    original_vec_hash = hashlib.md5(chunk.text_vec).hexdigest()
    tbl.update(
        values={"text": "new foo"},
        filters={"id": 1},
    )
    chunk = tbl.get(1)
    assert chunk.text_vec is not None
    updated_vec_hash = hashlib.md5(chunk.text_vec).hexdigest()
    assert original_vec_hash != updated_vec_hash


# TODO: Reranking
