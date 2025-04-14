import pytest

from pytidb import TiDBClient
from autoflow.models.embedding_models import EmbeddingModel
from autoflow.storage.doc_store.tidb_doc_store import TiDBDocumentStore
from autoflow.storage.doc_store.types import Document, Chunk
from autoflow.utils.hash import sha256


@pytest.fixture(scope="session")
def doc_store():
    tidb_client = TiDBClient.connect()
    return TiDBDocumentStore(namespace="doc_store", client=tidb_client, vector_dims=3)


@pytest.fixture(scope="session")
def doc_store_with_auto_embed():
    tidb_client = TiDBClient.connect()
    embedding_model = EmbeddingModel(model_name="text-embedding-3-small")
    return TiDBDocumentStore(
        namespace="doc_store_with_auto_embed",
        client=tidb_client,
        embedding_model=embedding_model,
    )


def test_crud(doc_store):
    doc_store.reset()

    # Create
    documents = doc_store.add(
        [
            Document(
                name="TiDB",
                content="TiDB is a distributed SQL database.",
                chunks=[
                    Chunk(
                        text="TiDB is a distributed SQL database.", text_vec=[1, 2, 3]
                    ),
                ],
            ),
            Document(
                name="TiKV",
                content="TiKV is a distributed key-value storage engine.",
                chunks=[
                    Chunk(
                        text="TiKV is a distributed key-value storage engine.",
                        text_vec=[4, 5, 6],
                    ),
                ],
            ),
            Document(
                name="TiFlash",
                content="TiFlash is a column-oriented storage engine.",
                chunks=[
                    Chunk(
                        text="TiFlash is a column-oriented storage engine.",
                        text_vec=[7, 8, 9],
                    ),
                ],
            ),
        ]
    )
    assert len(documents) == 3
    for doc in documents:
        assert doc.id is not None
        assert doc.created_at is not None
        assert len(doc.chunks) == 1

        chunk = doc.chunks[0]
        assert chunk.id is not None
        assert chunk.document_id == doc.id
        assert chunk.text == doc.content
        assert chunk.text_vec is not None
        assert len(chunk.text_vec) == 3

    # Retrieve - Vector Search
    results = doc_store.search([4, 5, 6], top_k=2)
    assert len(results.documents) == 2
    assert results.documents[0].name == "TiKV"
    assert results.chunks[0].score > 0

    # Update
    document_id = results.chunks[0].document_id
    old_chunk = results.chunks[0]
    old_vector_sha = sha256(str(old_chunk.text_vec))
    new_chunk = doc_store.update_chunk(
        old_chunk.id,
        {
            "text": "TiKV is a distributed key-value storage engine for TiDB.",
            "text_vec": [3, 6, 9],
        },
    )
    new_vector_sha = sha256(str(new_chunk.text_vec))
    assert new_vector_sha != old_vector_sha

    # Delete
    doc_store.delete_chunk(new_chunk.id)
    chunks = doc_store.list_doc_chunks(document_id)
    assert len(chunks) == 0


def test_crud_with_auto_embed(doc_store_with_auto_embed):
    doc_store_with_auto_embed.reset()

    # Create
    documents = doc_store_with_auto_embed.add(
        [
            Document(
                name="TiDB",
                content="TiDB is a distributed SQL database.",
                chunks=[
                    Chunk(text="TiDB is a distributed SQL database."),
                ],
            ),
            Document(
                name="TiKV",
                content="TiKV is a distributed key-value storage engine.",
                chunks=[
                    Chunk(text="TiKV is a distributed key-value storage engine."),
                ],
            ),
            Document(
                name="TiFlash",
                content="TiFlash is a column-oriented storage engine.",
                chunks=[
                    Chunk(text="TiFlash is a column-oriented storage engine."),
                ],
            ),
        ]
    )
    assert len(documents) == 3
    for doc in documents:
        assert doc.id is not None
        assert doc.created_at is not None
        assert len(doc.chunks) == 1

        chunk = doc.chunks[0]
        assert chunk.id is not None
        assert chunk.document_id == doc.id
        assert chunk.text == doc.content
        assert chunk.text_vec is not None
        assert len(chunk.text_vec) == 1536

    # Retrieve - Vector Search
    results = doc_store_with_auto_embed.search("tikv", top_k=2)
    assert len(results.documents) == 2
    assert results.documents[0].name == "TiKV"
    assert results.chunks[0].score > 0

    # Update
    document_id = results.chunks[0].document_id
    old_chunk = results.chunks[0]
    old_vector_sha = sha256(str(old_chunk.text_vec))
    new_chunk = doc_store_with_auto_embed.update_chunk(
        old_chunk.id,
        {"text": "TiKV is a distributed key-value storage engine for TiDB."},
    )
    new_vector_sha = sha256(str(new_chunk.text_vec))
    # To check the auto embedding_models is work on updating.
    assert new_vector_sha != old_vector_sha

    # Delete
    doc_store_with_auto_embed.delete_chunk(new_chunk.id)
    chunks = doc_store_with_auto_embed.list_doc_chunks(document_id)
    assert len(chunks) == 0
