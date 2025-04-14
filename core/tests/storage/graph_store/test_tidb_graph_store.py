from hashlib import sha256
import logging
import pytest

from autoflow.storage.graph_store import TiDBGraphStore
from autoflow.storage.graph_store.types import (
    EntityType,
    EntityUpdate,
    RelationshipUpdate,
)


logger = logging.getLogger(__name__)


@pytest.fixture(scope="session")
def graph_store(tidb_client, embedding_model):
    return TiDBGraphStore(
        client=tidb_client,
        embedding_model=embedding_model,
        namespace="tidb_graph_store_test",
    )


def test_entity_crud(graph_store: TiDBGraphStore):
    graph_store.reset()

    # Create entities
    tidb_entity = graph_store.create_entity(
        name="TiDB", description="TiDB is a relational database."
    )
    assert tidb_entity.id is not None
    logger.info(
        "tidb_graph_store: add new entity (name: %s, id: %s)",
        tidb_entity.name,
        tidb_entity.id,
    )

    tikv_entity = graph_store.create_entity(
        name="TiKV", description="TiKV is a distributed key-value storage engine."
    )
    assert tikv_entity.id is not None
    logger.info(
        "tidb_graph_store: add new entity (name: %s, id: %s)",
        tikv_entity.name,
        tikv_entity.id,
    )

    # Get entity
    entity = graph_store.get_entity(tidb_entity.id)
    assert entity.id is not None
    assert entity.entity_type == EntityType.original
    assert entity.name == "TiDB"
    assert entity.embedding is not None
    assert entity.created_at is not None
    assert entity.updated_at is not None

    # List entities
    entities = graph_store.list_entities(entity_id=tidb_entity.id)
    assert len(entities) == 1
    assert entities[0].id == tidb_entity.id

    entities = graph_store.list_entities(entity_id=[tidb_entity.id])
    assert len(entities) == 1
    assert entities[0].id == tidb_entity.id

    entities = graph_store.list_entities(entity_type=EntityType.original)
    assert len(entities) == 2

    # Update entity
    old_embedding = tidb_entity.embedding
    updated_tidb_entity = graph_store.update_entity(
        entity=tidb_entity,
        update=EntityUpdate(
            name="TiDB", description="TiDB is a MySQL-compatible database."
        ),
    )
    new_embedding = updated_tidb_entity.embedding
    assert updated_tidb_entity.id == tidb_entity.id
    assert updated_tidb_entity.name == "TiDB"
    assert updated_tidb_entity.description == "TiDB is a MySQL-compatible database."
    assert sha256(new_embedding) != sha256(old_embedding)

    # Delete entity
    graph_store.delete_entity(tidb_entity.id)
    try:
        graph_store.get_entity(tidb_entity.id)
        raise AssertionError("Entity should be deleted")
    except Exception as e:
        logger.info(
            "tidb_graph_store: entity %s should be deleted: %s", tidb_entity.id, e
        )

    graph_store.reset()


def test_relationship_crud(graph_store: TiDBGraphStore):
    graph_store.reset()

    # Create entities
    tidb_entity = graph_store.create_entity(
        name="TiDB", description="TiDB is a relational database."
    )
    tikv_entity = graph_store.create_entity(
        name="TiKV", description="TiKV is a distributed key-value storage engine."
    )

    # Create relationships
    relationship = graph_store.create_relationship(
        source_entity=tidb_entity,
        target_entity=tikv_entity,
        description="TiDB uses TiKV as its storage engine.",
        meta={
            "source": "TiDB's Documentation",
        },
    )
    assert relationship.id is not None
    assert relationship.source_entity_id == tidb_entity.id
    assert relationship.target_entity_id == tikv_entity.id
    assert relationship.description == "TiDB uses TiKV as its storage engine."
    assert relationship.embedding is not None
    assert relationship.created_at is not None
    assert relationship.updated_at is not None

    # List relationships
    relationships = graph_store.list_relationships(entity_id=tidb_entity.id)
    assert len(relationships) == 1
    assert relationships[0].id == relationship.id

    # Update relationship
    old_embedding = relationship.embedding
    old_updated_at = relationship.updated_at
    updated_relationship = graph_store.update_relationship(
        relationship=relationship,
        update=RelationshipUpdate(
            description="TiDB uses TiKV as its storage engine for TP workloads."
        ),
    )
    new_embedding = updated_relationship.embedding
    new_updated_at = updated_relationship.updated_at
    assert (
        updated_relationship.description
        == "TiDB uses TiKV as its storage engine for TP workloads."
    )
    assert sha256(new_embedding) != sha256(old_embedding)
    assert new_updated_at > old_updated_at

    # Delete relationship
    graph_store.delete_relationship(relationship.id)
    try:
        graph_store.get_relationship(relationship.id)
        raise AssertionError("Relationship should be deleted")
    except Exception as e:
        logger.info(
            "tidb_graph_store: relationship %s should be deleted: %s",
            relationship.id,
            e,
        )

    graph_store.reset()


def test_entity_degree(graph_store: TiDBGraphStore):
    graph_store.reset()

    # Create entities
    tidb_entity = graph_store.create_entity(
        name="TiDB", description="TiDB is a relational database."
    )
    tikv_entity = graph_store.create_entity(
        name="TiKV", description="TiKV is a distributed key-value storage engine."
    )
    tiflash_entity = graph_store.create_entity(
        name="TiFlash", description="TiFlash is a column-oriented database engine."
    )

    # Create relationships
    graph_store.create_relationship(
        source_entity=tidb_entity,
        target_entity=tikv_entity,
        description="TiDB uses TiKV as its storage engine.",
    )
    graph_store.create_relationship(
        source_entity=tidb_entity,
        target_entity=tiflash_entity,
        description="TiDB uses TiFlash as its analytical engine.",
    )

    # Calculate entity degree
    out_degree = graph_store.calc_entity_out_degree(tidb_entity.id)
    assert out_degree == 2

    in_degree = graph_store.calc_entity_in_degree(tidb_entity.id)
    assert in_degree == 0

    degree = graph_store.calc_entity_degree(tidb_entity.id)
    assert degree == 2

    # Calculate entities degree
    degrees = graph_store.calc_entities_degrees(
        [tidb_entity.id, tikv_entity.id, tiflash_entity.id]
    )
    assert degrees[tidb_entity.id].out_degree == 2
    assert degrees[tidb_entity.id].in_degree == 0
    assert degrees[tidb_entity.id].degrees == 2

    assert degrees[tikv_entity.id].out_degree == 0
    assert degrees[tikv_entity.id].in_degree == 1
    assert degrees[tikv_entity.id].degrees == 1

    assert degrees[tiflash_entity.id].out_degree == 0
    assert degrees[tiflash_entity.id].in_degree == 1
    assert degrees[tiflash_entity.id].degrees == 1

    graph_store.reset()
