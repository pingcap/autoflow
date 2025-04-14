import logging
from typing import Collection, Dict, List, Optional, Tuple, Type, Any
from uuid import UUID

from pydantic import PrivateAttr
from pytidb import Table, TiDBClient
from pytidb.datatype import JSON, Text
from pytidb.schema import (
    Column,
    Field,
    Relationship as SQLRelationship,
    TableModel,
    VectorField,
)
from pytidb.sql import func, select, or_
from pytidb.embeddings import EmbeddingFunction
from sqlalchemy import Index

from autoflow.models.embedding_models import EmbeddingModel
from autoflow.orms.base import UUIDBaseModel
from autoflow.storage.graph_store.base import GraphStore
from autoflow.storage.graph_store.types import (
    Entity,
    EntityDegree,
    EntityFilters,
    EntityType,
    EntityUpdate,
    KnowledgeGraph,
    KnowledgeGraphCreate,
    Relationship,
    RelationshipFilters,
    RelationshipUpdate,
)
from autoflow.storage.types import QueryBundle


logger = logging.getLogger(__name__)


def dynamic_create_models(
    namespace: Optional[str] = None,
    embedding_model: Optional[EmbeddingModel] = None,
    vector_dims: Optional[int] = None,
) -> tuple[Type[TableModel], Type[TableModel]]:
    if embedding_model is None and vector_dims is None:
        raise ValueError("Either `embedding_model` or `vector_dims` must be specified")

    # Determine the table names.
    suffix = f"_{namespace}" if namespace else ""
    entity_table_name = f"entities{suffix}"
    relationship_table_name = f"relationships{suffix}"
    entity_model_name = f"DBEntity{suffix}"
    relationship_model_name = f"DBRelationship{suffix}"

    # Embedding function.
    embed_fn = None
    if embedding_model is not None:
        embed_fn = EmbeddingFunction(
            model_name=embedding_model.model_name,
            dimensions=embedding_model.dimensions,
            api_key=embedding_model.api_key,
            api_base=embedding_model.api_base,
            timeout=embedding_model.timeout,
        )

    # Initialize the entity model.
    if embed_fn is not None:
        entity_vector_field = embed_fn.VectorField()
    else:
        entity_vector_field = VectorField(vector_dims)

    class DBEntity(UUIDBaseModel):
        __table_args__ = (
            Index("idx_entity_type", "entity_type"),
            Index("idx_entity_name", "name"),
        )
        entity_type: EntityType = EntityType.original
        name: str = Field(max_length=512)
        description: str = Field(sa_column=Column(Text))
        meta: Optional[Dict] = Field(default_factory=dict, sa_column=Column(JSON))
        embedding: Optional[Any] = entity_vector_field

        def __hash__(self):
            return hash(self.id)

        def __eq__(self, other):
            return self.id == other.id

    entity_model = type(
        entity_model_name,
        (DBEntity,),
        {
            "__tablename__": entity_table_name,
            "__table_args__": {
                "extend_existing": True,
            },
        },
        table=True,
    )

    # Initialize the relationship model.
    if embed_fn is not None:
        relationship_vector_field = embed_fn.VectorField()
    else:
        relationship_vector_field = VectorField(vector_dims)

    class DBRelationship(UUIDBaseModel):
        description: str = Field(sa_column=Column(Text))
        source_entity_id: UUID = Field(foreign_key=f"{entity_table_name}.id")
        target_entity_id: UUID = Field(foreign_key=f"{entity_table_name}.id")
        meta: Optional[Dict] = Field(default_factory=dict, sa_column=Column(JSON))
        embedding: Optional[Any] = relationship_vector_field
        weight: Optional[float] = Field(default=0)
        chunk_id: Optional[UUID] = Field(default=None)
        document_id: Optional[UUID] = Field(default=None)

        def __hash__(self):
            return hash(self.id)

        def __eq__(self, other):
            return self.id == other.id

    relationship_model = type(
        relationship_model_name,
        (DBRelationship,),
        {
            "__tablename__": relationship_table_name,
            "__table_args__": {"extend_existing": True},
            "__annotations__": {
                "source_entity": entity_model,
                "target_entity": entity_model,
            },
            "source_entity": SQLRelationship(
                sa_relationship_kwargs={
                    "primaryjoin": f"{relationship_model_name}.source_entity_id == {entity_model_name}.id",
                    "lazy": "joined",
                },
            ),
            "target_entity": SQLRelationship(
                sa_relationship_kwargs={
                    "primaryjoin": f"{relationship_model_name}.target_entity_id == {entity_model_name}.id",
                    "lazy": "joined",
                },
            ),
        },
        table=True,
    )

    return entity_model, relationship_model


class TiDBGraphStore(GraphStore):
    _db: TiDBClient = PrivateAttr()
    _entity_db_model: Type[TableModel] = PrivateAttr()
    _entity_table: Table = PrivateAttr()
    _relationship_db_model: Type[TableModel] = PrivateAttr()
    _relationship_table: Table = PrivateAttr()

    def __init__(
        self,
        client: TiDBClient,
        namespace: Optional[str] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        vector_dims: Optional[int] = None,
        entity_distance_threshold: Optional[float] = 0.1,
    ):
        super().__init__()
        self._db = client
        self._db_engine = client.db_engine
        self._embedding_model = embedding_model
        self._entity_distance_threshold = entity_distance_threshold
        self._init_store(namespace, vector_dims)

    def _init_store(
        self, namespace: Optional[str] = None, vector_dims: Optional[int] = None
    ):
        self._entity_db_model, self._relationship_db_model = dynamic_create_models(
            namespace=namespace,
            vector_dims=vector_dims,
            embedding_model=self._embedding_model,
        )
        self._entity_table = self._db.create_table(schema=self._entity_db_model)
        self._relationship_table = self._db.create_table(
            schema=self._relationship_db_model
        )

    # Entity Basic Operations

    def get_entity(self, entity_id: UUID) -> Entity:
        return self._entity_table.get(entity_id)

    def list_entities(
        self, filters: Optional[EntityFilters] = EntityFilters(), **kwargs
    ) -> List[Entity]:
        if isinstance(kwargs, dict):
            filters = filters.model_copy(update=kwargs)
        filter_dict = self._convert_entity_filters(filters)
        return self._entity_table.query(filter_dict)

    def search_entities(
        self,
        query: QueryBundle,
        top_k: int = 10,
        num_candidate: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        filters: Optional[EntityFilters] = None,
    ) -> List[Tuple[Entity, float]]:
        filter_dict = self._convert_entity_filters(filters)
        results = (
            self._entity_table.search(query.query_embedding or query.query_str)
            .num_candidate(num_candidate or top_k * 10)
            .filter(filter_dict)
            .distance_threshold(distance_threshold)
            .limit(top_k)
            .to_pydantic()
        )
        return [(item.hit, item.score) for item in results]

    def _convert_entity_filters(self, filters: Optional[EntityFilters]) -> dict:
        filter_dict = {}
        if filters is None:
            return filter_dict
        if filters.entity_type:
            filter_dict["entity_type"] = filters.entity_type.value
        if filters.entity_id:
            op = "$in" if isinstance(filters.entity_id, list) else "$eq"
            filter_dict["id"] = {op: filters.entity_id}
        return filter_dict

    def create_entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.original,
        description: Optional[str] = None,
        meta: Optional[dict] = None,
        embedding: Optional[list[float]] = None,
    ) -> Entity:
        if embedding is None:
            embedding = self._get_entity_embedding(name, description)
        entity = self._entity_db_model(
            name=name,
            entity_type=entity_type,
            description=description,
            meta=meta,
            embedding=embedding,
        )
        return self._entity_table.insert(entity)

    def _get_entity_embedding(self, name: str, description: str) -> list[float]:
        embedding_str = f"{name}: {description}"
        return self._embedding_model.get_text_embedding(embedding_str)

    def find_or_create_entity(
        self,
        name: str,
        entity_type: EntityType = EntityType.original,
        description: Optional[str] = None,
        meta: Optional[dict] = None,
        embedding: Optional[Any] = None,
    ) -> Entity:
        query_embedding = self._get_entity_embedding(name, description)
        query = QueryBundle(query_embedding=query_embedding)
        nearest_entity = self.search_entities(
            query, top_k=1, distance_threshold=self._entity_distance_threshold
        )
        if len(nearest_entity) != 0:
            return nearest_entity[0][0]
        else:
            return self.create_entity(
                name=name,
                entity_type=entity_type,
                description=description,
                meta=meta,
                embedding=embedding,
            )

    def update_entity(self, entity: Entity | UUID, update: EntityUpdate) -> Entity:
        if isinstance(entity, UUID):
            entity = self.get_entity(entity)

        update_dict = update.model_dump(exclude_none=True)
        if update.embedding is None:
            update_dict["embedding"] = self._get_entity_embedding(
                entity.name, entity.description
            )

        self._entity_table.update(values=update_dict, filters={"id": entity.id})
        # FIXME: pytidb should return the updated entity.
        entity = self._entity_table.get(entity.id)

        # Update connected relationships.
        connected_relationships = self.list_relationships(
            filters=RelationshipFilters(
                entity_id=entity.id,
            )
        )
        for relationship in connected_relationships:
            self.update_relationship(relationship, RelationshipUpdate())

        return entity

    def delete_entity(self, entity_id: UUID) -> None:
        with self._db.session():
            # Delete all relationships connected to the entity.
            self._relationship_table.delete({"source_entity_id": entity_id})
            self._relationship_table.delete({"target_entity_id": entity_id})

            # Delete the entity.
            self._entity_table.delete({"id": entity_id})

    def delete_orphan_entities(self):
        raise NotImplementedError()

    # Entity Degree Operations

    def calc_entity_out_degree(self, entity_id: UUID) -> int:
        stmt = select(func.count(self._relationship_db_model.id)).where(
            self._relationship_db_model.source_entity_id == entity_id
        )
        return self._db.query(stmt).scalar()

    def calc_entity_in_degree(self, entity_id: UUID) -> int:
        stmt = select(func.count(self._relationship_db_model.id)).where(
            self._relationship_db_model.target_entity_id == entity_id
        )
        return self._db.query(stmt).scalar()

    def calc_entity_degree(self, entity_id: UUID) -> int:
        stmt = select(func.count(self._relationship_db_model.id)).where(
            or_(
                self._relationship_db_model.target_entity_id == entity_id,
                self._relationship_db_model.source_entity_id == entity_id,
            )
        )
        return self._db.query(stmt).scalar()

    def calc_entities_degrees(
        self, entity_ids: Collection[UUID]
    ) -> Dict[UUID, EntityDegree]:
        entity_table_name = self._entity_table.table_name
        relationship_table_name = self._relationship_table.table_name
        stmt = f"""
            SELECT
                e.id as id,
                COALESCE(SUM(CASE WHEN r.target_entity_id = e.id THEN 1 ELSE 0 END), 0) AS in_degree,
                COALESCE(SUM(CASE WHEN r.source_entity_id = e.id THEN 1 ELSE 0 END), 0) AS out_degree,
                COALESCE(COUNT(e.id), 0) AS degree
            FROM {entity_table_name} e
            LEFT JOIN {relationship_table_name} r ON e.id = r.source_entity_id OR e.id = r.target_entity_id
            WHERE e.id IN :entity_ids
            GROUP BY e.id
        """
        results = self._db.query(
            stmt, {"entity_ids": [entity_id.hex for entity_id in entity_ids]}
        ).to_list()
        return {
            UUID(item["id"]): EntityDegree(
                in_degree=item["in_degree"],
                out_degree=item["out_degree"],
                degrees=item["degree"],
            )
            for item in results
        }

    # Relationship Basic Operations

    def get_relationship(self, relationship_id: UUID) -> Relationship:
        return self._relationship_table.get(relationship_id)

    def list_relationships(
        self, filters: RelationshipFilters = RelationshipFilters(), **kwargs
    ) -> List[Relationship]:
        if isinstance(kwargs, dict):
            filters = filters.model_copy(update=kwargs)
        filter_dict = self._convert_relationship_filters(filters)
        return self._relationship_table.query(filter_dict)

    def search_relationships(
        self,
        query: QueryBundle,
        top_k: int = 10,
        num_candidate: Optional[int] = None,
        distance_threshold: Optional[float] = None,
        distance_range: Optional[Tuple[float, float]] = None,
        filters: Optional[RelationshipFilters] = None,
    ) -> List[Tuple[Relationship, float]]:
        filter_dict = self._convert_relationship_filters(filters)
        results = (
            self._relationship_table.search(query.query_embedding or query.query_str)
            .num_candidate(num_candidate or top_k * 10)
            .filter(filter_dict)
            .distance_threshold(distance_threshold)
            .distance_range(distance_range[0], distance_range[1])
            .limit(top_k)
            .to_pydantic()
        )

        # FIXME: pytidb should return the relationship field: target_entity, source_entity.
        entity_ids = [item.hit.target_entity_id for item in results]
        entity_ids.extend([item.hit.source_entity_id for item in results])
        entities = self.list_entities(filters=EntityFilters(entity_id=entity_ids))
        entity_map = {entity.id: entity for entity in entities}
        for item in results:
            item.hit.target_entity = entity_map[item.hit.target_entity_id]
            item.hit.source_entity = entity_map[item.hit.source_entity_id]

        return [(item.hit, item.score) for item in results]

    def _convert_relationship_filters(self, filters: RelationshipFilters) -> dict:
        filter_dict = {}

        if filters.entity_id:
            if isinstance(filters.entity_id, list):
                if len(filters.entity_id) != 0:
                    filter_dict["$or"] = [
                        {"target_entity_id": {"$in": filters.entity_id}},
                        {"source_entity_id": {"$in": filters.entity_id}},
                    ]
            else:
                filter_dict["$or"] = [
                    {"target_entity_id": {"$eq": filters.entity_id}},
                    {"source_entity_id": {"$eq": filters.entity_id}},
                ]

        if filters.source_entity_id:
            if isinstance(filters.source_entity_id, list):
                if len(filters.source_entity_id) != 0:
                    filter_dict["$or"] = [
                        {"source_entity_id": {"$in": filters.source_entity_id}}
                    ]
            else:
                filter_dict["$or"] = [
                    {"source_entity_id": {"$eq": filters.source_entity_id}}
                ]

        if filters.target_entity_id:
            if isinstance(filters.target_entity_id, list):
                if len(filters.target_entity_id) != 0:
                    filter_dict["$or"] = [
                        {"target_entity_id": {"$in": filters.target_entity_id}}
                    ]
            else:
                filter_dict["$or"] = [
                    {"target_entity_id": {"$eq": filters.target_entity_id}}
                ]

        if filters.relationship_id:
            if isinstance(filters.relationship_id, list):
                if len(filters.relationship_id) != 0:
                    filter_dict["id"] = {"$in": filters.relationship_id}
            else:
                filter_dict["id"] = {"$eq": filters.relationship_id}

        if (
            filters.exclude_relationship_ids
            and len(filters.exclude_relationship_ids) != 0
        ):
            filter_dict["id"] = {"$nin": filters.exclude_relationship_ids}

        if filters.document_id:
            if isinstance(filters.document_id, list):
                if len(filters.document_id) != 0:
                    filter_dict["document_id"] = {"$in": filters.document_id}
            else:
                filter_dict["document_id"] = {"$eq": filters.document_id}

        if filters.chunk_id:
            if isinstance(filters.chunk_id, list):
                if len(filters.chunk_id) != 0:
                    filter_dict["chunk_id"] = {"$in": filters.chunk_id}
            else:
                filter_dict["chunk_id"] = {"$eq": filters.chunk_id}

        if filters.metadata:
            for key, value in filters.metadata.items():
                op = "$in" if isinstance(value, list) else "$eq"
                filter_dict[f"meta.{key}"] = {op: value}

        return filter_dict

    def create_relationship(
        self,
        source_entity: Entity | UUID,
        target_entity: Entity | UUID,
        description: Optional[str] = None,
        meta: Optional[dict] = {},
        embedding: Optional[Any] = None,
    ) -> Relationship:
        """
        Create a relationship between two entities.
        """
        if isinstance(source_entity, UUID):
            source_entity = self.get_entity(source_entity)
        if isinstance(target_entity, UUID):
            target_entity = self.get_entity(target_entity)

        if embedding is None:
            embedding = self._get_relationship_embedding(
                source_entity.name,
                source_entity.description,
                target_entity.name,
                target_entity.description,
                description,
            )

        relationship = self._relationship_db_model(
            source_entity_id=source_entity.id,
            target_entity_id=target_entity.id,
            description=description,
            meta=meta,
            embedding=embedding,
        )
        return self._relationship_table.insert(relationship)

    def _get_relationship_embedding(
        self,
        source_entity_name: str,
        source_entity_description,
        target_entity_name: str,
        target_entity_description: str,
        relationship_desc: str,
    ) -> List[float]:
        embedding_str = (
            f"{source_entity_name}({source_entity_description}) -> "
            f"{relationship_desc} -> {target_entity_name}({target_entity_description}) "
        )
        return self._embedding_model.get_text_embedding(embedding_str)

    def update_relationship(
        self, relationship: Relationship | UUID, update: RelationshipUpdate
    ) -> Relationship:
        if isinstance(relationship, UUID):
            relationship = self.get_relationship(relationship)

        update_dict = update.model_dump()
        if update.embedding is None:
            update_dict["embedding"] = self._get_relationship_embedding(
                relationship.source_entity.name,
                relationship.source_entity.description,
                relationship.target_entity.name,
                relationship.target_entity.description,
                relationship.description,
            )

        self._relationship_table.update(
            values=update_dict, filters={"id": relationship.id}
        )
        # FIXME: pytidb should return the updated relationship.
        relationship = self._relationship_table.get(relationship.id)
        return relationship

    def delete_relationship(self, relationship_id: UUID):
        return self._relationship_table.delete(filters={"id": relationship_id})

    # Knowledge Graph Operations

    def add(self, knowledge_graph: KnowledgeGraphCreate) -> Optional[KnowledgeGraph]:
        with self._db.session():
            # Create or find entities
            entity_map = {}
            for entity in knowledge_graph.entities:
                created_entity = self.find_or_create_entity(
                    entity_type=EntityType.original,
                    name=entity.name,
                    description=entity.description,
                    meta=entity.meta,
                )
                entity_map[entity.name] = created_entity
            entities = list(entity_map.values())

            # Create relationships
            relationships = []
            for rel in knowledge_graph.relationships:
                logger.info("Saving relationship: %s", rel.description)
                source_entity = entity_map.get(rel.source_entity_name)
                if not source_entity:
                    logger.warning(
                        "Source entity not found for relationship: %s", str(rel)
                    )
                    continue

                target_entity = entity_map.get(rel.target_entity_name)
                if not target_entity:
                    logger.warning(
                        "Target entity not found for relationship: %s", str(rel)
                    )
                    continue

                relationship = self.create_relationship(
                    source_entity=source_entity,
                    target_entity=target_entity,
                    description=rel.description,
                    meta=rel.meta,
                )
                relationships.append(relationship)

        return KnowledgeGraph(
            entities=[Entity(**entity.model_dump()) for entity in entities],
            relationships=[
                Relationship(**relationship.model_dump())
                for relationship in relationships
            ],
        )

    # Graph Store Operations

    def reset(self):
        with self._db.session():
            self._db.execute("SET FOREIGN_KEY_CHECKS = 0")
            self._relationship_table.truncate()
            self._entity_table.truncate()
            self._db.execute("SET FOREIGN_KEY_CHECKS = 1")

    def recreate(self):
        self._db.drop_table(self._relationship_table.table_name)
        self._db.drop_table(self._entity_table.table_name)
        self._entity_table = self._db.create_table(schema=self._entity_db_model)
        self._relationship_table = self._db.create_table(
            schema=self._relationship_db_model
        )

    def drop(self):
        self._db.drop_table(self._relationship_table.table_name)
        self._db.drop_table(self._entity_table.table_name)
