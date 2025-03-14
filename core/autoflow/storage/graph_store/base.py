from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum

from typing import (
    List,
    Optional,
    TypeVar,
    Type,
    Generic,
    Mapping,
    Any,
    Tuple,
    Sequence,
    Collection,
    Dict,
)
from uuid import UUID

from pydantic import BaseModel, model_validator, Field
from sqlmodel import SQLModel, Session

from autoflow.indices.knowledge_graph.schema import AIKnowledgeGraph
from autoflow.llms.embeddings import EmbeddingModel
from autoflow.models.entity import EntityType
from autoflow.storage.schema import QueryBundle


# Entity


class EntityCreate(BaseModel):
    entity_type: Optional[EntityType] = EntityType.original
    name: Optional[str] = None
    description: Optional[str] = None
    meta: Optional[dict] = None


class SynopsisEntityCreate(EntityCreate):
    topic: str
    entities: List[int] = Field(description="The id list of the related entities")

    @model_validator(mode="after")
    def validate_entities(self):
        if len(self.entities) == 0:
            raise ValueError("Entities list should not be empty")
        return self


class EntityFilters(BaseModel):
    entity_ids: Optional[List[int]] = None
    entity_type: Optional[EntityType] = None
    search: Optional[str] = None


class EntityUpdate(BaseModel):
    description: Optional[str] = None
    meta: Optional[dict] = None


class EntityDegree(BaseModel):
    out_degree: int = 0
    in_degree: int = 0
    degrees: int = 0


# Relationship


class RelationshipCreate(BaseModel):
    source_entity_id: int
    target_entity_id: int
    description: str


class RelationshipUpdate(BaseModel):
    description: Optional[str] = None


class RelationshipFilters(BaseModel):
    target_entity_id: Optional[int] = None
    source_entity_id: Optional[int] = None
    chunk_ids: Optional[List[UUID]] = None
    relationship_ids: Optional[List[int]] = None
    search: Optional[str] = None


# Stored Knowledge Graph


class StoredKnowledgeGraphVersion(int, Enum):
    V1 = 1


class StoredSubGraph(BaseModel):
    query: Optional[str] = None
    knowledge_base_id: Optional[int] = None
    entities: Optional[list[int]] = None
    relationships: Optional[list[int]] = None


class StoredKnowledgeGraph(StoredSubGraph):
    """
    StoredKnowledgeGraph represents the structure of the knowledge graph stored in the database.
    """

    # If not provided, it means that the old version of the storage format is used, which only
    # stores entities and relationships information.
    version: Optional[int] = StoredKnowledgeGraphVersion.V1
    knowledge_base_ids: Optional[list[int]] = []
    subgraphs: Optional[list["StoredSubGraph"]] = None


# Retrieved Knowledge Graph


class RetrievedEntity(BaseModel):
    id: int = Field(description="ID of the entity")
    knowledge_base_id: Optional[int] = Field(
        description="ID of the knowledge base", default=None
    )
    entity_type: Optional[EntityType] = Field(
        description="Type of the entity", default=EntityType.original
    )
    name: str = Field(description="Name of the entity")
    description: str = Field(description="Description of the entity")
    meta: Optional[Mapping[str, Any]] = Field(description="Metadata of the entity")
    similarity_score: Optional[float] = Field(default=None)

    @property
    def global_id(self) -> str:
        return f"{self.knowledge_base_id or 0}-{self.id}"

    def __hash__(self):
        return hash(self.global_id)


class RetrievedRelationship(BaseModel):
    id: int = Field(description="ID of the relationship")
    knowledge_base_id: int = Field(description="ID of the knowledge base", default=None)
    source_entity_id: int = Field(description="ID of the source entity")
    target_entity_id: int = Field(description="ID of the target entity")
    description: str = Field(description="Description of the relationship")
    meta: Optional[Mapping[str, Any]] = Field(
        description="Metadata of the relationship"
    )
    rag_description: Optional[str] = Field(
        description="RAG description of the relationship"
    )
    weight: Optional[float] = Field(description="Weight of the relationship")
    last_modified_at: Optional[datetime] = Field(
        description="Last modified at of the relationship", default=None
    )
    similarity_score: Optional[float] = Field()

    @property
    def global_id(self) -> str:
        return f"{self.knowledge_base_id or 0}-{self.id}"

    def __hash__(self):
        return hash(self.global_id)


class RetrievedSubGraph(BaseModel):
    query: Optional[str | list[str]] = Field(
        description="List of queries that are used to retrieve the knowledge graph",
        default=None,
    )
    # knowledge_base: Optional[KnowledgeBaseDescriptor] = Field(
    #     description="The knowledge base that the knowledge graph is retrieved from",
    #     default=None,
    # )
    entities: List[RetrievedEntity] = Field(
        description="List of entities in the knowledge graph", default_factory=list
    )
    relationships: List[RetrievedRelationship] = Field(
        description="List of relationships in the knowledge graph", default_factory=list
    )


class RetrievedKnowledgeGraph(RetrievedSubGraph):
    """
    RetrievedKnowledgeGraph represents the structure of the knowledge graph retrieved
    from the knowledge base.
    """

    # knowledge_bases: Optional[List[KnowledgeBaseDescriptor]] = Field(
    #     description="List of knowledge bases that the knowledge graph is retrieved from",
    #     default_factory=list,
    # )

    # subgraphs: Optional[List["RetrievedSubGraph"]] = Field(
    #     description="List of subgraphs of the knowledge graph", default_factory=list
    # )

    def to_subqueries_dict(self) -> dict:
        """
        For forward compatibility, we need to convert the subgraphs to a dictionary
        of subqueries and then pass it to the prompt template.
        """
        subqueries = {}
        for subgraph in self.subgraphs:
            if subgraph.query not in subqueries:
                subqueries[subgraph.query] = {
                    "entities": [e.model_dump() for e in subgraph.entities],
                    "relationships": [r.model_dump() for r in subgraph.relationships],
                }
            else:
                subqueries[subgraph.query]["entities"].extend(
                    [e.model_dump() for e in subgraph.entities]
                )
                subqueries[subgraph.query]["relationships"].extend(
                    [r.model_dump() for r in subgraph.relationships]
                )

        return subqueries

    def to_stored_graph_dict(self) -> dict:
        subgraph = self.to_stored_graph()
        return subgraph.model_dump()

    def to_stored_graph(self) -> StoredKnowledgeGraph:
        return StoredKnowledgeGraph(
            query=self.query,
            knowledge_base_id=self.knowledge_base.id if self.knowledge_base else None,
            knowledge_base_ids=[kb.id for kb in self.knowledge_bases]
            if self.knowledge_bases
            else None,
            entities=[e.id for e in self.entities],
            relationships=[r.id for r in self.relationships],
            subgraphs=[s.to_stored_graph() for s in self.subgraphs],
        )


KnowledgeGraphRetrievalResult = RetrievedKnowledgeGraph

# Graph Search


class GraphSearchAlgorithm(str, Enum):
    WEIGHTED_SEARCH = "weighted"


# Knowledge Graph Store

E = TypeVar("E", SQLModel, Type[SQLModel])
R = TypeVar("R", SQLModel, Type[SQLModel])
C = TypeVar("C", SQLModel, Type[SQLModel])


class KnowledgeGraphStore(ABC, Generic[E, R, C]):
    """Abstract base class for knowledge graph storage"""

    def __init__(self, embed_model: EmbeddingModel):
        self._embed_model = embed_model

    # Schema Operations
    @abstractmethod
    def ensure_table_schema(self) -> None:
        """Ensure database table schema exists"""
        raise NotImplementedError

    @abstractmethod
    def drop_table_schema(self) -> None:
        """Drop database table schema"""
        raise NotImplementedError

    @abstractmethod
    def save_knowledge_graph(
        self, knowledge_graph: AIKnowledgeGraph, chunk: SQLModel
    ) -> None:
        raise NotImplementedError

    # Entity Basic Operations

    @abstractmethod
    def list_entities(
        self,
        filters: Optional[EntityFilters] = EntityFilters(),
        db_session: Session = None,
    ) -> Sequence[E]:
        """List all entities matching the filters"""
        raise NotImplementedError

    @abstractmethod
    def get_entity_by_id(self, entity_id: int) -> Type[E]:
        """Get entity by ID"""
        raise NotImplementedError

    @abstractmethod
    def must_get_entity_by_id(self, entity_id: int) -> Type[E]:
        """Get entity by ID, raise error if not found"""
        raise NotImplementedError

    @abstractmethod
    def create_entity(
        self, create: EntityCreate, commit: bool = True, db_session: Session = None
    ) -> E:
        """Create a new entity"""
        raise NotImplementedError

    @abstractmethod
    def find_or_create_entity(
        self, create: EntityCreate, commit: bool = True, db_session: Session = None
    ) -> E:
        """Find existing entity or create new one"""
        raise NotImplementedError

    @abstractmethod
    def update_entity(
        self,
        entity: Type[E],
        update: EntityUpdate,
        commit: bool = True,
        db_session: Session = None,
    ) -> Type[E]:
        """Update an existing entity"""
        raise NotImplementedError

    @abstractmethod
    def delete_entity(
        self, entity: Type[E], commit: bool = True, db_session: Session = None
    ) -> None:
        """Delete an entity"""
        raise NotImplementedError

    @abstractmethod
    def list_entity_relationships(self, entity_id: int) -> List[R]:
        """List all relationships for an entity"""
        raise NotImplementedError

    # Entity Degree Operations
    @abstractmethod
    def calc_entity_out_degree(self, entity_id: int) -> Optional[int]:
        """Calculate out-degree of an entity"""
        raise NotImplementedError

    @abstractmethod
    def calc_entity_in_degree(self, entity_id: int) -> Optional[int]:
        """Calculate in-degree of an entity"""
        raise NotImplementedError

    @abstractmethod
    def calc_entity_degree(self, entity_id: int) -> Optional[int]:
        """Calculate total degree of an entity"""
        raise NotImplementedError

    @abstractmethod
    def bulk_calc_entities_degrees(
        self, entity_ids: Collection[int]
    ) -> Mapping[int, EntityDegree]:
        """Calculate degrees for multiple entities in bulk"""
        raise NotImplementedError

    # Entity Retrieve Operations
    @abstractmethod
    def retrieve_entities(
        self,
        query: str,
        entity_type: EntityType = EntityType.original,
        top_k: int = 10,
        nprobe: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
    ) -> List[RetrievedEntity]:
        """Retrieve entities based on semantic search"""
        raise NotImplementedError

    @abstractmethod
    def search_similar_entities(
        self,
        query: QueryBundle,
        top_k: int = 10,
        nprobe: Optional[int] = None,
        entity_type: EntityType = EntityType.original,
        similarity_threshold: Optional[float] = None,
    ) -> List[Tuple[E, float]]:
        """Search for similar entities using vector similarity"""
        raise NotImplementedError

    # Relationship Basic Operations
    @abstractmethod
    def get_relationship_by_id(self, relationship_id: int) -> R:
        """Get relationship by ID"""
        raise NotImplementedError

    @abstractmethod
    def list_relationships(self, filters: RelationshipFilters) -> Sequence[R]:
        """List all relationships matching the filters"""
        raise NotImplementedError

    @abstractmethod
    def create_relationship(
        self,
        source_entity: E,
        target_entity: E,
        description: Optional[str] = None,
        metadata: Optional[dict] = {},
        commit: bool = True,
        db_session: Session = None,
    ) -> R:
        """Create a new relationship between entities"""
        raise NotImplementedError

    @abstractmethod
    def update_relationship(
        self,
        relationship: R,
        update: RelationshipUpdate,
        commit: bool = True,
        db_session: Session = None,
    ) -> R:
        """Update an existing relationship"""
        raise NotImplementedError

    @abstractmethod
    def delete_relationship(
        self, relationship: R, commit: bool = True, db_session: Session = None
    ):
        """Delete a relationship"""
        raise NotImplementedError

    @abstractmethod
    def clear_orphan_entities(self, db_session: Session = None):
        """Remove entities that have no relationships"""
        raise NotImplementedError

    # Relationship Retrieve Operations
    @abstractmethod
    def retrieve_relationships(
        self,
        query: str,
        top_k: int = 10,
        nprobe: Optional[int] = None,
        similarity_threshold: Optional[float] = 0,
    ) -> List[RetrievedRelationship]:
        """Retrieve relationships based on semantic search"""
        raise NotImplementedError

    @abstractmethod
    def search_similar_relationships(
        self,
        query: QueryBundle,
        top_k: int = 10,
        nprobe: Optional[int] = None,
        similarity_threshold: Optional[float] = None,
        distance_range: Optional[Tuple[float, float]] = None,
        weight_threshold: Optional[float] = None,
        exclude_relationship_ids: Optional[List[str]] = None,
        source_entity_ids: Optional[List[int]] = None,
        target_entity_ids: Optional[List[int]] = None,
        metadata_filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[R, float]]:
        """Search for similar relationships using vector similarity

        Args:
            query: Query bundle containing search text or embedding
            top_k: Number of results to return
            nprobe: Number of clusters to probe in ANN search
            similarity_threshold: Minimum similarity score threshold
            distance_range: Range of acceptable distances
            weight_threshold: Minimum relationship weight threshold
            exclude_relationship_ids: Relationship IDs to exclude
            source_entity_ids: Filter by source entity IDs
            target_entity_ids: Filter by target entity IDs
            metadata_filters: Filter by metadata fields

        Returns:
            List of tuples containing relationship and similarity score
        """
        raise NotImplementedError

    # Graph Basic Operations
    @abstractmethod
    def list_relationships_by_ids(
        self, relationship_ids: list[int], **kwargs
    ) -> List[R]:
        """Get multiple relationships by their IDs"""
        raise NotImplementedError

    # Knowledge Graph Retrieve Operations
    def search(
        self,
        query: QueryBundle,
        depth: int = 2,
        include_meta: bool = False,
        metadata_filters: Optional[dict] = None,
        search_algorithm: Optional[
            GraphSearchAlgorithm
        ] = GraphSearchAlgorithm.WEIGHTED_SEARCH,
        **kwargs,
    ) -> RetrievedKnowledgeGraph:
        """Search the knowledge graph using configurable search algorithm

        Args:
            query: Query bundle containing search text or embedding
            depth: Maximum search depth in the graph
            include_meta: Whether to include metadata in results
            metadata_filters: Filters to apply on metadata
            search_algorithm: Algorithm class to use for graph search

        Returns:
            Retrieved subgraph containing matching entities and relationships

        """
        # Ensure query has embedding
        if query.query_embedding is None and hasattr(self, "_embed_model"):
            query.query_embedding = self._embed_model.get_query_embedding(
                query.query_str
            )

        # Initialize and execute search algorithm
        if search_algorithm is GraphSearchAlgorithm.WEIGHTED_SEARCH:
            from autoflow.storage.graph_store.algorithms.weighted import (
                WeightedGraphSearchRetriever,
            )

            retriever = WeightedGraphSearchRetriever(self, **kwargs)
        else:
            raise NotImplementedError(f"Unknown search algorithm <{search_algorithm}>")

        relationships, entities = retriever.search(
            query_embedding=query.query_embedding,
            depth=depth,
            metadata_filters=metadata_filters,
        )

        # Construct result graph
        return RetrievedKnowledgeGraph(
            relationships=[
                RetrievedRelationship(
                    id=r.id,
                    source_entity_id=r.relationship.source_entity_id,
                    target_entity_id=r.relationship.target_entity_id,
                    description=r.relationship.description,
                    rag_description=f"{r.relationship.source_entity.name} -> {r.relationship.description} -> {r.relationship.target_entity.name}",
                    meta=r.relationship.meta if include_meta else None,
                    weight=r.relationship.weight,
                    last_modified_at=r.relationship.last_modified_at
                    if hasattr(r.relationship, "last_modified_at")
                    else None,
                    similarity_score=r.score if hasattr(r, "score") else None,
                )
                for r in relationships
            ],
            entities=[
                RetrievedEntity(
                    id=e.id,
                    entity_type=e.entity_type,
                    name=e.name,
                    description=e.description,
                    meta=e.meta if include_meta else None,
                )
                for e in entities
            ],
        )
