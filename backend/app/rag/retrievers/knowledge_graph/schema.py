import datetime
import json
from abc import ABC
from enum import Enum

from hashlib import sha256
from typing import Optional, Mapping, Any, List
from llama_index.core.schema import BaseNode, MetadataMode
from pydantic import BaseModel, Field

from app.models.entity import EntityType
from app.api.admin_routes.models import KnowledgeBaseDescriptor

# Retriever Config


class MetadataFilterConfig(BaseModel):
    enabled: bool = True
    filters: dict[str, Any] = None


class KnowledgeGraphRetrieverConfig(BaseModel):
    depth: int = 2
    include_meta: bool = False
    with_degree: bool = False
    metadata_filter: Optional[MetadataFilterConfig] = None


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
    description: str = Field(description="Description of the entity", default="")
    meta: Optional[Mapping[str, Any]] = Field(
        description="Metadata of the entity", default={}
    )

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
    last_modified_at: Optional[datetime.datetime] = Field(
        description="Last modified at of the relationship", default=None
    )

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
    knowledge_base: Optional[KnowledgeBaseDescriptor] = Field(
        description="The knowledge base that the knowledge graph is retrieved from",
        default=None,
    )
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

    knowledge_bases: Optional[List[KnowledgeBaseDescriptor]] = Field(
        description="List of knowledge bases that the knowledge graph is retrieved from",
        default_factory=list,
    )

    subgraphs: Optional[List["RetrievedSubGraph"]] = Field(
        description="List of subgraphs of the knowledge graph", default_factory=list
    )

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


class KnowledgeGraphRetriever(ABC):
    def retrieve_knowledge_graph(self, query_str: str) -> KnowledgeGraphRetrievalResult:
        raise NotImplementedError


# KnowledgeGraphNode

DEFAULT_KNOWLEDGE_GRAPH_TMPL = """
Query:
------
{query}

Entities:
------
{entities_str}

Relationships:
------
{relationships_str}
"""
DEFAULT_ENTITY_TMPL = """
- Name: {{ name }}
  Description: {{ description }}
"""
DEFAULT_RELATIONSHIP_TMPL = """
- Description: {{ rag_description }}
  Weight: {{ weight }}
  Last Modified At: {{ last_modified_at }}
  Meta: {{ meta }}
"""


class KnowledgeGraphNode(BaseNode):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    query: Optional[str] = Field(description="Query of the knowledge graph")

    knowledge_base_id: Optional[int] = Field(
        description="The id of the knowledge base that the knowledge graph belongs to",
        default=None,
    )
    knowledge_base_ids: Optional[List[int]] = Field(
        description="List of ids of the knowledge base that the knowledge graph belongs to",
        default_factory=list,
    )

    entities: List[RetrievedEntity] = Field(
        description="The list of entities in the knowledge graph", default_factory=list
    )
    relationships: List[RetrievedRelationship] = Field(
        description="The list of relationships in the knowledge graph",
        default_factory=list,
    )
    children: Optional[List["KnowledgeGraphNode"]] = Field(
        description="The children of the knowledge graph",
        default_factory=list,
    )

    # Template

    knowledge_base_template: str = Field(
        default=DEFAULT_KNOWLEDGE_GRAPH_TMPL,
        description="The template to render the knowledge graph as string",
    )
    entity_template: str = Field(
        default=DEFAULT_ENTITY_TMPL,
        description="The template to render the entity list as string",
    )
    relationship_template: str = Field(
        default=DEFAULT_RELATIONSHIP_TMPL,
        description="The template to render the relationship list as string",
    )

    @classmethod
    def get_type(cls) -> str:
        return "KnowledgeGraphNode"

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.ALL) -> str:
        return f"""
        Query:
        ------
        {self.query}

        Entities:
        ------
        {self._get_entities_str()}
        
        Relationships:
        ------
        {self._get_relationships_str()}
        """

    def _get_entities_str(self) -> str:
        strs = []
        for entity in self.entities:
            strs.append(
                self.entity_template.format(
                    name=entity.name, description=entity.description
                )
            )
        return "\n\n".join(strs)

    def _get_relationships_str(self) -> str:
        strs = []
        for relationship in self.relationships:
            strs.append(
                self.entity_template.format(
                    rag_description=relationship.rag_description,
                    weight=relationship.weight,
                    last_modified_at=relationship.last_modified_at,
                    meta=json.dumps(relationship.meta, indent=2, ensure_ascii=False),
                )
            )
        return "\n\n".join(strs)

    def _get_knowledge_graph_str(self) -> str:
        return self.knowledge_base_template.format(
            query=self.query,
            entities_str=self._get_entities_str(),
            relationships_str=self._get_relationships_str(),
        )

    def set_content(self, kg: RetrievedKnowledgeGraph):
        self.query = kg.query
        self.knowledge_base_id = kg.knowledge_base.id if kg.knowledge_base else None
        self.knowledge_base_ids = []
        self.entities = kg.entities
        self.relationships = kg.relationships
        self.children = [
            KnowledgeGraphNode(
                query=subgraph.query,
                knowledge_base_id=subgraph.knowledge_base.id
                if subgraph.knowledge_base
                else None,
                entities=subgraph.entities,
                relationships=subgraph.relationships,
            )
            for subgraph in kg.subgraphs
        ]

    @property
    def hash(self) -> str:
        kg_identity = self._get_knowledge_graph_str().encode("utf-8")
        return str(sha256(kg_identity).hexdigest())
