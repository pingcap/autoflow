import datetime
import json

from hashlib import sha256
from typing import Optional, Mapping, Any, List
from llama_index.core.schema import BaseNode, MetadataMode
from pydantic import BaseModel, Field

# Retriever Config


class KnowledgeGraphRetrieverConfig(BaseModel):
    depth: int = 2
    include_meta: bool = False
    with_degree: bool = False
    enable_metadata_filter: bool = False
    metadata_filters: Optional[dict] = None


# Retrieved Knowledge Graph


class RetrievedEntity(BaseModel):
    id: int = Field(description="Unique identifier for the entity")
    name: str = Field(description="Name of the entity")
    description: str = Field(description="Description of the entity")
    meta: Optional[Mapping[str, Any]] = Field(description="Metadata of the entity")


class RetrievedRelationship(BaseModel):
    id: int = Field(description="Unique identifier for the relationship")
    source_entity_id: int = Field(description="Unique identifier for the source entity")
    target_entity_id: int = Field(description="Unique identifier for the target entity")
    description: str = Field(description="Description of the relationship")
    meta: Optional[Mapping[str, Any]] = Field(
        description="Metadata of the relationship"
    )
    rag_description: Optional[str] = Field(
        description="RAG description of the relationship"
    )
    weight: Optional[float] = Field(description="Weight of the relationship")
    last_modified_at: Optional[datetime.datetime] = Field(
        description="Last modified at of the relationship"
    )


class RetrievedKnowledgeGraph(BaseModel):
    query: Optional[str] = None
    entities: List[RetrievedEntity] = Field(
        description="List of entities in the knowledge graph", default_factory=list
    )
    relationships: List[RetrievedRelationship] = Field(
        description="List of relationships in the knowledge graph", default_factory=list
    )
    subqueries: Optional[List["RetrievedKnowledgeGraph"]] = Field(
        description="List of subqueries in the knowledge graph", default_factory=list
    )

    def to_subqueries_dict(self) -> dict:
        sub_queries = {}
        for subquery in self.subqueries:
            sub_queries[subquery.query] = {
                "entities": [e.model_dump() for e in subquery.entities],
                "relationships": [r.model_dump() for r in subquery.relationships],
            }
        return sub_queries

    def to_graph_data_dict(self) -> dict:
        return {
            "entities": [e.id for e in self.entities],
            "relationships": [r.id for r in self.relationships],
        }


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
    entities: List[RetrievedEntity] = Field(
        description="The list of entities in the knowledge graph", default_factory=list
    )
    relationships: List[RetrievedRelationship] = Field(
        description="The list of relationships in the knowledge graph",
        default_factory=list,
    )
    subqueries: Optional[Mapping[str, "KnowledgeGraphNode"]] = Field(
        description="Subqueries",
        default_factory=dict,
    )

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
                    meta=self._get_metadata_str(relationship.meta),
                )
            )
        return "\n\n".join(strs)

    def _get_metadata_str(self, meta: Mapping[str, Any]) -> str:
        return json.dumps(meta, indent=2, ensure_ascii=False)

    def _get_knowledge_graph_str(self) -> str:
        return self.knowledge_base_template.format(
            query=self.query,
            entities_str=self._get_entities_str(),
            relationships_str=self._get_relationships_str(),
        )

    def set_content(self, value: RetrievedKnowledgeGraph) -> None:
        self.query = value.query
        self.entities = value.entities
        self.relationships = value.relationships
        self.subqueries = value.subqueries

    @property
    def hash(self) -> str:
        kg_identity = self._get_knowledge_graph_str().encode("utf-8")
        return str(sha256(kg_identity).hexdigest())
