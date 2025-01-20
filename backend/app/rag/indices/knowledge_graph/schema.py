import datetime
from hashlib import sha256

from llama_index.core.schema import BaseNode, MetadataMode
from pydantic import BaseModel, Field
from typing import Mapping, Any, List, Optional

from app.utils.jinja2 import get_prompt_by_jinja2_template


class Entity(BaseModel):
    """List of entities extracted from the text to form the knowledge graph"""

    name: str = Field(
        description="Name of the entity, it should be a clear and concise term"
    )
    description: str = Field(
        description=(
            "Description of the entity, it should be a complete and comprehensive sentence, not few words. "
            "Sample description of entity 'TiDB in-place upgrade': "
            "'Upgrade TiDB component binary files to achieve upgrade, generally use rolling upgrade method'"
        )
    )
    metadata: Mapping[str, Any] = Field(
        description=(
            "The covariates (which is a comprehensive json TREE, the first field is always: 'topic') to claim the entity. "
        )
    )


class EntityWithID(Entity):
    """Entity extracted from the text to form the knowledge graph with an ID."""

    id: int = Field(description="Unique identifier for the entity.")


class SynopsisInfo(BaseModel):
    """A synopsis corresponds to a group of entities that share the same topic and can contribute to synopsis topic."""

    topic: str = Field(
        description="The shared topic of the synopsis, and each entity in the group can contribute factual data from its own perspective."
    )
    entities: List[int] = Field(
        description="A group of entity(only IDs) that can contribute to the synopsis base on the analysis of entity descriptions and metadata."
    )


class SynopsisEntity(Entity):
    """Unified synopsis entity with comprehensive description and metadata based on the entities group."""

    group_info: SynopsisInfo = Field(
        description="Group of entities to be unified into a single synopsis entity."
    )


class ExistingSynopsisEntity(SynopsisEntity):
    """Unified synopsis entity with comprehensive description and metadata based on the entities group."""

    id: int = Field(description="Unique identifier for the entity.")


class Relationship(BaseModel):
    """List of relationships extracted from the text to form the knowledge graph"""

    source_entity: str = Field(
        description="Source entity name of the relationship, it should an existing entity in the Entity list"
    )
    target_entity: str = Field(
        description="Target entity name of the relationship, it should an existing entity in the Entity list"
    )
    relationship_desc: str = Field(
        description=(
            "Description of the relationship, it should be a complete and comprehensive sentence, not few words. "
            "Sample relationship description: 'TiDB will release a new LTS version every 6 months.'"
        )
    )


class RelationshipReasoning(Relationship):
    """Relationship between two entities extracted from the query"""

    reasoning: str = Field(
        description=(
            "Category reasoning for the relationship, e.g., 'the main conerns of the user', 'the problem the user is facing', 'the user case scenario', etc."
        )
    )


class KnowledgeGraph(BaseModel):
    """Graph representation of the knowledge for text."""

    entities: List[Entity] = Field(
        description="List of entities in the knowledge graph"
    )
    relationships: List[Relationship] = Field(
        description="List of relationships in the knowledge graph"
    )


class EntityCovariateInput(BaseModel):
    """List of entities extracted from the text to form the knowledge graph"""

    name: str = Field(description="Name of the entity")
    description: str = Field(description=("Description of the entity"))


class EntityCovariateOutput(BaseModel):
    """List of entities extracted from the text to form the knowledge graph"""

    name: str = Field(description="Name of the entity")
    description: str = Field(description=("Description of the entity"))
    covariates: Mapping[str, Any] = Field(
        description=(
            "The attributes (which is a comprehensive json TREE, the first field is always: 'topic') to claim the entity. "
        )
    )


class DecomposedFactors(BaseModel):
    """Decomposed factors extracted from the query to form the knowledge graph"""

    relationships: List[RelationshipReasoning] = Field(
        description="List of relationships to represent critical concepts and their relationships extracted from the query."
    )


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
    entities: List[RetrievedEntity] = Field(
        description="List of entities in the knowledge graph"
    )
    relationships: List[RetrievedRelationship] = Field(
        description="List of relationships in the knowledge graph"
    )


# KnowledgeGraphNode

DEFAULT_ENTITY_TEMPL = """
{% for entity in entities %}

- Name: {{ entity.name }}
- Description: {{ entity.description }}

{% endfor %}
"""

DEFAULT_RELATIONSHIP_TEMPL = """
{% for relationship in relationships %}

- Description: {{ relationship.rag_description }}
- Weight: {{ relationship.weight }}
- Last Modified At: {{ relationship.last_modified_at }}
- Meta: {{ relationship.meta | tojson(indent=2) }}

{% endfor %}
"""


class KnowledgeGraphNode(BaseNode):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    query: str = Field(description="Query of the knowledge graph")
    queries: Optional[list[str]] = Field(description="Queries of the knowledge graph")
    entities: List[Entity] = Field(
        default_factory=list, description="The list of entities in the knowledge graph"
    )
    relationships: List[Relationship] = Field(
        default_factory=list,
        description="The list of relationships in the knowledge graph",
    )
    entity_template: str = Field(
        default=DEFAULT_ENTITY_TEMPL,
        description="The template to render the entity list as string",
    )
    relationship_template: str = Field(
        default=DEFAULT_RELATIONSHIP_TEMPL,
        description="The template to render the relationship list as string",
    )

    @classmethod
    def get_type(cls) -> str:
        return "KnowledgeGraphNode"

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.ALL) -> str:
        return f"""
        Entities:
        ------
        {self._get_entities_str()}
        
        Relationships:
        ------
        {self._get_relationships_str()}
        """

    def _get_entities_str(self):
        return get_prompt_by_jinja2_template(
            self.entity_template, entities=self.entities
        )

    def _get_relationships_str(self):
        return get_prompt_by_jinja2_template(
            self.relationship_template, relationships=self.relationships
        )

    def set_content(self, value: KnowledgeGraph) -> None:
        self.entities = value.entities
        self.relationships = value.relationships

    @property
    def hash(self) -> str:
        knowledge_graph_identity = (
            self._get_entities_str() + self._get_relationships_str()
        )
        return str(sha256(knowledge_graph_identity).hexdigest())
