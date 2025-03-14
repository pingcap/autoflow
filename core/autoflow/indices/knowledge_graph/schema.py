from pydantic import BaseModel, Field
from typing import Mapping, Any, List


class AIEntity(BaseModel):
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


class AIEntityWithID(AIEntity):
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


class AISynopsisEntity(AIEntity):
    """Unified synopsis entity with comprehensive description and metadata based on the entities group."""

    group_info: SynopsisInfo = Field(
        description="Group of entities to be unified into a single synopsis entity."
    )


class ExistingSynopsisEntity(AISynopsisEntity):
    """Unified synopsis entity with comprehensive description and metadata based on the entities group."""

    id: int = Field(description="Unique identifier for the entity.")


class AIRelationship(BaseModel):
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


class AIRelationshipWithEntityDesc(AIRelationship):
    source_entity_description: str = Field()
    target_entity_description: str = Field()
    meta: Mapping[str, Any] = Field()


class AIRelationshipReasoning(AIRelationship):
    """Relationship between two entities extracted from the query"""

    reasoning: str = Field(
        description=(
            "Category reasoning for the relationship, e.g., 'the main conerns of the user', 'the problem the user is facing', 'the user case scenario', etc."
        )
    )


class AIKnowledgeGraph(BaseModel):
    """Graph representation of the knowledge for text."""

    entities: List[AIEntity] = Field(
        description="List of entities in the knowledge graph"
    )
    relationships: List[AIRelationship] = Field(
        description="List of relationships in the knowledge graph"
    )


class ProcessedAIKnowledgeGraph(BaseModel):
    entities: List[AIEntity]
    relationships: List[AIRelationshipWithEntityDesc]


class EntityCovariateInput(BaseModel):
    """List of entities extracted from the text to form the knowledge graph"""

    name: str = Field(description="Name of the entity")
    description: str = Field(description="Description of the entity")


class EntityCovariateOutput(BaseModel):
    """List of entities extracted from the text to form the knowledge graph"""

    name: str = Field(description="Name of the entity")
    description: str = Field(description="Description of the entity")
    covariates: Mapping[str, Any] = Field(
        description=(
            "The attributes (which is a comprehensive json TREE, the first field is always: 'topic') to claim the entity. "
        )
    )


class DecomposedFactors(BaseModel):
    """Decomposed factors extracted from the query to form the knowledge graph"""

    relationships: List[AIRelationshipReasoning] = Field(
        description="List of relationships to represent critical concepts and their relationships extracted from the query."
    )
