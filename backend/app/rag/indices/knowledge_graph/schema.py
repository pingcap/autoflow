from typing import Mapping, Any, List, Optional
import dspy
from dspy import TypedPredictor
from pydantic import BaseModel, model_validator, Field
from app.models.entity import EntityType


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


class SynopsisEntity(AIEntity):
    """Unified synopsis entity with comprehensive description and metadata based on the entities group."""

    group_info: SynopsisInfo = Field(
        description="Group of entities to be unified into a single synopsis entity."
    )


class ExistingSynopsisEntity(SynopsisEntity):
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

    relationships: List[AIRelationshipReasoning] = Field(
        description="List of relationships to represent critical concepts and their relationships extracted from the query."
    )


class MergeEntities(dspy.Signature):
    """As a knowledge expert assistant specialized in database technologies, evaluate the two provided entities. These entities have been pre-analyzed and have same name but different descriptions and metadata.
    Please carefully review the detailed descriptions and metadata for both entities to determine if they genuinely represent the same concept or object(entity).
    If you conclude that the entities are identical, merge the descriptions and metadata fields of the two entities into a single consolidated entity.
    If the entities are distinct despite their same name that may be due to different contexts or perspectives, do not merge the entities and return none as the merged entity.

    Considerations: Ensure your decision is based on a comprehensive analysis of the content and context provided within the entity descriptions and metadata.
    Please only response in JSON Format.
    """

    entities: List[AIEntity] = dspy.InputField(
        desc="List of entities identified from previous analysis."
    )
    merged_entity: Optional[AIEntity] = dspy.OutputField(
        desc="Merged entity with consolidated descriptions and metadata."
    )


class MergeEntitiesProgram(dspy.Module):
    def __init__(self):
        self.prog = TypedPredictor(MergeEntities)

    def forward(self, entities: List[AIEntity]):
        if len(entities) != 2:
            raise ValueError("The input should contain exactly two entities")

        pred = self.prog(entities=entities)
        return pred


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
    entity_id: int
    out_degree: int
    in_degree: int
    degrees: int


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
    relationship_ids: Optional[List[int]] = None
    search: Optional[str] = None
