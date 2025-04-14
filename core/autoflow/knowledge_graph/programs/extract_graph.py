import logging
from typing import List

import dspy
from dspy import Predict
from pydantic import BaseModel, Field

from autoflow.knowledge_graph.types import (
    GeneratedEntity,
    GeneratedKnowledgeGraph,
    GeneratedRelationship,
)

logger = logging.getLogger(__name__)


class PredictEntity(BaseModel):
    """Entity extracted from the text to form the knowledge graph"""

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


class PredictRelationship(BaseModel):
    """Relationship extracted from the text to form the knowledge graph"""

    source_entity: str = Field(
        description="Source entity name of the relationship, it should an existing entity in the Entity list"
    )
    target_entity: str = Field(
        description="Target entity name of the relationship, it should an existing entity in the Entity list"
    )
    relationship_desc: str = Field(
        description=(
            "Description of the relationship, it should be a complete and comprehensive sentence, not few words. "
            "For example: 'TiDB will release a new LTS version every 6 months.'"
        )
    )


class PredictKnowledgeGraph(BaseModel):
    """Graph representation of the knowledge for text."""

    entities: List[PredictEntity] = Field(
        description="List of entities in the knowledge graph"
    )
    relationships: List[PredictRelationship] = Field(
        description="List of relationships in the knowledge graph"
    )

    def to_pandas(self):
        from pandas import DataFrame

        return {
            "entities": DataFrame(
                [
                    {
                        "name": entity.name,
                        "description": entity.description,
                    }
                    for entity in self.entities
                ]
            ),
            "relationships": DataFrame(
                [
                    {
                        "source_entity": relationship.source_entity,
                        "relationship_desc": relationship.relationship_desc,
                        "target_entity": relationship.target_entity,
                    }
                    for relationship in self.relationships
                ]
            ),
        }


class ExtractKnowledgeGraph(dspy.Signature):
    """Carefully analyze the provided text from database documentation and community blogs to thoroughly identify all entities related to database technologies, including both general concepts and specific details.

    Follow these Step-by-Step Analysis:

    1. Extract Meaningful Entities:
      - Identify all significant nouns, proper nouns, and technical terminologies that represent database-related concepts, objects, components, features, issues, key steps, execute order, user case, locations, versions, or any substantial entities.
      - Ensure that you capture entities across different levels of detail, from high-level overviews to specific technical specifications, to create a comprehensive representation of the subject matter.
      - Choose names for entities that are specific enough to indicate their meaning without additional context, avoiding overly generic terms.
      - Consolidate similar entities to avoid redundancy, ensuring each represents a distinct concept at appropriate granularity levels.

    2. Establish Relationships:
      - Carefully examine the text to identify all relationships between clearly-related entities, ensuring each relationship is correctly captured with accurate details about the interactions.
      - Analyze the context and interactions between the identified entities to determine how they are interconnected, focusing on actions, associations, dependencies, or similarities.
      - Clearly define the relationships, ensuring accurate directionality that reflects the logical or functional dependencies among entities. \
         This means identifying which entity is the source, which is the target, and what the nature of their relationship is (e.g., $source_entity depends on $target_entity for $relationship).

    Some key points to consider:
      - Please endeavor to extract all meaningful entities and relationships from the text, avoid subsequent additional gleanings.

    Objective: Produce a detailed and comprehensive knowledge graph that captures the full spectrum of entities mentioned in the text, along with their interrelations, reflecting both broad concepts and intricate details specific to the database domain.

    Please only response in JSON format.
    """

    text = dspy.InputField(
        desc="a paragraph of text to extract entities and relationships to form a knowledge graph"
    )
    knowledge: PredictKnowledgeGraph = dspy.OutputField(
        desc="Graph representation of the knowledge extracted from the text."
    )


class KnowledgeGraphExtractor(dspy.Module):
    def __init__(self, dspy_lm: dspy.LM):
        super().__init__()
        self.dspy_lm = dspy_lm
        self.program = Predict(ExtractKnowledgeGraph)

    def forward(self, text: str) -> GeneratedKnowledgeGraph:
        with dspy.settings.context(lm=self.dspy_lm):
            prediction = self.program(text=text)
            entities = [
                GeneratedEntity(
                    name=entity.name,
                    description=entity.description,
                    meta={},
                )
                for entity in prediction.knowledge.entities
            ]
            relationships = [
                GeneratedRelationship(
                    source_entity_name=relationship.source_entity,
                    target_entity_name=relationship.target_entity,
                    description=relationship.relationship_desc,
                    meta={},
                )
                for relationship in prediction.knowledge.relationships
            ]
            return GeneratedKnowledgeGraph(
                entities=entities,
                relationships=relationships,
            )
