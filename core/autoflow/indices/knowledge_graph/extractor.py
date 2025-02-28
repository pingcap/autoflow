import logging
from typing import List

import dspy
from dspy import Predict

from autoflow.indices.knowledge_graph.schema import (
    AIKnowledgeGraph,
    EntityCovariateInput,
    EntityCovariateOutput,
)

logger = logging.getLogger(__name__)


class ExtractGraphTriplet(dspy.Signature):
    """Carefully analyze the provided text from database documentation and community blogs to thoroughly identify all entities related to database technologies, including both general concepts and specific details.

    Follow these Step-by-Step Analysis:

    1. Extract Meaningful Entities:
      - Identify all significant nouns, proper nouns, and technical terminologies that represent database-related concepts, objects, components, features, issues, key steps, execute order, user case, locations, versions, or any substantial entities.
      - Ensure that you capture entities across different levels of detail, from high-level overviews to specific technical specifications, to create a comprehensive representation of the subject matter.
      - Choose names for entities that are specific enough to indicate their meaning without additional context, avoiding overly generic terms.
      - Consolidate similar entities to avoid redundancy, ensuring each represents a distinct concept at appropriate granularity levels.

    2. Extract Metadata to claim the entities:
      - Carefully review the provided text, focusing on identifying detailed covariates associated with each entity.
      - Extract and link the covariates (which is a comprehensive json TREE, the first field is always: "topic") to their respective entities.
      - Ensure all extracted covariates is clearly connected to the correct entity for accuracy and comprehensive understanding.
      - Ensure that all extracted covariates are factual and verifiable within the text itself, without relying on external knowledge or assumptions.
      - Collectively, the covariates should provide a thorough and precise summary of the entity's characteristics as described in the source material.

    3. Establish Relationships:
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
    knowledge: AIKnowledgeGraph = dspy.OutputField(
        desc="Graph representation of the knowledge extracted from the text."
    )


class ExtractCovariate(dspy.Signature):
    """Please carefully review the provided text and entities list which are already identified in the text. Focusing on identifying detailed covariates associated with each entities provided.
    Extract and link the covariates (which is a comprehensive json TREE, the first field is always: "topic") to their respective entities.
    Ensure all extracted covariates is clearly connected to the correct entity for accuracy and comprehensive understanding.
    Ensure that all extracted covariates are factual and verifiable within the text itself, without relying on external knowledge or assumptions.
    Collectively, the covariates should provide a thorough and precise summary of the entity's characteristics as described in the source material.

    Please only response in JSON format.
    """

    text = dspy.InputField(
        desc="a paragraph of text to extract covariates to claim the entities."
    )

    entities: List[EntityCovariateInput] = dspy.InputField(
        desc="List of entities identified in the text."
    )
    covariates: List[EntityCovariateOutput] = dspy.OutputField(
        desc="Graph representation of the knowledge extracted from the text."
    )


class KnowledgeGraphExtractor(dspy.Module):
    def __init__(self, dspy_lm: dspy.LM):
        super().__init__()
        self.dspy_lm = dspy_lm
        self.prog_graph = Predict(ExtractGraphTriplet)
        self.prog_covariates = Predict(ExtractCovariate)

    def forward(self, text: str) -> AIKnowledgeGraph:
        with dspy.settings.context(lm=self.dspy_lm):
            pred_graph = self.prog_graph(
                text=text,
            )

            # Extract the covariates.
            entities_for_covariates = [
                EntityCovariateInput(
                    name=entity.name,
                    description=entity.description,
                )
                for entity in pred_graph.knowledge.entities
            ]

            pred_covariates = self.prog_covariates(
                text=text,
                entities=entities_for_covariates,
            )

            # Replace the entities with the covariates.
            for entity in pred_graph.knowledge.entities:
                for covariate in pred_covariates.covariates:
                    if entity.name == covariate.name:
                        entity.metadata = covariate.covariates

            return pred_graph
