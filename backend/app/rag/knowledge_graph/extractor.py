import logging
from copy import deepcopy
import pandas as pd
import dspy
from dspy.functional import TypedPredictor
from typing import Mapping, Optional, List
from llama_index.core.schema import BaseNode

from app.rag.knowledge_graph.schema import (
    Entity,
    Relationship,
    KnowledgeGraph,
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
    knowledge: KnowledgeGraph = dspy.OutputField(
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


def get_relation_metadata_from_node(node: BaseNode):
    metadata = deepcopy(node.metadata)
    for key in [
        "_node_content",
        "_node_type",
        "excerpt_keywords",
        "questions_this_excerpt_can_answer",
        "section_summary",
    ]:
        metadata.pop(key, None)
    metadata["chunk_id"] = node.node_id
    return metadata


class Extractor(dspy.Module):
    def __init__(self, dspy_lm: dspy.LM):
        super().__init__()
        self.dspy_lm = dspy_lm
        self.prog_graph = TypedPredictor(ExtractGraphTriplet)
        self.prog_covariates = TypedPredictor(ExtractCovariate)

    def get_llm_output_config(self):
        if "openai" in self.dspy_lm.provider.lower():
            return {
                "response_format": {"type": "json_object"},
            }
        elif "ollama" in self.dspy_lm.provider.lower():
            # ollama support set format=json in the top-level request config, but not in the request's option
            # https://github.com/ollama/ollama/blob/5e2653f9fe454e948a8d48e3c15c21830c1ac26b/api/types.go#L70
            return {}
        else:
            return {
                "response_mime_type": "application/json",
            }

    def forward(self, text):
        with dspy.settings.context(lm=self.dspy_lm):
            pred_graph = self.prog_graph(
                text=text,
                config=self.get_llm_output_config(),
            )

            # extract the covariates
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
                config=self.get_llm_output_config(),
            )

            # replace the entities with the covariates
            for entity in pred_graph.knowledge.entities:
                for covariate in pred_covariates.covariates:
                    if entity.name == covariate.name:
                        entity.metadata = covariate.covariates

            return pred_graph


class SimpleGraphExtractor:
    def __init__(
        self, dspy_lm: dspy.LM, complied_extract_program_path: Optional[str] = None
    ):
        self.extract_prog = Extractor(dspy_lm=dspy_lm)
        if complied_extract_program_path is not None:
            self.extract_prog.load(complied_extract_program_path)

    def extract(self, text: str, node: BaseNode):
        pred = self.extract_prog(text=text)
        metadata = get_relation_metadata_from_node(node)
        return self._to_df(
            pred.knowledge.entities, pred.knowledge.relationships, metadata
        )

    def _to_df(
        self,
        entities: list[Entity],
        relationships: list[Relationship],
        extra_meta: Mapping[str, str],
    ):
        # Create lists to store dictionaries for entities and relationships
        entities_data = []
        relationships_data = []

        # Iterate over parsed entities and relationships to create dictionaries
        for entity in entities:
            entity_dict = {
                "name": entity.name,
                "description": entity.description,
                "meta": entity.metadata,
            }
            entities_data.append(entity_dict)

        mapped_entities = {entity["name"]: entity for entity in entities_data}

        for relationship in relationships:
            source_entity_description = ""
            if relationship.source_entity not in mapped_entities:
                new_source_entity = {
                    "name": relationship.source_entity,
                    "description": (
                        f"Derived from from relationship: "
                        f"{relationship.source_entity} -> {relationship.relationship_desc} -> {relationship.target_entity}"
                    ),
                    "meta": {"status": "need-revised"},
                }
                entities_data.append(new_source_entity)
                mapped_entities[relationship.source_entity] = new_source_entity
                source_entity_description = new_source_entity["description"]
            else:
                source_entity_description = mapped_entities[relationship.source_entity][
                    "description"
                ]

            target_entity_description = ""
            if relationship.target_entity not in mapped_entities:
                new_target_entity = {
                    "name": relationship.target_entity,
                    "description": (
                        f"Derived from from relationship: "
                        f"{relationship.source_entity} -> {relationship.relationship_desc} -> {relationship.target_entity}"
                    ),
                    "meta": {"status": "need-revised"},
                }
                entities_data.append(new_target_entity)
                mapped_entities[relationship.target_entity] = new_target_entity
                target_entity_description = new_target_entity["description"]
            else:
                target_entity_description = mapped_entities[relationship.target_entity][
                    "description"
                ]

            relationship_dict = {
                "source_entity": relationship.source_entity,
                "source_entity_description": source_entity_description,
                "target_entity": relationship.target_entity,
                "target_entity_description": target_entity_description,
                "relationship_desc": relationship.relationship_desc,
                "meta": {
                    **extra_meta,
                },
            }
            relationships_data.append(relationship_dict)

        # Create DataFrames for entities and relationships
        entities_df = pd.DataFrame(entities_data)
        relationships_df = pd.DataFrame(relationships_data)
        return entities_df, relationships_df
