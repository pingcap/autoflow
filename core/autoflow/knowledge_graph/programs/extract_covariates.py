import logging
from typing import List, Mapping, Any

import dspy
from dspy import Predict
from pydantic import BaseModel, Field

from autoflow.knowledge_graph.types import GeneratedEntity


logger = logging.getLogger(__name__)


class InputEntity(BaseModel):
    """List of entities extracted from the text to form the knowledge graph"""

    name: str = Field(description="Name of the entity")
    description: str = Field(description="Description of the entity")


class OutputEntity(BaseModel):
    """List of entities extracted from the text to form the knowledge graph"""

    name: str = Field(description="Name of the entity")
    description: str = Field(description="Description of the entity")
    covariates: Mapping[str, Any] = Field(
        description=(
            "The attributes (which is a comprehensive json TREE, the first field is always: 'topic') to claim the entity. "
        )
    )


class ExtractEntityCovariate(dspy.Signature):
    """Please carefully review the provided text and entities list which are already identified in the text.
    Focusing on identifying detailed covariates associated with each entities provided.

    Extract and link the covariates (which is a comprehensive json TREE, the first field is always: "topic") to their respective entities.
    Ensure all extracted covariates is clearly connected to the correct entity for accuracy and comprehensive understanding.
    Ensure that all extracted covariates are factual and verifiable within the text itself, without relying on external knowledge or assumptions.
    Collectively, the covariates should provide a thorough and precise summary of the entity's characteristics as described in the source material.

    Please only response in JSON format.
    """

    text = dspy.InputField(
        desc="a paragraph of text to extract covariates to claim the entities."
    )
    input: List[InputEntity] = dspy.InputField(
        desc="List of entities identified in the text."
    )
    output: List[OutputEntity] = dspy.OutputField(
        desc="List of entities with their covariates."
    )


class EntityCovariateExtractor(dspy.Module):
    def __init__(self, dspy_lm: dspy.LM):
        super().__init__()
        self.dspy_lm = dspy_lm
        self.program = Predict(ExtractEntityCovariate)

    def forward(
        self, text: str, entities: List[GeneratedEntity]
    ) -> List[GeneratedEntity]:
        with dspy.settings.context(lm=self.dspy_lm):
            input_entities = [
                InputEntity(
                    name=entity.name,
                    description=entity.description,
                )
                for entity in entities
            ]

            predict = self.program(
                text=text,
                input=input_entities,
            )

            output_entity_map = {entity.name: entity for entity in predict.output}
            for entity in entities:
                if entity.name in output_entity_map:
                    # Update the covariates in the metadata of the entity.
                    entity.meta = output_entity_map[entity.name].covariates

            return entities
