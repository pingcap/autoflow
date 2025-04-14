import logging

import dspy
from dspy import Predict
from pydantic import BaseModel

from autoflow.knowledge_graph.types import GeneratedKnowledgeGraph

# Initialize logger
logger = logging.getLogger(__name__)


class EvaluateKnowledgeGraph(dspy.Signature):
    """
    Evaluate the differences between two knowledge graphs and provide scores for each entity and relationship,
    as well as a final score for the entire knowledge graph.

    Steps:
    1. Iterate over each entity in the expected knowledge graph
    2. For each expected entity, find the most similar entity in the actual knowledge graph
    3. Calculate the score (range from 0 to 1) for the entity based on the similarity
    4. Iterate over each relationship in the expected knowledge graph
    5. For each expected relationship, find the most similar relationship in the actual knowledge graph
    6. Calculate the score (range from 0 to 1) for the relationship based on the similarity
    7. Average all the scores of entities and relationships to get the final score

    Please only respond in JSON format.
    """

    actual_graph: GeneratedKnowledgeGraph = dspy.InputField(
        desc="The actual knowledge graph extracted from the text"
    )
    expected_graph: GeneratedKnowledgeGraph = dspy.InputField(
        desc="The expected knowledge graph"
    )
    score: float = dspy.OutputField(
        desc="The final score of the actual knowledge graph"
    )


class KGEvaluationResult(BaseModel):
    expected: GeneratedKnowledgeGraph
    actual: GeneratedKnowledgeGraph
    score: float


class KnowledgeGraphEvaluator(dspy.Module):
    def __init__(self, dspy_lm: dspy.LM):
        super().__init__()
        self.dspy_lm = dspy_lm
        self.program = Predict(EvaluateKnowledgeGraph)

    def forward(
        self,
        actual: GeneratedKnowledgeGraph,
        expected: GeneratedKnowledgeGraph,
    ) -> KGEvaluationResult:
        # Evaluate the knowledge graph using the provided language model
        with dspy.settings.context(lm=self.dspy_lm):
            prediction = self.program(actual_graph=actual, expected_graph=expected)
            return KGEvaluationResult(
                actual=actual,
                expected=expected,
                score=prediction.score,
            )
