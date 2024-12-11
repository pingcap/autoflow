import logging
import dspy
from dspy.functional import TypedChainOfThought, TypedPredictor
from typing import List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class Prerequisites(BaseModel):
    """Decomposed prerequisite questions extracted from the main query"""

    questions: List[str] = Field(
        description="List of prerequisite questions necessary for answering the main query."
    )


class DecomposePrerequisites(dspy.Signature):
    """You are an expert in query analysis and decomposition. Your task is to identify any prerequisite questions that need to be answered in order to fully address the main query.

    Step-by-Step Analysis:

    1. Analyze the main query to identify terms, references, or concepts that are unclear, ambiguous, or require additional information.
        - Focus on entities like versions, specific terminologies, or any references that may not be immediately clear.
    2. Formulate clear and concise prerequisite questions that, when answered, will provide the necessary information to address the main query effectively.

    ## Instructions:

    - Limit the number of prerequisite questions to no more than 3.
    - Ensure that the prerequisite questions are directly relevant and necessary for answering the main query.
    - Do not include unnecessary or unrelated questions.
    - Ensure that the questions are grounded and factual, based on the query provided.
    """

    query: str = dspy.InputField(
        desc="The main query text that needs to be analyzed for prerequisite questions."
    )
    prerequisites: Prerequisites = dspy.OutputField(
        desc="The decomposed prerequisite questions extracted from the main query."
    )


class DecomposePrerequisitesModule(dspy.Module):
    def __init__(self, dspy_lm: dspy.LM):
        super().__init__()
        self.dspy_lm = dspy_lm
        self.prog = TypedChainOfThought(DecomposePrerequisites)

    def forward(self, query):
        with dspy.settings.context(lm=self.dspy_lm):
            return self.prog(query=query)


class PrerequisiteAnalyzer:
    def __init__(self, dspy_lm: dspy.LM, compiled_program_path: Optional[str] = None):
        self.prerequisite_analysis_prog = DecomposePrerequisitesModule(dspy_lm=dspy_lm)
        if compiled_program_path is not None:
            self.prerequisite_analysis_prog.load(compiled_program_path)

    def analyze(self, query: str) -> Prerequisites:
        return self.prerequisite_analysis_prog(query=query).prerequisites
