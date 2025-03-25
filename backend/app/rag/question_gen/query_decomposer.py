import logging
import dspy
from typing import List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SubQuestion(BaseModel):
    """Representation of a single step-by-step question extracted from the user query."""

    question: str = Field(
        description="A step-by-step question to address the user query."
    )
    reasoning: str = Field(
        description="The rationale behind the question to explain its relevance."
    )


class SubQuestions(BaseModel):
    """Representation of the user's step-by-step questions extracted from the query."""

    questions: List[SubQuestion] = Field(
        description="List of questions representing a plan to address the user query."
    )


class DecomposeQuery(dspy.Signature):
    """You are an expert in knowledge base graph construction, specializing in building comprehensive knowledge graphs.
    Your current task is to deconstruct the user's query into a series of step-by-step questions.

    ## Instructions:

    1. Dependency Analysis:

        - Analyze the user's query to identify the underlying dependencies and relationships between different components.
        - Construct a dependency graph that visually represents these relationships.

    2. Question Breakdown: Divide the query into a sequence of step-by-step questions necessary to address the main query comprehensively.

    3. Provide Reasoning: Explain the rationale behind each question.

    4. Constraints:
        - Limit the output to no more than 5 questions to maintain focus and relevance.
        - Ensure accuracy by reflecting the user's true intentions based on the provided query.
        - Ground all questions in factual information derived directly from the user's input.

    Please only response in JSON format.
    """

    query: str = dspy.InputField(
        desc="The query text to extract the user's step-by-step questions."
    )
    subquestions: SubQuestions = dspy.OutputField(
        desc="Representation of the user's step-by-step questions extracted from the query."
    )


class DecomposeQueryModule(dspy.Module):
    def __init__(self, dspy_lm: dspy.LM):
        super().__init__()
        self.dspy_lm = dspy_lm
        self.prog = dspy.Predict(DecomposeQuery)

    def forward(self, query):
        with dspy.settings.context(lm=self.dspy_lm):
            return self.prog(query=query)


class QueryDecomposer:
    def __init__(self, dspy_lm: dspy.LM, complied_program_path: Optional[str] = None):
        self.decompose_query_prog = DecomposeQueryModule(dspy_lm=dspy_lm)
        if complied_program_path is not None:
            self.decompose_query_prog.load(complied_program_path)

    def decompose(self, query: str) -> SubQuestions:
        return self.decompose_query_prog(query=query).subquestions
