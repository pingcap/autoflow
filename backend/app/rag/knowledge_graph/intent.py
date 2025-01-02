import logging
import dspy
from dspy.functional import TypedChainOfThought, TypedPredictor
from typing import List, Optional
from pydantic import BaseModel, Field
from llama_index.core.tools import FunctionTool

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
        self.prog = TypedPredictor(DecomposeQuery)

    def forward(self, query):
        with dspy.settings.context(lm=self.dspy_lm):
            return self.prog(query=query)


class IntentAnalyzer:
    def __init__(self, dspy_lm: dspy.LM, complied_program_path: Optional[str] = None):
        self.intent_anlysis_prog = DecomposeQueryModule(dspy_lm=dspy_lm)
        if complied_program_path is not None:
            self.intent_anlysis_prog.load(complied_program_path)

    def analyze(self, query: str) -> SubQuestions:
        return self.intent_anlysis_prog(query=query).subquestions

    def as_tool_call(self):
        def breakDownQuery(query: str) -> list[str]:
            """
            Break down the complex user query into sub-questions and solve them step-by-step.

            If the query is complex, this function decomposes it into smaller sub-questions
            and solves them individually. For simple and straightforward queries,
            avoid calling this function.

            Args:
            query (str): The user's input query.

            Returns:
            str: The final result after solving all sub-questions.
            """

            return [query]

        return FunctionTool.from_defaults(fn=breakDownQuery)
