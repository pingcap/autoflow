from typing import List, Sequence

from llama_index.core.question_gen.types import BaseQuestionGenerator, SubQuestion

from llama_index.core.tools.types import ToolMetadata

from app.models.chat_message import ChatMessage
from app.rag.indexes.knowledge_graph import IntentAnalyzer
from app.utils import dspy
from app.core.config import settings


class LLMQuestionAnalyzer(BaseQuestionGenerator):
    """
    LLMQuestionAnalyzer represents to analyze the intent of user, decompose the question into
    sub-questions, and decide using which way to retrieve the relevant documents.
    """

    def __init__(self, dspy_lm: dspy.LM) -> None:
        self._dspy_lm = dspy_lm
        self._intents = IntentAnalyzer(
            dspy_lm=dspy_lm,
            complied_program_path=settings.COMPLIED_INTENT_ANALYSIS_PROGRAM_PATH,
        )

    def generate(
        self,
        user_question: str,
        chat_history: List[ChatMessage],
        tools: Sequence[ToolMetadata],
    ) -> List[SubQuestion]:
        chat_content = self.get_chat_content(user_question, chat_history)

        intents = self._intents.analyze(chat_content)
        return [r.question for r in intents.questions]

    def get_chat_content(self, user_question: str, chat_history: List[ChatMessage]):
        if len(chat_history) > 0:
            chat_history_strings = [
                f"{message.role.value}: {message.content}" for message in chat_history
            ]
            query_with_history = (
                "++++ Chat History ++++\n"
                + "\n".join(chat_history_strings)
                + "++++ Chat History ++++\n"
            )
            chat_content = (
                query_with_history + "\n\nThen the user asksq:\n" + user_question
            )
        return chat_content
