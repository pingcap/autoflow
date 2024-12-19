import logging
from typing import Optional, Sequence
from llama_index.llms.openai import OpenAI
from llama_index.core.tools.types import BaseTool
from llama_index.core.tools import FunctionTool

logger = logging.getLogger(__name__)

DefaultSystemPrompt = """
You are a highly skilled customer assistant, responsible for dispatching user questions to the most appropriate tools or resources. Your primary objective is to ensure each user question is handled accurately and efficiently by selecting the best-suited tool for the task.
For more complex questions, you should break them down into clear, manageable sub-questions and route each to the relevant tools for individual resolution. It's important to maintain clarity and precision in this process, ensuring that the sub-questions are well-defined and can be resolved independently.
If you encounter concepts or entities you are not familiar with, you can break the query down into a sub-question to clarify the specific concept or entity. For example, if the query involves “what is the latest version,” you can treat this as a sub-question to better understand the context before proceeding with the solution.
"""


class QueryDispatcher:
    def __init__(self, llm: OpenAI, system_prompt: Optional[str] = None):
        if system_prompt is None:
            system_prompt = DefaultSystemPrompt

        self._llm = llm
        self._llm.system_prompt = system_prompt

    def route(self, query: str, tools: Sequence["BaseTool"]) -> str:
        response = self._llm.chat_with_tools(
            tools, query, allow_parallel_tool_calls=True, verbose=True
        )

        try:
            tool_calls = self._llm.get_tool_calls_from_response(
                response, error_on_no_tool_call=True
            )
        except Exception as e:
            logger.exception(e)
            return f"An error occurred while processing the query: {query}"

        return tool_calls


# mock the answer process
def answer(query: str) -> str:
    """
    Answer a user query. The query should be simple and straightforward.
    """
    return f"I need some time to answer your question: {query}."


answer_tool = FunctionTool.from_defaults(fn=answer)
