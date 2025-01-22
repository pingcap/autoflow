from enum import Enum
from typing import List, Optional, Tuple

from llama_index.core import QueryBundle
from llama_index.core.llms import LLM
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.base.base_selector import BaseSelector
from llama_index.core.tools import ToolMetadata
from llama_index.core.selectors import LLMSingleSelector, LLMMultiSelector

from app.api.routes.chat import logger


class KBSelectMode(str, Enum):
    ALL = "ALL"
    MULTIPLE_SELECTION = "MULTIPLE_SELECTION"
    SINGLE_SECTION = "SINGLE_SECTION"


class MultiKBSelector:
    _selector: Optional[BaseSelector] = None
    _retrievers: list[BaseRetriever] = list
    _retriever_choices: list[ToolMetadata] = list

    def __init__(
        self,
        llm: LLM,
        select_mode: KBSelectMode = KBSelectMode.ALL,
        retrievers: List[BaseRetriever] = list,
        retriever_choices: List[ToolMetadata] = list,
    ):
        if select_mode == KBSelectMode.ALL:
            self._selector = None
        if select_mode == KBSelectMode.MULTIPLE_SELECTION:
            self._selector = LLMMultiSelector.from_defaults(llm=llm)
        if select_mode == KBSelectMode.SINGLE_SECTION:
            self._selector = LLMSingleSelector.from_defaults(llm=llm)

        self._retrievers = retrievers
        self._retriever_choices = retriever_choices

    def select(self, query: QueryBundle) -> List[Tuple[BaseRetriever, int]]:
        if len(self._retrievers) == 0:
            raise ValueError("No retriever selected")

        if self._selector is None or len(self._retrievers) == 1:
            return [(self._retrievers[0], 0)]

        result = self._selector.select(self._retriever_choices, query)
        if len(result.selections) == 0:
            raise ValueError("No selection selected")

        retrievers = []
        for selection in result.selections:
            retrievers.append((self._retrievers[selection.index], selection.index))

        return retrievers

    async def aselect(self, query: QueryBundle) -> List[Tuple[BaseRetriever, int]]:
        if len(self._retrievers) == 0:
            raise ValueError("No retriever selected")

        if len(self._retrievers) == 1 or len(self._retrievers) == 1:
            return [(self._retrievers[0], 0)]

        result = await self._selector.aselect(self._retriever_choices, query)
        if len(result.selections) == 0:
            raise ValueError("No selection selected")

        retrievers = []
        for selection in result.selections:
            retrievers.append((self._retrievers[selection.index], selection.index))

        logger.info(f"Selected {len(retrievers)} retrievers for query '{query}'")
        return retrievers
