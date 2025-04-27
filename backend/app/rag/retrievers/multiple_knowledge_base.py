from abc import abstractmethod

import dspy

from typing import List, Optional, Dict, Tuple

from llama_index.core import QueryBundle
from llama_index.core.async_utils import run_async_tasks
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from pydantic import BaseModel
from sqlmodel import Session

from app.core.config import settings
from app.rag.question_gen.query_decomposer import QueryDecomposer
from app.rag.types import MyCBEventType
from app.rag.llms.dspy import get_dspy_lm_by_llama_llm


class FusionRetrievalBaseConfig(BaseModel):
    llm_id: Optional[int] = None
    knowledge_base_ids: List[int]
    use_query_decompose: Optional[bool] = None


class MultiKBFusionRetriever(BaseRetriever):
    def __init__(
        self,
        retrievers: List[BaseRetriever],
        db_session: Session,
        llm: LLM,
        dspy_lm: Optional[dspy.LM] = None,
        use_query_decompose: bool = True,
        callback_manager: Optional[CallbackManager] = CallbackManager([]),
        **kwargs,
    ):
        super().__init__(callback_manager, **kwargs)
        self._use_query_decompose = use_query_decompose
        self._db_session = db_session
        self._callback_manager = callback_manager

        # Setup query decomposer.
        self._dspy_lm = dspy_lm or get_dspy_lm_by_llama_llm(llm)
        self._query_decomposer = QueryDecomposer(
            dspy_lm=self._dspy_lm,
            complied_program_path=settings.COMPLIED_INTENT_ANALYSIS_PROGRAM_PATH,
        )

        # Setup multiple knowledge base selector.
        self._retrievers = retrievers

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self._use_query_decompose:
            queries = self._gen_sub_queries(query_bundle)
        else:
            queries = [query_bundle]

        with self.callback_manager.event(
            MyCBEventType.RUN_SUB_QUERIES, payload={"queries": queries}
        ):
            tasks, task_queries = [], []

            for query in queries:
                for i, retriever in enumerate(self._retrievers):
                    tasks.append(retriever.aretrieve(query.query_str))
                    task_queries.append((query.query_str, i))

            task_results = run_async_tasks(tasks)
            results = {}
            for query_tuple, query_result in zip(task_queries, task_results):
                results[query_tuple] = query_result

        return self._fusion(query_bundle.query_str, results)

    def _gen_sub_queries(self, query_bundle: QueryBundle) -> List[QueryBundle]:
        queries = self._query_decomposer.decompose(query_bundle.query_str)
        return [QueryBundle(r.question) for r in queries.questions]

    @abstractmethod
    def _fusion(
        self, query: str, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        """fusion method"""
