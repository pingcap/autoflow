import dspy

from typing import List, Optional, Dict, Tuple

from llama_index.core import QueryBundle
from llama_index.core.async_utils import run_async_tasks
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.callbacks import CallbackManager, EventPayload
from llama_index.core.llms import LLM
from llama_index.core.schema import NodeWithScore
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import ToolMetadata
from sqlmodel import Session

from app.core.config import Settings
from app.core.db import engine
from app.rag.indices.vector_search.retriever.base_retriever import VectorSearchRetriever
from app.rag.indices.vector_search.schema import (
    VectorSearchRetrieverConfig,
    RetrievedChunk,
)
from app.rag.question_gen.query_decomposer import QueryDecomposer
from app.rag.types import MyCBEventType
from app.repositories import knowledge_base_repo


class VectorSearchFusionRetriever(BaseRetriever):
    def __init__(
        self,
        knowledge_base_ids: List[int],
        config: VectorSearchRetrieverConfig,
        llm: LLM,
        dspy_lm: dspy.LM,
        callback_manager: Optional[CallbackManager] = CallbackManager([]),
        use_async: bool = True,
        use_query_decompose: bool = True,
        use_query_router: bool = True,
        **kwargs,
    ):
        super().__init__(callback_manager, **kwargs)
        self._config = config
        self._callback_manager = callback_manager

        with Session(engine) as session:
            self._retrievers = []
            self._retrievers_metadata = []
            self.use_async = use_async
            self.use_query_decompose = use_query_decompose
            self._query_decomposer = QueryDecomposer(
                dspy_lm=dspy_lm,
                # TODO: move to arguments of the constructor
                complied_program_path=Settings.COMPLIED_INTENT_ANALYSIS_PROGRAM_PATH,
            )
            self.use_query_router = use_query_router
            self.selector = LLMSingleSelector.from_defaults(llm=llm)

            for knowledge_base_id in knowledge_base_ids:
                kb = knowledge_base_repo.get(session, knowledge_base_id)
                self._retrievers.append(
                    VectorSearchRetriever(
                        knowledge_base_id=knowledge_base_id,
                        config=config,
                        callback_manager=self._callback_manager,
                    )
                )
                self._retrievers_metadata.append(
                    ToolMetadata(
                        name=kb.name,
                        description=kb.description,
                    )
                )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self.use_query_decompose:
            queries = self._gen_sub_queries(query_bundle)
        else:
            queries = [query_bundle]

        if self.use_async:
            results = self._run_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)
        return self._simple_fusion(results)

    def retrieve_chunks(
        self, query_bundle: QueryBundle, db_session: Session
    ) -> List[RetrievedChunk]:
        nodes_with_score = self._retrieve(query_bundle)
        return self.map_nodes_to_chunks(nodes_with_score, db_session)

    def map_nodes_to_chunks(
        self, nodes_with_score, db_session: Optional[Session] = None
    ):
        chunk_ids = [ns.node.node_id for ns in nodes_with_score]
        chunk_to_document_map = self._get_chunk_to_document_map(chunk_ids, db_session)

        return [
            RetrievedChunk(
                id=ns.node.node_id,
                text=ns.node.text,
                metadata=ns.node.metadata,
                document=chunk_to_document_map[ns.node.node_id],
                score=ns.score,
            )
            for ns in nodes_with_score
        ]

    def _gen_sub_queries(self, query_bundle: QueryBundle) -> List[QueryBundle]:
        """
        Decompose the query into subqueries.
        """
        with self._callback_manager.event(
            MyCBEventType.INTENT_DECOMPOSITION,
            payload={EventPayload.QUERY_STR: query_bundle.query_str},
        ) as event:
            queries = self._query_decomposer.decompose(query_bundle.query_str)
            subqueries = [QueryBundle(r.question) for r in queries.questions]
            event.on_end(payload={"queries": queries})
        return subqueries

    def _run_async_queries(
        self, queries: List[QueryBundle]
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        tasks, task_queries = [], []
        for query in queries:
            query_str = query.query_str
            retriever, i = self._select_retriever(query_str)
            tasks.append(retriever.aretrieve(query))
            task_queries.append((query_str, i))

        task_results = run_async_tasks(tasks)

        results = {}
        for query_tuple, query_result in zip(task_queries, task_results):
            results[query_tuple] = query_result

        return results

    def _run_sync_queries(
        self, queries: List[QueryBundle]
    ) -> Dict[Tuple[str, int], List[NodeWithScore]]:
        results = {}
        for query in queries:
            query_str = query.query_str
            retriever, i = self._select_retriever(query_str)
            results[(query_str, i)] = retriever.retrieve(query)

        return results

    def _select_retriever(self, query_str: str) -> Tuple[BaseRetriever, int]:
        """
        Using the LLM to select the appropriate retriever based on the query string.

        Args:
            query_str: the query string

        Returns:
            retriever: the retriever to use
            i: the index of the retriever

        """
        if len(self._retrievers) == 0:
            raise ValueError("No retriever selected")
        if len(self._retrievers) == 1:
            return self._retrievers[0], 0
        result = self.selector.select(self._retrievers_metadata, query_str)
        if len(result.selections) == 0:
            raise ValueError("No selection selected")
        i = result.selections[0].index
        return self._retrievers[i], i

    def _simple_fusion(
        self, results: Dict[Tuple[str, int], List[NodeWithScore]]
    ) -> List[NodeWithScore]:
        """Apply simple fusion."""
        # Use a dict to de-duplicate nodes
        all_nodes: Dict[str, NodeWithScore] = {}
        for nodes_with_scores in results.values():
            for node_with_score in nodes_with_scores:
                hash = node_with_score.node.hash
                if hash in all_nodes:
                    max_score = max(
                        node_with_score.score or 0.0, all_nodes[hash].score or 0.0
                    )
                    all_nodes[hash].score = max_score
                else:
                    all_nodes[hash] = node_with_score

        return sorted(all_nodes.values(), key=lambda x: x.score or 0.0, reverse=True)
