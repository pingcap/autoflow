import logging
from typing import List, Optional, Dict, Tuple

import dspy
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
from app.rag.question_gen.query_decomposer import QueryDecomposer
from app.rag.indices.knowledge_graph.retriever.base_retriever import (
    KnowledgeGraphRetriever,
)
from app.rag.indices.knowledge_graph.retriever.config import (
    KnowledgeGraphRetrieverConfig,
)
from app.rag.indices.knowledge_graph.schema import (
    KnowledgeGraphNode,
    Entity,
    Relationship,
)
from app.rag.types import MyCBEventType
from app.repositories import knowledge_base_repo

logger = logging.getLogger(__name__)


class KnowledgeGraphFusionRetriever(BaseRetriever):
    _query_decomposer: QueryDecomposer
    _retrievers: List[KnowledgeGraphRetriever]

    def __init__(
        self,
        knowledge_base_ids: List[int],
        config: KnowledgeGraphRetrieverConfig,
        llm: LLM,
        dspy_lm: dspy.LM,
        callback_manager: Optional[CallbackManager] = CallbackManager([]),
        use_query_decompose: bool = False,
        use_async: bool = True,
        **kwargs,
    ):
        super().__init__(callback_manager, **kwargs)
        with Session(engine) as session:
            self.use_async = use_async
            self.use_query_decompose = use_query_decompose
            self._callback_manager = callback_manager
            self._query_decomposer = QueryDecomposer(
                dspy_lm=dspy_lm,
                # TODO: move to arguments of the constructor
                complied_program_path=Settings.COMPLIED_INTENT_ANALYSIS_PROGRAM_PATH,
            )
            self.selector = LLMSingleSelector.from_defaults(llm=llm)
            self._retrievers = []
            self._retrievers_metadata = []
            for knowledge_base_id in knowledge_base_ids:
                kb = knowledge_base_repo.get(session, knowledge_base_id)
                self._retrievers.append(
                    KnowledgeGraphRetriever(
                        dspy_lm=dspy_lm,
                        knowledge_base_id=knowledge_base_id,
                        config=config,
                        callback_manager=callback_manager,
                    )
                )
                self._retrievers_metadata.append(
                    ToolMetadata(
                        name=kb.name,
                        description=kb.description,
                    )
                )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        entities, relationships = self.retrieve_knowledge_graph(query_bundle)
        return [
            NodeWithScore(
                node=KnowledgeGraphNode(
                    entities=entities,
                    relationships=relationships,
                ),
                score=1,
            )
        ]

    def retrieve_knowledge_graph(
        self, query_bundle: QueryBundle
    ) -> Tuple[List[Entity], List[Relationship]]:
        if self.use_query_decompose:
            queries = self._gen_sub_queries(query_bundle)
        else:
            queries = [query_bundle]

        if self.use_async:
            results = self._run_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)
        return self._simple_fusion(results)

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
    ) -> Dict[Tuple[str, int], List[KnowledgeGraphNode]]:
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
    ) -> Dict[Tuple[str, int], List[KnowledgeGraphNode]]:
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
        self, results: Dict[Tuple[str, int], List[KnowledgeGraphNode]]
    ) -> Tuple[List[Entity], List[Relationship]]:
        merged_entities = {}
        merged_relationships = {}
        for nodes_with_scores in results.values():
            if len(nodes_with_scores) == 0:
                continue
            node = nodes_with_scores[0].node
            for e in node.entities:
                if merged_entities[e["id"]] is None:
                    merged_entities[e["id"]] = e

            for r in node.relationships:
                key = (r["source_entity_id"], r["target_entity_id"], r["description"])
                if merged_relationships[key] is None:
                    merged_relationships[key] = {
                        "id": r["id"],
                        "source_entity_id": r["source_entity_id"],
                        "target_entity_id": r["target_entity_id"],
                        "description": r["description"],
                        "rag_description": r["rag_description"],
                        "weight": 0,
                        "meta": r["meta"],
                        "last_modified_at": r["last_modified_at"],
                    }
                else:
                    merged_relationships[key]["weight"] += r["weight"]

        return (list(merged_entities.values()), list(merged_relationships.values()))
