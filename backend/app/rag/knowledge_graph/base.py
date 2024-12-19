import dspy
import logging
import traceback
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple
from collections import defaultdict
import concurrent.futures
from sqlmodel import Session

from llama_index.core.data_structs import IndexLPG
from llama_index.core.callbacks import CallbackManager, trace_method
from llama_index.core.callbacks.schema import CBEventType, EventPayload
from llama_index.core.indices.base import BaseIndex
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import BaseNode, TransformComponent
import llama_index.core.instrumentation as instrument

from app.rag.knowledge_graph.extractor import SimpleGraphExtractor
from app.rag.knowledge_graph.intent import IntentAnalyzer
from app.rag.knowledge_graph.prerequisite import PrerequisiteAnalyzer
from app.rag.types import MyCBEventType
from app.core.config import settings
from app.core.db import Scoped_Session

logger = logging.getLogger(__name__)

dispatcher = instrument.get_dispatcher(__name__)


class KnowledgeGraphStore(ABC):
    @abstractmethod
    def save(self, entities_df, relationships_df) -> None:
        """Upsert entities and relationships to the graph store."""
        pass

    @abstractmethod
    def retrieve_with_weight(
        self,
        query: str,
        embedding: list,
        depth: int = 2,
        include_meta: bool = False,
        with_degree: bool = False,
        with_chunks: bool = True,
        relationship_meta_filters: Dict = {},
        session: Optional[Session] = None,
    ) -> Tuple[list, list, list]:
        """Retrieve nodes and relationships with weights."""
        pass


class KnowledgeGraphIndex(BaseIndex[IndexLPG]):
    """An index for a property graph.

    Args:
        nodes (Optional[Sequence[BaseNode]]):
            A list of nodes to insert into the index.
        dspy_lm (dspy.BaseLLM):
            The language model of dspy to use for extracting triplets.
        callback_manager (Optional[CallbackManager]):
            The callback manager to use.
        transformations (Optional[List[TransformComponent]]):
            A list of transformations to apply to the nodes before inserting them into the index.
            These are applied prior to the `kg_extractors`.
        storage_context (Optional[StorageContext]):
            The storage context to use.
        show_progress (bool):
            Whether to show progress bars for transformations. Defaults to `False`.
    """

    index_struct_cls = IndexLPG

    def __init__(
        self,
        dspy_lm: dspy.LM,
        kg_store: KnowledgeGraphStore,
        nodes: Optional[Sequence[BaseNode]] = None,
        # parent class params
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> None:
        self._dspy_lm = dspy_lm
        self._kg_store = kg_store
        self._intents = IntentAnalyzer(
            dspy_lm=dspy_lm,
            complied_program_path=settings.COMPLIED_INTENT_ANALYSIS_PROGRAM_PATH,
        )
        self._prerequisites_analyzer = PrerequisiteAnalyzer(
            dspy_lm=dspy_lm,
            compiled_program_path=settings.COMPLIED_PREREQUISITE_ANALYSIS_PROGRAM_PATH,
        )

        super().__init__(
            nodes=nodes,
            callback_manager=callback_manager,
            storage_context=storage_context,
            transformations=transformations,
            show_progress=show_progress,
            **kwargs,
        )

    @classmethod
    def from_existing(
        cls: "KnowledgeGraphIndex",
        dspy_lm: dspy.LM,
        kg_store: KnowledgeGraphStore,
        callback_manager: Optional[CallbackManager] = None,
        transformations: Optional[List[TransformComponent]] = None,
        storage_context: Optional[StorageContext] = None,
        show_progress: bool = False,
        **kwargs: Any,
    ) -> "KnowledgeGraphIndex":
        return cls(
            dspy_lm=dspy_lm,
            kg_store=kg_store,
            nodes=[],  # no nodes to insert
            callback_manager=callback_manager,
            transformations=transformations,
            storage_context=storage_context,
            show_progress=show_progress,
            **kwargs,
        )

    def _insert_nodes(self, nodes: Sequence[BaseNode]):
        """Insert nodes to the index struct."""
        if len(nodes) == 0:
            return nodes

        extractor = SimpleGraphExtractor(dspy_lm=self._dspy_lm)
        for node in nodes:
            entities_df, rel_df = extractor.extract(
                text=node.get_content(),
                node=node,
            )
            self._kg_store.save(node.node_id, entities_df, rel_df)

    def _build_index_from_nodes(self, nodes: Optional[Sequence[BaseNode]]) -> IndexLPG:
        """Build index from nodes."""
        nodes = self._insert_nodes(nodes or [])
        return IndexLPG()

    def as_retriever(self, **kwargs: Any):
        """Return a retriever for the index."""
        # Our retriever params is more complex than the base retriever,
        # so we can't use the base retriever.
        raise NotImplementedError(
            "Retriever not implemented for KnowledgeGraphIndex, use `retrieve_with_weight` instead."
        )

    def retrieve_with_weight(
        self,
        query: str,
        embedding: list,
        depth: int = 2,
        include_meta: bool = False,
        with_degree: bool = False,
        with_chunks: bool = False,
        # experimental feature to filter relationships based on meta, can be removed in the future
        relationship_meta_filters: Dict = {},
    ) -> Tuple[list, list, list]:
        """Retrieve entities, relations, and chunks with weights."""
        params = {
            "query": query,
            "depth": depth,
            "include_meta": include_meta,
            "with_degree": with_degree,
            "with_chunks": with_chunks,
            "relationship_meta_filters": relationship_meta_filters,
        }
        with self._callback_manager.as_trace("retrieve_with_weight"):
            with self._callback_manager.event(
                MyCBEventType.RETRIEVE_FROM_GRAPH,
                payload={EventPayload.QUERY_STR: params},
            ) as event:
                entities, relations, chunks = self._kg_store.retrieve_with_weight(
                    query,
                    embedding,
                    depth,
                    include_meta,
                    with_degree,
                    with_chunks,
                    relationship_meta_filters,
                )
                event.on_end(
                    payload={
                        "entities": entities,
                        "relationships": relations,
                        "chunks": chunks,
                    },
                )
        return entities, relations, chunks

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        raise NotImplementedError(
            "Delete node not implemented for KnowledgeGraphIndex."
        )

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Index-specific logic for inserting nodes to the index struct."""
        self._insert_nodes(nodes)

    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        raise NotImplementedError(
            "Ref doc info not implemented for KnowledgeGraphIndex. "
            "All inserts are already upserts."
        )

    def intent_analyze(
        self,
        query: str,
        chat_history: list = [],
    ) -> List[str]:
        """Analyze the intent of the query."""
        chat_content = query
        if len(chat_history) > 0:
            chat_history_strings = [
                f"{message.role.value}: {message.content}" for message in chat_history
            ]
            query_with_history = (
                "++++ Chat History ++++\n"
                + "\n".join(chat_history_strings)
                + "++++ Chat History ++++\n"
            )
            chat_content = query_with_history + "\n\nThen the user asksq:\n" + query

        with self._callback_manager.as_trace("intent_based_search"):
            with self._callback_manager.event(
                MyCBEventType.INTENT_DECOMPOSITION,
                payload={EventPayload.QUERY_STR: chat_content},
            ) as event:
                intents = self._intents.analyze(chat_content)
                semantic_queries = [r.question for r in intents.questions]
                event.on_end(payload={"semantic_queries": semantic_queries})

        return semantic_queries

    def prerequisite_analyze(
        self,
        query: str,
        chat_history: list = [],
    ) -> List[str]:
        """Analyze the prerequisite of the query."""
        chat_content = query
        if len(chat_history) > 0:
            chat_history_strings = [
                f"{message.role.value}: {message.content}" for message in chat_history
            ]
            query_with_history = (
                "++++ Chat History ++++\n"
                + "\n".join(chat_history_strings)
                + "++++ Chat History ++++\n"
            )
            chat_content = query_with_history + "\n\nThen the user asksq:\n" + query

        with self._callback_manager.as_trace("prerequisites_based_search"):
            with self._callback_manager.event(
                MyCBEventType.INTENT_DECOMPOSITION,
                payload={EventPayload.QUERY_STR: chat_content},
            ) as event:
                prerequisites = self._prerequisites_analyzer.analyze(chat_content)
                event.on_end(payload={"semantic_queries": prerequisites.questions})
        return prerequisites.questions

    def graph_semantic_search(
        self,
        semantic_queries: List[str],
        depth: int = 2,
        include_meta: bool = False,
        relationship_meta_filters: Dict = {},
    ) -> Mapping[str, Any]:
        result = {"queries": {}, "graph": None}
        all_entities = []
        all_relationships = defaultdict(
            lambda: {
                "id": None,
                "source_entity_id": None,
                "target_entity_id": None,
                "description": None,
                "rag_description": None,
                "meta": None,
                "weight": 0,
                "last_modified_at": None,
            }
        )

        def add_relationships(relationships):
            for r in relationships:
                key = (r["source_entity_id"], r["target_entity_id"], r["description"])
                all_relationships[key]["weight"] += r["weight"]
                if all_relationships[key]["id"] is None:
                    all_relationships[key].update(
                        {
                            "id": r["id"],
                            "source_entity_id": r["source_entity_id"],
                            "target_entity_id": r["target_entity_id"],
                            "description": r["description"],
                            "rag_description": r["rag_description"],
                            "meta": r["meta"],
                            "last_modified_at": r["last_modified_at"],
                        }
                    )

        def process_query(sub_query):
            logger.info(f"Processing query: {sub_query}")
            tmp_session = Scoped_Session()
            try:
                entities, relationships, _ = self._kg_store.retrieve_with_weight(
                    sub_query,
                    [],
                    depth,
                    include_meta,
                    with_chunks=False,
                    relationship_meta_filters=relationship_meta_filters,
                    session=tmp_session,
                )
            except Exception as exc:
                tmp_session.rollback()
                traceback.print_exc()
                raise
            finally:
                tmp_session.close()
                Scoped_Session.remove()
            return sub_query, entities, relationships

        with self._callback_manager.as_trace("intent_based_search"):
            with self._callback_manager.event(
                MyCBEventType.GRAPH_SEMANTIC_SEARCH,
                payload={EventPayload.QUERY_STR: semantic_queries},
            ) as event:
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future_to_query = {
                        executor.submit(process_query, sub_query): sub_query
                        for sub_query in semantic_queries
                    }
                    for future in concurrent.futures.as_completed(future_to_query):
                        sub_query = future_to_query[future]
                        try:
                            # It can't record the real execution time of the sub_query
                            sub_query, entities, relationships = future.result()
                            result["queries"][sub_query] = {
                                "entities": entities,
                                "relationships": relationships,
                            }
                            all_entities.extend(entities)
                            add_relationships(relationships)
                        except Exception as exc:
                            logger.error(f"{sub_query} generated an exception: {exc}")

                event.on_end(
                    payload={
                        "queries": result["queries"],
                    },
                )

        unique_entities = {e["id"]: e for e in all_entities}.values()

        result["graph"] = {
            "entities": list(unique_entities),
            "relationships": [
                {
                    "id": v["id"],
                    "source_entity_id": v["source_entity_id"],
                    "target_entity_id": v["target_entity_id"],
                    "description": v["description"],
                    "rag_description": v["rag_description"],
                    "meta": v["meta"],
                    "weight": v["weight"],
                    "last_modified_at": v["last_modified_at"],
                }
                for v in all_relationships.values()
            ],
        }

        return result
