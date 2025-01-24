import dspy
import logging

from typing import Any, Dict, List, Optional, Sequence
from llama_index.core.data_structs import IndexLPG
from llama_index.core.callbacks import CallbackManager
from llama_index.core.indices.base import BaseIndex
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.storage.storage_context import StorageContext
from llama_index.core.schema import BaseNode, TransformComponent
import llama_index.core.instrumentation as instrument
from app.rag.indices.knowledge_graph.extractor import SimpleGraphExtractor
from app.rag.indices.knowledge_graph.graph_store import KnowledgeGraphStore


logger = logging.getLogger(__name__)

dispatcher = instrument.get_dispatcher(__name__)


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
        **kwargs: Any,
    ) -> None:
        self._dspy_lm = dspy_lm
        self._kg_store = kg_store
        super().__init__(
            nodes=nodes,
            callback_manager=callback_manager,
            **kwargs,
        )

    @classmethod
    def from_existing(
        cls: "KnowledgeGraphIndex",
        dspy_lm: dspy.LM,
        kg_store: KnowledgeGraphStore,
        # parent class params
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

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Index-specific logic for inserting nodes to the index struct."""
        self._insert_nodes(nodes)

    def ref_doc_info(self) -> Dict[str, RefDocInfo]:
        """Retrieve a dict mapping of ingested documents and their nodes+metadata."""
        raise NotImplementedError(
            "Ref doc info not implemented for KnowledgeGraphIndex. "
            "All inserts are already upserts."
        )

    def _delete_node(self, node_id: str, **delete_kwargs: Any) -> None:
        """Delete a node."""
        raise NotImplementedError(
            "Delete node not implemented for KnowledgeGraphIndex."
        )
