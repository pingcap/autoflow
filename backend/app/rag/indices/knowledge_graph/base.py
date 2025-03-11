import dspy
import logging

from typing import Any, Dict, List, Optional, Sequence
from llama_index.core.data_structs import IndexLPG
from llama_index.core.indices.base import BaseIndex
from llama_index.core.storage.docstore.types import RefDocInfo
from llama_index.core.schema import BaseNode
import llama_index.core.instrumentation as instrument
from sqlmodel import Session

from app.rag.indices.knowledge_graph.extractor import Extractor
from app.rag.indices.knowledge_graph.graph_store import TiDBGraphStore
from app.rag.indices.knowledge_graph.schema import AIEntity, EntityCreate

logger = logging.getLogger(__name__)

dispatcher = instrument.get_dispatcher(__name__)


class KnowledgeGraphIndex(BaseIndex[IndexLPG]):
    """An index for a knowledge graph.

    Args:
        dspy_lm (dspy.BaseLLM):
            The language model of dspy to use for extracting triplets.
    """

    index_struct_cls = IndexLPG

    def __init__(
        self,
        dspy_lm: dspy.LM,
        kg_store: TiDBGraphStore,
        **kwargs: Any,
    ) -> None:
        self._dspy_lm = dspy_lm
        self._kg_store = kg_store
        self._kg_extractor = Extractor(dspy_lm=self._dspy_lm)
        super().__init__(
            **kwargs,
        )

    @classmethod
    def from_existing(
        cls,
        dspy_lm: dspy.LM,
        kg_store: TiDBGraphStore,
        **kwargs: Any,
    ) -> "KnowledgeGraphIndex":
        return cls(
            dspy_lm=dspy_lm,
            kg_store=kg_store,
            **kwargs,
        )

    def insert_nodes(self, db_session: Session, nodes: Sequence[BaseNode]):
        """Insert nodes to the index struct."""
        if len(nodes) == 0:
            return nodes

        for node in nodes:
            self._inert_node(db_session, node)

    def _inert_node(self, db_session: Session, node: BaseNode):
        node_id = node.node_id
        logger.info("Extracting entities and relationships for node %s", node_id)

        knowledge_graph = self._kg_extractor.forward(text=node.get_content())
        if knowledge_graph.entities is None or knowledge_graph.relationships is None:
            logger.warning(
                f"Entities or relationships of node {node_id} are empty, not need to insert to index."
            )
            return

        if self._kg_store.exists_chunk_relationships(node_id):
            logger.info(
                f"Node #{node_id} already exists in the relationship table, skip."
            )
            return

        for extracted_entity in knowledge_graph.entities:
            self._kg_store.find_or_create_entity(
                EntityCreate(
                    name=extracted_entity.name,
                    description=extracted_entity.description,
                    meta=extracted_entity.meta,
                ),
                commit=False,
            )

        for r in knowledge_graph.relationships:
            source_entity = self._kg_store.find_or_create_entity(
                EntityCreate(
                    name=r.source_entity,
                    description=r.source_entity_description,
                ),
                commit=False,
            )
            target_entity = self._kg_store.find_or_create_entity(
                EntityCreate(
                    name=r.target_entity, description=r.target_entity_description
                ),
                commit=False,
            )
            self._kg_store.create_relationship(
                source_entity,
                target_entity,
                r.relationship_desc,
                metadata=node.metadata,
                commit=False,
            )

    def _try_merge_entities(self, entities: List[AIEntity]) -> AIEntity:
        logger.info(f"Trying to merge entities: {entities[0].name}")
        try:
            with dspy.settings.context(lm=self._dspy_lm):
                pred = self.merge_entities_prog(entities=entities)
                return pred.merged_entity
        except Exception as e:
            logger.error(f"Failed to merge entities: {e}", exc_info=True)
            return None

    def _build_index_from_nodes(self, nodes: Optional[Sequence[BaseNode]]) -> IndexLPG:
        """Build index from nodes."""
        nodes = self.insert_nodes(nodes or [])
        return IndexLPG()

    def as_retriever(self, **kwargs: Any):
        """Return a retriever for the index."""
        raise NotImplementedError(
            "Retriever not implemented for KnowledgeGraphIndex, use `retrieve_with_weight` instead."
        )

    def _insert(self, nodes: Sequence[BaseNode], **insert_kwargs: Any) -> None:
        """Index-specific logic for inserting nodes to the index struct."""
        self.insert_nodes(nodes)

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
