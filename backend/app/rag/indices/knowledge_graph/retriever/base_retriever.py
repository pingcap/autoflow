from typing import Optional, List, Tuple

from sqlmodel import Session
from llama_index.core import QueryBundle
from llama_index.core.callbacks import CallbackManager, EventPayload
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore
from app.core.db import engine
from app.models.entity import get_kb_entity_model
from app.models.relationship import get_kb_relationship_model
from app.rag.indices.knowledge_graph.schema import (
    KnowledgeGraphNode,
    RetrievedEntity,
    RetrievedRelationship,
)
from app.rag.knowledge_base.config import get_kb_embed_model, get_kb_dspy_llm
from app.rag.graph_store import TiDBGraphStore
from app.rag.indices.knowledge_graph.retriever.config import (
    KnowledgeGraphRetrieverConfig,
)
from app.rag.types import MyCBEventType
from app.repositories import knowledge_base_repo


class KnowledgeGraphRetriever(BaseRetriever):
    def __init__(
        self,
        knowledge_base_id: int,
        config: KnowledgeGraphRetrieverConfig,
        callback_manager: Optional[CallbackManager] = CallbackManager([]),
        **kwargs,
    ):
        super().__init__(callback_manager, **kwargs)
        self.config = config
        self._callback_manager = callback_manager

        with Session(engine) as db_session:
            self.kb = knowledge_base_repo.must_get(db_session, knowledge_base_id)
            self.embed_model = get_kb_embed_model(db_session, self.kb)
            self.entity_db_model = get_kb_entity_model(self.kb)
            self.relationship_db_model = get_kb_relationship_model(self.kb)
            # TODO: remove it
            dspy_lm = get_kb_dspy_llm(db_session, self.kb)
            self._kg_store = TiDBGraphStore(
                dspy_lm=dspy_lm,
                session=db_session,
                embed_model=self.embed_model,
                entity_db_model=self.entity_db_model,
                relationship_db_model=self.relationship_db_model,
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
    ) -> Tuple[List[RetrievedEntity], List[RetrievedRelationship]]:
        with self._callback_manager.event(
            MyCBEventType.RETRIEVE_FROM_GRAPH,
            payload={
                EventPayload.QUERY_STR: query_bundle.query_str,
                EventPayload.ADDITIONAL_KWARGS: self.config,
            },
        ) as event:
            entities, relationships = self._kg_store.retrieve_with_weight(
                query_bundle.query_str,
                embedding=[],
                depth=self.config.depth,
                include_meta=self.config.include_meta,
                with_degree=self.config.with_degree,
                relationship_meta_filters=self.config.metadata_filter,
            )
            event.on_end(
                payload={"entities": entities, "relationships": relationships},
            )

        return entities, relationships
