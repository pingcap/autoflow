import logging
from typing import List
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
from sqlmodel import Session

from app.models.chunk import get_kb_chunk_model
from app.models.entity import get_kb_entity_model
from app.models.relationship import get_kb_relationship_model
from app.rag.chat import get_prompt_by_jinja2_template
from app.rag.chat_config import ChatEngineConfig
from app.rag.knowledge_base.config import get_kb_embed_model
from app.rag.indexes.knowledge_graph import KnowledgeGraphIndex
from app.rag.indexes.vector_search.vector_store.tidb_vector_store import TiDBVectorStore
from app.rag.indexes.knowledge_graph.graph_store import TiDBGraphStore
from app.repositories.knowledge_base import knowledge_base_repo


logger = logging.getLogger(__name__)


class ChatEngineBasedRetriever(BaseRetriever):
    """
    Chat engine based retriever, which is dependent on the configuration of the chat engine.
    """

    def __init__(
        self,
        db_session: Session,
        engine_name: str = "default",
        chat_engine_config: ChatEngineConfig = None,
        top_k: int = 10,
        similarity_top_k: int = None,
        oversampling_factor: int = 5,
        enable_kg_enhance_query_refine: bool = False,
    ):
        self.db_session = db_session
        self.engine_name = engine_name
        self.top_k = top_k
        self.similarity_top_k = similarity_top_k
        self.oversampling_factor = oversampling_factor
        self.enable_kg_enhance_query_refine = enable_kg_enhance_query_refine

        self.chat_engine_config = chat_engine_config or ChatEngineConfig.load_from_db(
            db_session, engine_name
        )
        self.db_chat_engine = self.chat_engine_config.get_db_chat_engine()
        self._fast_llm = self.chat_engine_config.get_fast_llama_llm(self.db_session)
        self._fast_dspy_lm = self.chat_engine_config.get_fast_dspy_lm(self.db_session)
        self._reranker = self.chat_engine_config.get_reranker(db_session)

        if self.chat_engine_config.knowledge_base:
            # TODO: Support multiple knowledge base retrieve.
            linked_knowledge_base = (
                self.chat_engine_config.knowledge_base.linked_knowledge_base
            )
            kb = knowledge_base_repo.must_get(db_session, linked_knowledge_base.id)
            self._chunk_model = get_kb_chunk_model(kb)
            self._entity_model = get_kb_entity_model(kb)
            self._relationship_model = get_kb_relationship_model(kb)
            self._embed_model = get_kb_embed_model(self.db_session, kb)

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if not self.chat_engine_config.knowledge_base:
            logger.warn(
                "The chat engine does not configured the retrieve knowledge base, return empty list"
            )
            return []

        # 1. Retrieve entities, relations, and chunks from the knowledge graph
        # 2. Refine the user question using graph information
        refined_query = self._refine_query(query_bundle.query_str)

        # 3. Retrieve the related chunks from the vector store
        # 4. Rerank after the retrieval
        vector_store = TiDBVectorStore(
            session=self.db_session,
            chunk_db_model=self._chunk_model,
            oversampling_factor=self.oversampling_factor,
        )
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self._embed_model,
        )

        # Node postprocessors
        metadata_filter = self.chat_engine_config.get_metadata_filter()
        reranker = self.chat_engine_config.get_reranker(
            self.db_session, top_n=self.top_k
        )
        if reranker:
            node_postprocessors = [metadata_filter, reranker]
        else:
            node_postprocessors = [metadata_filter]

        # Retriever Engine
        retrieve_engine = vector_index.as_retriever(
            node_postprocessors=node_postprocessors,
            similarity_top_k=self.similarity_top_k or self.top_k,
        )

        return retrieve_engine.retrieve(refined_query)

    def _refine_query(self, query: str) -> str:
        kg_config = self.chat_engine_config.knowledge_graph
        if kg_config.enabled:
            graph_store = TiDBGraphStore(
                dspy_lm=self._fast_dspy_lm,
                session=self.db_session,
                embed_model=self._embed_model,
                entity_db_model=self._entity_model,
                relationship_db_model=self._relationship_model,
            )
            graph_index: KnowledgeGraphIndex = KnowledgeGraphIndex.from_existing(
                dspy_lm=self._fast_dspy_lm,
                kg_store=graph_store,
            )

            if kg_config.using_intent_search:
                sub_queries = graph_index.intent_analyze(query)
                result = graph_index.graph_semantic_search(
                    sub_queries, include_meta=True
                )
                graph_knowledges = get_prompt_by_jinja2_template(
                    self.chat_engine_config.llm.intent_graph_knowledge,
                    sub_queries=result["queries"],
                )
                graph_knowledges_context = graph_knowledges.template
            else:
                entities, relations, _ = graph_index.retrieve_with_weight(
                    query,
                    [],
                    depth=kg_config.depth,
                    include_meta=kg_config.include_meta,
                    with_degree=kg_config.with_degree,
                    with_chunks=False,
                )
                graph_knowledges = get_prompt_by_jinja2_template(
                    self.chat_engine_config.llm.normal_graph_knowledge,
                    entities=entities,
                    relationships=relations,
                )
                graph_knowledges_context = graph_knowledges.template
        else:
            entities, relations = [], []
            graph_knowledges_context = ""

        refined_query = self._fast_llm.predict(
            get_prompt_by_jinja2_template(
                self.chat_engine_config.llm.condense_question_prompt,
                graph_knowledges=graph_knowledges_context,
                question=query,
            ),
        )

        return refined_query
