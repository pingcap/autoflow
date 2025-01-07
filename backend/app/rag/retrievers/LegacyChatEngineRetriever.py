import logging
from llama_index.core.retrievers import BaseRetriever
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore, QueryBundle
from pydantic import List
from sqlmodel import Session

from backend.app.models.chunk import get_kb_chunk_model
from backend.app.models.entity import get_kb_entity_model
from backend.app.models.relationship import get_kb_relationship_model
from backend.app.rag.chat import get_prompt_by_jinja2_template
from backend.app.rag.chat_config import ChatEngineConfig
from backend.app.rag.knowledge_base.config import get_kb_embed_model
from backend.app.rag.knowledge_graph.base import KnowledgeGraphIndex
from backend.app.rag.vector_store.tidb_vector_store import TiDBVectorStore
from backend.app.rag.knowledge_graph.graph_store.tidb_graph_store import TiDBGraphStore
from backend.app.repositories.knowledge_base import knowledge_base_repo


logger = logging.getLogger(__name__)


class LegacyChatEngineRetriever(BaseRetriever):
    """
    Legacy chat engine retriever, which is dependent on the configuration of the chat engine.
    """

    def __init__(
        self,
        db_session: Session,
        engine_name: str = "default",
        chat_engine_config: ChatEngineConfig = None,
        top_k: int = 10,
    ):
        self.db_session = db_session
        self.engine_name = engine_name
        self.top_k = top_k

        self.chat_engine_config = chat_engine_config or ChatEngineConfig.load_from_db(
            db_session, engine_name
        )
        self.db_chat_engine = self.chat_engine_config.get_db_chat_engine()
        self._llm = self.chat_engine_config.get_llama_llm(self.db_session)
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
                sub_queries = graph_index.intent_analyze(query_bundle.query_str)
                result = graph_index.graph_semantic_search(
                    sub_queries, include_meta=True
                )
                graph_knowledges = get_prompt_by_jinja2_template(
                    self.chat_engine_config.llm.intent_graph_knowledge,
                    sub_queries=result["queries"],
                )
                graph_knowledges_context = graph_knowledges.template
            else:
                entities, relations = graph_index.retrieve_with_weight(
                    query_bundle.query_str,
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

        # 2. Refine the user question using graph information and chat history
        refined_question = self._fast_llm.predict(
            get_prompt_by_jinja2_template(
                self.chat_engine_config.llm.condense_question_prompt,
                graph_knowledges=graph_knowledges_context,
                question=query_bundle.query_str,
            ),
        )

        # 3. Retrieve the related chunks from the vector store
        # 4. Rerank after the retrieval
        vector_store = TiDBVectorStore(
            session=self.db_session, chunk_db_model=self._chunk_model
        )
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self._embed_model,
        )
        retrieve_engine = vector_index.as_retriever(
            node_postprocessors=[self._reranker],
            similarity_top_k=self.top_k,
        )

        return retrieve_engine.retrieve(refined_question)
