import logging
from typing import List, Optional, Type

from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore
from sqlmodel import Session, select

from app.models import (
    Document as DBDocument,
    Chunk as DBChunk,
    Entity as DBEntity,
    Relationship as DBRelationship,
)
from app.models.chunk import get_kb_chunk_model
from app.models.entity import get_kb_entity_model
from app.rag.chat import get_prompt_by_jinja2_template
from app.rag.chat_config import ChatEngineConfig, get_default_embed_model
from app.models.relationship import get_kb_relationship_model
from app.models.patch.sql_model import SQLModel
from app.rag.knowledge_base.config import get_kb_embed_model
from app.rag.knowledge_graph import KnowledgeGraphIndex
from app.rag.knowledge_graph.graph_store import TiDBGraphStore
from app.rag.vector_store.tidb_vector_store import TiDBVectorStore
from app.repositories.knowledge_base import knowledge_base_repo


logger = logging.getLogger(__name__)


class RetrieveService:
    _chunk_model: Type[SQLModel] = DBChunk
    _entity_model: Type[SQLModel] = DBEntity
    _relationship_model: Type[SQLModel] = DBRelationship

    def __init__(
        self,
        db_session: Session,
        engine_name: str = "default",
    ) -> None:
        self.db_session = db_session
        self.engine_name = engine_name

        self.chat_engine_config = ChatEngineConfig.load_from_db(db_session, engine_name)
        self.db_chat_engine = self.chat_engine_config.get_db_chat_engine()
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
        else:
            self._embed_model = get_default_embed_model(self.db_session)

    def retrieve(self, question: str, top_k: int = 10) -> List[DBDocument]:
        """
        Retrieve the related documents based on the user question.
        Args:
            question: The question from client.
            top_k: The number of top-k CHUNKS, and retrieve its related documents.
                   Which means it might return less than top_k documents.

        Returns:
            A list of related documents.
        """
        try:
            return self._retrieve(question, top_k)
        except Exception as e:
            logger.exception(e)

    def _retrieve(self, question: str, top_k: int) -> List[DBDocument]:
        # TODO: move to __init__?
        _llm = self.chat_engine_config.get_llama_llm(self.db_session)
        _fast_llm = self.chat_engine_config.get_fast_llama_llm(self.db_session)
        _fast_dspy_lm = self.chat_engine_config.get_fast_dspy_lm(self.db_session)

        # 1. Retrieve entities, relations, and chunks from the knowledge graph
        kg_config = self.chat_engine_config.knowledge_graph
        if kg_config.enabled:
            graph_store = TiDBGraphStore(
                dspy_lm=_fast_dspy_lm,
                session=self.db_session,
                embed_model=self._embed_model,
                entity_db_model=self._entity_model,
                relationship_db_model=self._relationship_model,
            )
            graph_index: KnowledgeGraphIndex = KnowledgeGraphIndex.from_existing(
                dspy_lm=_fast_dspy_lm,
                kg_store=graph_store,
            )

            if kg_config.using_intent_search:
                sub_queries = graph_index.intent_analyze(question)
                result = graph_index.graph_semantic_search(
                    sub_queries, include_meta=True
                )

                graph_knowledges = get_prompt_by_jinja2_template(
                    self.chat_engine_config.llm.intent_graph_knowledge,
                    sub_queries=result["queries"],
                )
                graph_knowledges_context = graph_knowledges.template
            else:
                entities, relations, chunks = graph_index.retrieve_with_weight(
                    question,
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
            entities, relations, chunks = [], [], []
            graph_knowledges_context = ""

        # 2. Refine the user question using graph information and chat history
        refined_question = _fast_llm.predict(
            get_prompt_by_jinja2_template(
                self.chat_engine_config.llm.condense_question_prompt,
                graph_knowledges=graph_knowledges_context,
                question=question,
            ),
        )

        # 3. Retrieve the related chunks from the vector store
        # 4. Rerank after the retrieval
        # 5. Generate a response using the refined question and related chunks
        text_qa_template = get_prompt_by_jinja2_template(
            self.chat_engine_config.llm.text_qa_prompt,
            graph_knowledges=graph_knowledges_context,
        )
        refine_template = get_prompt_by_jinja2_template(
            self.chat_engine_config.llm.refine_prompt,
            graph_knowledges=graph_knowledges_context,
        )
        vector_store = TiDBVectorStore(
            session=self.db_session, chunk_db_model=self._chunk_model
        )
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store,
            embed_model=self._embed_model,
        )

        retrieve_engine = vector_index.as_retriever(
            node_postprocessors=[self._reranker],
            streaming=True,
            text_qa_template=text_qa_template,
            refine_template=refine_template,
            similarity_top_k=top_k,
        )

        node_list: List[NodeWithScore] = retrieve_engine.retrieve(refined_question)
        source_documents = self._get_source_documents(node_list)

        return source_documents

    def _embedding_retrieve(self, question: str, top_k: int) -> List[NodeWithScore]:
        vector_store = TiDBVectorStore(
            session=self.db_session, chunk_db_model=self._chunk_model
        )
        vector_index = VectorStoreIndex.from_vector_store(
            vector_store, embed_model=self._embed_model
        )

        retrieve_engine = vector_index.as_retriever(
            node_postprocessors=[self._reranker],
            similarity_top_k=top_k,
        )

        node_list: List[NodeWithScore] = retrieve_engine.retrieve(question)
        return node_list

    def _get_source_documents(self, node_list: List[NodeWithScore]) -> List[DBDocument]:
        source_nodes_ids = [s_n.node_id for s_n in node_list]
        stmt = select(DBDocument).where(
            DBDocument.id.in_(
                select(
                    self._chunk_model.document_id,
                ).where(
                    self._chunk_model.id.in_(source_nodes_ids),
                )
            ),
        )

        return list(self.db_session.exec(stmt).all())
