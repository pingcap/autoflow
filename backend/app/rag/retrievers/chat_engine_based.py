import logging
from datetime import datetime
from typing import List, Tuple
from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle
from sqlmodel import Session

from app.rag.chat import get_prompt_by_jinja2_template
from app.rag.chat_config import (
    ChatEngineConfig,
    KnowledgeGraphOption,
    KnowledgeBaseOption,
)
from app.rag.retrievers.knowledge_graph.fusion_retriever import (
    KnowledgeGraphFusionRetriever,
)
from app.rag.retrievers.knowledge_graph.schema import (
    KnowledgeGraphRetrievalResult,
    KnowledgeGraphRetrieverConfig,
)
from app.rag.retrievers.chunk.fusion_retriever import ChunkFusionRetriever
from app.rag.retrievers.chunk.schema import VectorSearchRetrieverConfig
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

        # Init LLM.
        self._llm = self.chat_engine_config.get_llama_llm(self.db_session)
        self._fast_llm = self.chat_engine_config.get_fast_llama_llm(self.db_session)
        self._fast_dspy_lm = self.chat_engine_config.get_fast_dspy_lm(self.db_session)

        # Load knowledge bases.
        kb_config: KnowledgeBaseOption = self.chat_engine_config.knowledge_base
        linked_knowledge_base_ids = []
        if len(kb_config.linked_knowledge_bases) == 0:
            linked_knowledge_base_ids.append(
                self.chat_engine_config.knowledge_base.linked_knowledge_base.id
            )
        else:
            linked_knowledge_base_ids.extend(
                [kb.id for kb in kb_config.linked_knowledge_bases]
            )
        self.knowledge_base_ids = linked_knowledge_base_ids
        self.knowledge_bases = knowledge_base_repo.get_by_ids(
            self.db_session, knowledge_base_ids=linked_knowledge_base_ids
        )

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        if self.enable_kg_enhance_query_refine:
            refined_question = self._kg_enhance_query_refine(query_bundle.query_str)
        else:
            refined_question = query_bundle.query_str

        return self._search_relevant_chunks(refined_question=refined_question)

    def _kg_enhance_query_refine(self, query_str):
        # 1. Retrieve Knowledge graph related to the user question.
        kg_config = self.chat_engine_config.knowledge_graph
        knowledge_graph_context = ""
        if kg_config is not None and kg_config.enabled:
            _, knowledge_graph_context = self._search_knowledge_graph(
                user_question=query_str, kg_config=kg_config
            )

        # 2. Refine the user question using knowledge graph and chat history.
        refined_question = self._refine_user_question(
            user_question=query_str,
            knowledge_graph_context=knowledge_graph_context,
            refined_question_prompt=self.chat_engine_config.llm.condense_question_prompt,
        )

        return refined_question

    def _search_knowledge_graph(
        self, user_question: str, kg_config: KnowledgeGraphOption
    ) -> Tuple[KnowledgeGraphRetrievalResult, str]:
        # For forward compatibility of chat engine config.
        enable_metadata_filter = kg_config.enable_metadata_filter or (
            kg_config.relationship_meta_filters is not None
        )
        metadata_filters = (
            kg_config.metadata_filters or kg_config.relationship_meta_filters
        )

        kg_retriever = KnowledgeGraphFusionRetriever(
            db_session=self.db_session,
            knowledge_base_ids=self.knowledge_base_ids,
            llm=self._llm,
            use_query_decompose=kg_config.using_intent_search,
            use_async=True,
            config=KnowledgeGraphRetrieverConfig(
                depth=kg_config.depth,
                include_metadata=kg_config.include_meta,
                with_degree=kg_config.with_degree,
                enable_metadata_filter=enable_metadata_filter,
                metadata_filters=metadata_filters,
            ),
            callback_manager=self.callback_manager,
        )

        if kg_config.using_intent_search:
            knowledge_graph = kg_retriever.retrieve_knowledge_graph(user_question)
            kg_context_template = get_prompt_by_jinja2_template(
                self.chat_engine_config.llm.intent_graph_knowledge,
                # For forward compatibility considerations.
                sub_queries=knowledge_graph.to_subqueries_dict(),
            )
            knowledge_graph_context = kg_context_template.template
        else:
            knowledge_graph = kg_retriever.retrieve_knowledge_graph(user_question)
            kg_context_template = get_prompt_by_jinja2_template(
                self.chat_engine_config.llm.normal_graph_knowledge,
                entities=knowledge_graph.entities,
                relationships=knowledge_graph.relationships,
            )
            knowledge_graph_context = kg_context_template.template

        return (
            knowledge_graph,
            knowledge_graph_context,
        )

    def _refine_user_question(
        self,
        user_question: str,
        refined_question_prompt: str,
        knowledge_graph_context: str = "",
    ) -> str:
        return self._fast_llm.predict(
            get_prompt_by_jinja2_template(
                refined_question_prompt,
                graph_knowledges=knowledge_graph_context,
                question=user_question,
                current_date=datetime.now().strftime("%Y-%m-%d"),
            ),
        )

    def _search_relevant_chunks(self, refined_question: str) -> List[NodeWithScore]:
        retriever = ChunkFusionRetriever(
            db_session=self.db_session,
            knowledge_base_ids=self.knowledge_base_ids,
            llm=self._llm,
            config=VectorSearchRetrieverConfig(
                similarity_top_k=self.similarity_top_k,
                oversampling_factor=self.oversampling_factor,
                top_k=self.top_k,
            ),
            use_query_decompose=False,
            use_async=True,
            callback_manager=self.callback_manager,
        )

        return retriever.retrieve(QueryBundle(refined_question))
