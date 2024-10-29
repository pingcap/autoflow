import time
import dspy
import logging
from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from sqlmodel import Session, select, func

from llama_index.core.embeddings.utils import EmbedType, resolve_embed_model
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType

from app.models import SemanticCache

logger = logging.getLogger(__name__)


class SemanticItem(BaseModel):
    """A single question-answer pair for semantic search."""

    question: str = Field(description="The question in the question-answer pair.")
    answer: str = Field(description="The answer corresponding to the question.")


class SemanticCandidate(BaseModel):
    """A single question pair for semantic search."""

    question: str = Field(description="The question in the question-answer pair.")


class SemanticGroup(BaseModel):
    """A collection of question-answer pairs for semantic search."""

    items: List[SemanticCandidate] = Field(description="A list of questions.")


class QASemanticOutput(BaseModel):
    """The output of the semantic search operation."""

    match_type: Literal["exact_match", "no_match", "similar_match"] = Field(
        description=(
            "The type of match found during the search. Use 'exact_match' if the query semantically matches the same "
            "question, meaning it is asking about the exact same topic. "
            "For all other cases, classify the match as 'no_match' or 'similar_match'."
        )
    )
    items: List[SemanticCandidate] = Field(
        description=(
            "The question-answer pair that matches the query. "
            "If the match_type is 'no_match', return an empty list. "
            "If the match_type is 'similar_match', return the most relevant questions."
            "If the match_type is 'exact_match', return the question that is semantically identical to the query."
        )
    )


class QASemanticSearchModule(dspy.Signature):
    """
    This module performs a semantic search to identify the best matching question-answer pairs from a given set of candidates.

    The semantic search process includes:
    - Comparing the query against a set of candidate question-answer pairs.
    - Returning an 'exact_match' if the query semantically matches a candidate question, meaning it is asking the exact same question.
    - Returning a 'similar_match' if the query is related to a candidate question but does not semantically match exactly.
    - Returning a 'no_match' if the query does not match any candidate question.

    Note: The output items must be selected from the provided candidates.
    """

    query: str = dspy.InputField(
        description="The query string to search for within the candidates."
    )
    candidats: SemanticGroup = dspy.InputField(
        description="A collection of frequently asked questions to search through."
    )

    output: QASemanticOutput = dspy.OutputField(
        description="The question that best matches the query string. "
    )


class SemanticSearchProgram(dspy.Module):
    def __init__(self, dspy_lm: dspy.LM):
        super().__init__()
        self.dspy_lm = dspy_lm
        self.prog = dspy.TypedChainOfThought(QASemanticSearchModule)

    def forward(self, query: str, candidats: SemanticGroup):
        with dspy.settings.context(lm=self.dspy_lm):
            return self.prog(query=query, candidats=candidats)


class SemanticCacheManager:
    def __init__(
        self,
        dspy_llm: dspy.LM,
        embed_model: Optional[EmbedType] = None,
        complied_sc_search_program_path: Optional[str] = None,
    ):
        self._dspy_lm = dspy_llm
        if embed_model:
            self._embed_model = resolve_embed_model(embed_model)
        else:
            self._embed_model = OpenAIEmbedding(
                model=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL
            )
        self.prog = SemanticSearchProgram(dspy_lm=dspy_llm)
        if complied_sc_search_program_path is not None:
            self.prog.load(complied_sc_search_program_path)

    def get_query_embedding(self, query: str):
        return self._embed_model.get_query_embedding(query)

    def add_cache(
        self,
        session: Session,
        item: SemanticItem,
        namespace: str,
        metadata: Optional[dict] = None,
    ):
        if metadata is None:
            metadata = {}
        metadata["namespace"] = namespace

        object = SemanticCache(
            query=item.question,
            query_vec=self.get_query_embedding(item.question),
            value=item.answer,
            value_vec=self.get_query_embedding(item.answer),
            meta=metadata,
        )
        session.add(object)
        session.commit()

    def search(
        self, session: Session, query: str, namespace: Optional[str] = None
    ) -> QASemanticOutput:
        start_time = time.time()
        embedding = self.get_query_embedding(query)
        logger.debug(
            f"[search_semantic_cache] Get query embedding {time.time() - start_time:.2f} seconds"
        )
        start_time = time.time()

        sql = (
            select(
                SemanticCache,
                SemanticCache.query_vec.cosine_distance(embedding).label("distance"),
            )
            .having(SemanticCache.query_vec.cosine_distance(embedding) < 0.8)
            .order_by("distance")
            .limit(20)
        )
        if namespace:
            sql = sql.where(
                func.json_extract(SemanticCache.meta, "$.namespace") == namespace
            )

        results = session.execute(sql).all()
        candidates = SemanticGroup(
            items=[
                SemanticCandidate(
                    question=result.SemanticCache.query,
                )
                for result in results
            ]
        )
        logger.debug(
            f"[search_semantic_cache] Search semantic cache {time.time() - start_time:.2f} seconds"
        )
        start_time = time.time()

        if len(candidates.items) == 0:
            return {
                "match_type": "no_match",
                "items": [],
            }

        pred = self.prog(query=query, candidats=candidates)
        logger.debug(
            f"[search_semantic_cache] Predict semantic cache {time.time() - start_time:.2f} seconds"
        )
        logger.debug(f"[search_semantic_cache] Predict semantic cache {pred.output}")

        # filter the matched items and it's metadata
        matched_items = []
        for item in pred.output.items:
            question = item.question
            # find the matched item in the results
            for result in results:
                if result.SemanticCache.query == question:
                    matched_items.append(
                        {
                            "question": result.SemanticCache.query,
                            "answer": result.SemanticCache.value,
                            "meta": result.SemanticCache.meta,
                        }
                    )
                    break

        return {"match_type": pred.output.match_type, "items": matched_items}
