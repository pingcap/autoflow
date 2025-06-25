import json
import os
import logging
from logging import getLogger

import pytest
import dspy

from litellm import verbose_logger
from llama_index.core import PromptTemplate
from llama_index.core.base.llms.base import BaseLLM

from app.rag.indices.knowledge_graph.extractor import Extractor
from app.rag.llms.provider import LLMProvider
from app.rag.llms.resolver import resolve_llm
from app.rag.question_gen.query_decomposer import QueryDecomposer
from app.rag.llms.dspy import get_dspy_lm_by_llama_llm


question = "Is TiDB open source? (Yes/No)"
content = """
TiDB is a distributed database that you can use the MySQL client to connect to.
"""

os.environ["LITELLM_LOG"] = "DEBUG"
verbose_logger.setLevel(logging.WARN)

logger = getLogger(__name__)


def check_llm_answer_simple_question(llm: BaseLLM):
    prompt = PromptTemplate(question)
    output = llm.predict(prompt)

    assert "yes" in output.lower()

    logger.info(f"Generated answer: \n{output}")


def check_dspy_lm_decompose_question(lm: dspy.LM):
    decomposer = QueryDecomposer(lm)
    subquestions = decomposer.decompose("What is TiDB").questions

    assert len(subquestions) >= 1

    questions = [q.question for q in subquestions]
    logger.info(f"Generated sub-question: \n{questions}")


def check_dspy_lm_extract_graph(lm: dspy.LM):
    extractor = Extractor(lm)
    kg = extractor.forward(content).knowledge

    assert len(kg.entities) >= 2
    assert len(kg.relationships) >= 1

    entities = [f"{e.name}: {e.description}" for e in kg.entities]
    relationships = [
        f"{r.source_entity} -> {r.relationship_desc} -> {r.target_entity}"
        for r in kg.relationships
    ]
    logger.info(f"Extracted entities: \n{entities}")
    logger.info(f"Extracted relationships: \n{relationships}")


def test_openai():
    llm = resolve_llm(
        provider=LLMProvider.OPENAI,
        model="gpt-4.1-2025-04-14-mini",
        config={},
        credentials=os.getenv("OPENAI_API_KEY"),
    )
    check_llm_answer_simple_question(llm)

    lm = get_dspy_lm_by_llama_llm(llm)
    check_dspy_lm_decompose_question(lm)
    check_dspy_lm_extract_graph(lm)


@pytest.mark.skipif(
    os.getenv("GITHUB_ACTIONS"), reason="ollama is not available on GitHub Actions"
)
def test_ollama():
    llm = resolve_llm(
        provider=LLMProvider.OLLAMA,
        model="gemma3:4b",
    )
    check_llm_answer_simple_question(llm)

    lm = get_dspy_lm_by_llama_llm(llm)
    check_dspy_lm_decompose_question(lm)
    check_dspy_lm_extract_graph(lm)


def test_gitee_ai():
    llm = resolve_llm(
        provider=LLMProvider.GITEEAI,
        model="Qwen2.5-72B-Instruct",
        credentials=os.getenv("GITEEAI_API_KEY"),
    )
    check_llm_answer_simple_question(llm)

    lm = get_dspy_lm_by_llama_llm(llm)
    check_dspy_lm_decompose_question(lm)
    check_dspy_lm_extract_graph(lm)


def test_bedrock():
    llm = resolve_llm(
        provider=LLMProvider.BEDROCK,
        model="meta.llama3-1-70b-instruct-v1:0",
        credentials={
            "aws_access_key_id": os.getenv("AWS_ACCESS_KEY_ID"),
            "aws_secret_access_key": os.getenv("AWS_SECRET_ACCESS_KEY"),
            "aws_region_name": os.getenv("AWS_REGION_NAME"),
        },
    )
    check_llm_answer_simple_question(llm)

    lm = get_dspy_lm_by_llama_llm(llm)
    check_dspy_lm_decompose_question(lm)
    check_dspy_lm_extract_graph(lm)


def test_vertex_ai():
    llm = resolve_llm(
        provider=LLMProvider.VERTEX_AI,
        model="gemini-2.0-flash-001",
        credentials=json.loads(os.getenv("GOOGLE_SERVICE_ACCOUNT_CREDENTIALS")),
        config={"location": "us-west1"},
    )
    check_llm_answer_simple_question(llm)

    lm = get_dspy_lm_by_llama_llm(llm)
    check_dspy_lm_decompose_question(lm)
    check_dspy_lm_extract_graph(lm)


def test_gemini():
    llm = resolve_llm(
        provider=LLMProvider.GEMINI,
        model="models/gemini-2.0-flash-001",
        credentials=os.getenv("GEMINI_API_KEY"),
    )
    check_llm_answer_simple_question(llm)

    lm = get_dspy_lm_by_llama_llm(llm)
    check_dspy_lm_decompose_question(lm)
    check_dspy_lm_extract_graph(lm)


def test_azure_ai():
    llm = resolve_llm(
        provider=LLMProvider.AZURE_OPENAI,
        model="gpt-4.1-2025-04-14-mini",
        credentials=os.getenv("AZURE_AI_API_KEY"),
        config={
            "azure_endpoint": os.getenv("AZURE_AI_ENDPOINT"),
            "engine": "gpt-4.1-2025-04-14",
            "api_version": "2025-01-01-preview",
        },
    )
    check_llm_answer_simple_question(llm)

    lm = get_dspy_lm_by_llama_llm(llm)
    check_dspy_lm_decompose_question(lm)
    check_dspy_lm_extract_graph(lm)
