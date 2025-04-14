import logging
from pathlib import Path
import pytest
from autoflow.knowledge_graph.programs.eval_graph import KnowledgeGraphEvaluator
from autoflow.knowledge_graph.programs.extract_graph import KnowledgeGraphExtractor
from autoflow.knowledge_graph.types import GeneratedKnowledgeGraph

from autoflow.models.llms.dspy import get_dspy_lm_by_llm

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def extractor(llm):
    dspy_lm = get_dspy_lm_by_llm(llm)
    extractor = KnowledgeGraphExtractor(dspy_lm=dspy_lm)
    return extractor


@pytest.fixture(scope="module")
def evaluator(llm):
    dspy_lm = get_dspy_lm_by_llm(llm)
    evaluator = KnowledgeGraphEvaluator(dspy_lm=dspy_lm)
    return evaluator


def test_extract_graph(extractor, evaluator):
    text = Path("tests/fixtures/tidb-overview.md").read_text()

    # Expected knowledge graph output
    expected_knowledge = GeneratedKnowledgeGraph.model_validate(
        {
            "entities": [
                {
                    "name": "TiDB",
                    "description": "An open-source distributed SQL database that supports HTAP workloads.",
                },
                {
                    "name": "TiDB Self-Managed",
                    "description": "A product option of TiDB where users deploy and manage TiDB on their own infrastructure.",
                },
                {
                    "name": "TiDB Cloud",
                    "description": "The fully-managed TiDB service for deploying and running TiDB clusters in the cloud.",
                },
                {
                    "name": "TiDB Operator",
                    "description": "A tool that helps manage TiDB on Kubernetes and automates tasks related to operating TiDB clusters",
                },
                {
                    "name": "TiKV",
                    "description": "A row-based storage engine used by TiDB.",
                },
                {
                    "name": "TiFlash",
                    "description": "A columnar storage engine used by TiDB.",
                },
                {
                    "name": "Multi-Raft Learner protocol",
                    "description": "A protocol used by TiDB to replicate data from TiKV to TiFlash.",
                },
            ],
            "relationships": [
                {
                    "source_entity_name": "TiDB",
                    "target_entity_name": "TiKV",
                    "description": "TiDB uses TiKV as its storage engine",
                },
                {
                    "source_entity_name": "TiDB",
                    "target_entity_name": "TiFlash",
                    "description": "TiDB uses TiFlash as its analytics engine",
                },
                {
                    "source_entity_name": "TiDB",
                    "target_entity_name": "Multi-Raft Learner protocol",
                    "description": "TiDB uses the Multi-Raft Learner protocol to replicate data from TiKV to TiFlash.",
                },
                {
                    "source_entity_name": "TiDB",
                    "target_entity_name": "HTAP",
                    "description": "TiDB supports HTAP workloads",
                },
                {
                    "source_entity_name": "TiDB Self-Managed",
                    "target_entity_name": "TiDB",
                    "description": "TiDB Self-Managed is a product option of TiDB",
                },
                {
                    "source_entity_name": "TiDB Cloud",
                    "target_entity_name": "TiDB",
                    "description": "TiDB Cloud is a fully-managed TiDB service",
                },
                {
                    "source_entity_name": "TiDB Operator",
                    "target_entity_name": "TiDB Cloud",
                    "description": "TiDB Operator is a tool that helps manage TiDB on Kubernetes and automates tasks related to operating TiDB clusters",
                },
            ],
        }
    )

    # Generate knowledge graph
    actual_knowledge = extractor.forward(text)

    # Use LLM to evaluate the completeness
    evaluation_result = evaluator.forward(expected_knowledge, actual_knowledge)
    final_score = evaluation_result.score

    logger.info(f"Final score: {final_score}")
    assert final_score > 0.4, "The completeness score should be greater than 0.4."
