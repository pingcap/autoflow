from pathlib import Path
from autoflow.knowledge_graph.extractors.simple import SimpleKGExtractor
from autoflow.models.llms.dspy import get_dspy_lm_by_llm


def test_kg_extractor(llm):
    text = Path("./tests/fixtures/tidb-overview.md").read_text()
    dspy_lm = get_dspy_lm_by_llm(llm)
    extractor = SimpleKGExtractor(dspy_lm)
    knowledge_graph = extractor.extract(text)
    assert knowledge_graph is not None
    assert len(knowledge_graph.entities) >= 2
    assert len(knowledge_graph.relationships) >= 1

    for entity in knowledge_graph.entities:
        assert entity.name is not None
        assert entity.description is not None
        assert len(entity.meta) > 0

    for relationship in knowledge_graph.relationships:
        assert relationship.source_entity_name is not None
        assert relationship.target_entity_name is not None
        assert relationship.description is not None
