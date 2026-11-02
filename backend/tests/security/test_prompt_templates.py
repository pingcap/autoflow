from unittest.mock import Mock

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from llama_index.core.llms import MockLLM
from llama_index.core.prompts.rich import RichPromptTemplate

from app.rag import default_prompt
from app.rag.chat.config import LLMOption
from app.rag.retrievers.knowledge_graph.schema import (
    RetrievedEntity,
    RetrievedRelationship,
)
from app.repositories.chat_engine import chat_engine_repo
from app.models import ChatEngine, ChatEngineUpdate
from app.security.prompt_template import SafePromptTemplate, PromptSafetyError


@pytest.mark.parametrize(
    "source",
    [
        "{{ cycler.__init__.__globals__.os.popen('id').read() }}",
        "{{ item|attr('__class__') }}",
        "{{ '/etc/passwd'|document }}",
        "{{ 'http://169.254.169.254'|image }}",
        "{% completion %}run{% endcompletion %}",
        "{% include '/etc/passwd' %}",
        "{% macro recurse() %}{{ recurse() }}{% endmacro %}{{ recurse() }}",
        "{{ 'x' * 1000000000 }}",
        "{{ 10 ** 1000000000 }}",
        "{{ items|join(separator) }}",
        "{{ item['__class__'] }}",
    ],
)
def test_unsafe_templates_rejected_before_compilation_or_rendering(source):
    with pytest.raises(PromptSafetyError):
        SafePromptTemplate(source)
    with pytest.raises(ValidationError):
        LLMOption(text_qa_prompt=source)


def test_all_shipped_templates_preserve_their_rendered_content():
    entity = RetrievedEntity(id=1, name="TiDB", description="SQL database")
    relationship = RetrievedRelationship(
        id=1,
        source_entity_id=1,
        target_entity_id=2,
        description="related",
        meta={"version": 2},
        rag_description="TiDB uses TiKV",
        weight=0.9,
    )
    context = dict(
        entities=[entity],
        relationships=[relationship],
        sub_queries={
            "storage": {"entities": [entity], "relationships": [relationship]}
        },
        graph_knowledges="graph facts",
        chat_history="old conversation",
        question="What is TiDB?",
        current_date="2026-11-03",
        context_str="document facts",
        query_str="What is TiDB?",
        original_question="What is TiDB?",
        answer="A distributed SQL database",
    )
    checked = 0
    for name in dir(default_prompt):
        if name.startswith("DEFAULT_") and isinstance(
            source := getattr(default_prompt, name), str
        ):
            prompt = SafePromptTemplate(source)
            values = {key: context.get(key, "fixture") for key in prompt.template_vars}
            assert prompt.format(**values) == RichPromptTemplate(source).format(
                **values
            ), name
            checked += 1
    assert checked >= 7


def test_prompt_limits_bound_empty_loops_and_streamed_output():
    empty_loop = SafePromptTemplate(
        '{% for a in items %}{% for b in items %}{{ "" }}{% endfor %}{% endfor %}'
    )
    with pytest.raises(PromptSafetyError, match="iteration"):
        empty_loop.format(items=list(range(101)))
    output = SafePromptTemplate("{% for item in items %}{{ text }}{% endfor %}")
    with pytest.raises(PromptSafetyError, match="output"):
        output.format(items=[1, 2, 3, 4], text="a" * 300000)
    with pytest.raises(PromptSafetyError, match="context"):
        SafePromptTemplate("{{ text }}").format(text="a" * 600000)


def test_rendering_never_calls_arbitrary_python_objects():
    stringify = Mock(return_value="secret")

    class Object:
        def __str__(self):
            return stringify()

    with pytest.raises(PromptSafetyError, match="data values"):
        SafePromptTemplate("{{ item }}").format(item=Object())
    stringify.assert_not_called()


def test_partial_format_and_llama_predict_remain_compatible():
    prompt = SafePromptTemplate(
        "Context: {{ context }}; question: {{ question }}"
    ).partial_format(context="facts")
    assert prompt.format(question="why") == "Context: facts; question: why"
    assert (
        prompt.format_messages(question="why")[0].content
        == "Context: facts; question: why"
    )
    assert "facts" in MockLLM().predict(prompt, question="why")


def test_engine_writes_reject_unsafe_prompts_before_any_mutation():
    session = Mock()
    options = {"llm": {"text_qa_prompt": "{{ '/etc/passwd'|document }}"}}
    with pytest.raises(HTTPException) as error:
        chat_engine_repo.create(
            session, ChatEngine(name="fixture", engine_options=options)
        )
    assert error.value.status_code == 400
    session.add.assert_not_called()
    session.commit.assert_not_called()
    existing = ChatEngine(name="fixture", engine_options={})
    with pytest.raises(HTTPException):
        chat_engine_repo.update(
            session, existing, ChatEngineUpdate(engine_options=options)
        )
    assert existing.engine_options == {}


def test_retrieved_text_is_data_and_does_not_become_another_template():
    payload = "{{ cycler.__init__.__globals__.os.popen('id').read() }}"
    assert SafePromptTemplate("{{ text }}").format(text=payload) == payload


def test_real_llama_response_synthesizer_accepts_safe_partial_prompt(monkeypatch):
    from llama_index.core import Settings
    from llama_index.core.response_synthesizers import get_response_synthesizer
    from llama_index.core.schema import NodeWithScore, TextNode

    monkeypatch.setattr(Settings, "_tokenizer", lambda text: text.split())
    prompt = SafePromptTemplate(default_prompt.DEFAULT_TEXT_QA_PROMPT).partial_format(
        current_date="2026-11-03",
        graph_knowledges="graph facts",
        original_question="What is TiDB?",
    )
    synthesizer = get_response_synthesizer(llm=MockLLM(), text_qa_template=prompt)
    response = synthesizer.synthesize(
        query="What is TiDB?",
        nodes=[
            NodeWithScore(
                node=TextNode(text="TiDB is a distributed SQL database."), score=0.9
            )
        ],
    )
    assert "distributed SQL database" in str(response)
