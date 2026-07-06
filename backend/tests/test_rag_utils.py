from app.rag.utils import (
    iter_without_hidden_reasoning,
    parse_question_lines,
    strip_hidden_reasoning,
)


def test_strip_hidden_reasoning_removes_think_block():
    output = strip_hidden_reasoning(
        "<think>reasoning that should not be shown</think>\n\nFinal answer"
    )

    assert output == "Final answer"


def test_strip_hidden_reasoning_removes_unclosed_think_block():
    output = strip_hidden_reasoning("Visible answer\n<think>unfinished reasoning")

    assert output == "Visible answer"


def test_iter_without_hidden_reasoning_handles_split_tags():
    chunks = ["Hello ", "<thi", "nk>hidden", " reasoning</th", "ink>", " world"]

    assert "".join(iter_without_hidden_reasoning(chunks)) == "Hello  world"


def test_parse_question_lines_filters_reasoning_and_metadata():
    questions = parse_question_lines(
        """
        <think>
        Need to generate follow-up questions.
        </think>
        1. 如何定位 TiDB 写入热点？
        - TiDB Dashboard 中哪些面板可以辅助排查热点？
        How can I assist you today?
        This is not a question
        """
    )

    assert questions == [
        "如何定位 TiDB 写入热点？",
        "TiDB Dashboard 中哪些面板可以辅助排查热点？",
    ]
