from llama_index.core.workflow import Event


class SearchKnowledgeGraphEvent(Event):
    """Search knowledge graph event"""


class AggregateKGSearchResultEvent(Event):
    """Aggregate knowledge graph search result event"""


class RefineQuestionEvent(Event):
    """Refine question event"""


class RetrieveEvent(Event):
    """Retrieve event"""


class ClarifyQuestionEvent(Event):
    """Clarify question event"""


class GenerateAnswerEvent(Event):
    """Generate answer event"""


class GenerateAnswerStreamEvent(Event):
    """Generate answer stream event"""

    def __init__(self, chunk: str):
        super().__init__()
        self.chunk = chunk


class EarlyStopEvent(Event):
    """Early stop event"""

    def __init__(self, answer: str, **kwargs):
        super().__init__(**kwargs)
        self.answer = answer
