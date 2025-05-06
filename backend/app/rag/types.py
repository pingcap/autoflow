import enum


# Langfuse needs an enum class for event types,
# but the CBEventType in llama-index does not have sufficient types.
class MyCBEventType(str, enum.Enum):
    CHUNKING = "chunking"
    NODE_PARSING = "node_parsing"
    EMBEDDING = "embedding"
    LLM = "llm"
    QUERY = "query"
    RETRIEVE = "retrieve"
    SYNTHESIZE = "synthesize"
    TREE = "tree"
    SUB_QUESTION = "sub_question"
    TEMPLATING = "templating"
    FUNCTION_CALL = "function_call"
    RERANKING = "reranking"
    EXCEPTION = "exception"
    AGENT_STEP = "agent_step"
    CLARIFYING_QUESTION = "clarifying_question"
    CONDENSE_QUESTION = "condense_question"
    REFINE_QUESTION = "refine_question"
    RETRIEVE_FROM_GRAPH = "retrieve_from_graph"
    INTENT_DECOMPOSITION = "intent_decomposition"
    GRAPH_SEMANTIC_SEARCH = "graph_semantic_search"
    SELECT_KNOWLEDGE_BASE = "select_knowledge_base"
    RUN_SUB_QUERIES = "run_sub_queries"


# Chat stream response event types
class ChatEventType(int, enum.Enum):
    # Following vercel ai sdk's event type
    # https://github.com/vercel/ai/blob/84871281ab5a2c080e3f8e18d02cd09c7e1691c4/packages/ui-utils/src/stream-parts.ts#L368
    TEXT_PART = 0
    DATA_PART = 2
    ERROR_PART = 3
    MESSAGE_ANNOTATIONS_PART = 8


class ChatMessageSate(int, enum.Enum):
    TRACE = 0
    SOURCE_NODES = 1
    KG_RETRIEVAL = 2
    REFINE_QUESTION = 3
    SEARCH_RELATED_DOCUMENTS = 4
    GENERATE_ANSWER = 5
    FINISHED = 9
