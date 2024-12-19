import enum


class LLMProvider(str, enum.Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    ANTHROPIC_VERTEX = "anthropic_vertex"
    OPENAI_LIKE = "openai_like"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"


class EmbeddingProvider(str, enum.Enum):
    OPENAI = "openai"
    JINA = "jina"
    COHERE = "cohere"
    OLLAMA = "ollama"
    LOCAL = "local"
    OPENAI_LIKE = "openai_like"


class RerankerProvider(str, enum.Enum):
    JINA = "jina"
    COHERE = "cohere"
    BAISHENG = "baisheng"
    LOCAL = "local"


class MimeTypes(str, enum.Enum):
    PLAIN_TXT = "text/plain"
    MARKDOWN = "text/markdown"
    PDF = "application/pdf"
    DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    PPTX = "application/vnd.openxmlformats-officedocument.presentationml.presentation"
    XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    CSV = "text/csv"
