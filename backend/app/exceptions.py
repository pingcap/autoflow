from http import HTTPStatus
from fastapi import HTTPException


class InternalServerError(HTTPException):
    def __init__(self):
        super().__init__(HTTPStatus.INTERNAL_SERVER_ERROR)


class ChatException(Exception):
    pass


class ChatNotFound(ChatException):
    pass


class DBLLMNotFoundError(HTTPException):
    status_code = 404

    def __init__(self, knowledge_base_id: int):
        self.detail = f"llm #{knowledge_base_id} is not found"


class KnowledgeBaseNotFoundError(HTTPException):
    status_code = 404

    def __init__(self, knowledge_base_id: int):
        self.detail = f"knowledge base #{knowledge_base_id} is not found"

class KBNoLLMConfiguredError(HTTPException):
    status_code = 500

    def __init__(self):
        self.detail = f"Must configured a LLM for knowledge base"


class KBNoEmbedModelConfiguredError(HTTPException):
    status_code = 500

    def __init__(self):
        self.detail = f"Must configured a embedding model for knowledge base"


class KBNoVectorIndexConfiguredError(HTTPException):
    status_code = 500

    def __init__(self):
        self.detail = f"Must configured vector index as one of the index method for knowledge base, which is required for now"


class EmbeddingModelNotFoundError(HTTPException):
    status_code = 404

    def __init__(self, model_id: int):
        self.detail = f"Embedding model with id {model_id} not found"


class NotAllowedUpdateEmbeddingModel(HTTPException):
    status_code = 500

    def __init__(self):
        self.detail = f"Doesn't allowed update embedding model once knowledge base been created"