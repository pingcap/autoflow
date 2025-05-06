from http import HTTPStatus
from uuid import UUID

from fastapi import HTTPException

# Common


class InternalServerError(HTTPException):
    def __init__(self):
        super().__init__(HTTPStatus.INTERNAL_SERVER_ERROR)


# Chat


class ChatException(HTTPException):
    pass


class ChatNotFound(ChatException):
    status_code = 404

    def __init__(self, chat_id: UUID):
        self.detail = f"chat #{chat_id} is not found"


class ChatMessageNotFound(ChatException):
    status_code = 404

    def __init__(self, message_id: int):
        self.detail = f"chat message #{message_id} is not found"


# LLM


class LLMException(HTTPException):
    pass


class LLMNotFound(LLMException):
    status_code = 404

    def __init__(self, llm_id: int):
        self.detail = f"llm #{llm_id} is not found"


class DefaultLLMNotFound(LLMException):
    status_code = 404

    def __init__(self):
        self.detail = "default llm is not found"


# Embedding model


class EmbeddingModelException(HTTPException):
    pass


class EmbeddingModelNotFound(EmbeddingModelException):
    status_code = 404

    def __init__(self, model_id: int):
        self.detail = f"embedding model with id {model_id} not found"


class DefaultEmbeddingModelNotFound(EmbeddingModelException):
    status_code = 404

    def __init__(self):
        self.detail = "default embedding model is not found"


# Reranker model


class RerankerModelException(HTTPException):
    pass


class RerankerModelNotFound(RerankerModelException):
    status_code = 404

    def __init__(self, model_id: int):
        self.detail = f"reranker model #{model_id} not found"


class DefaultRerankerModelNotFound(RerankerModelException):
    status_code = 404

    def __init__(self):
        self.detail = "default reranker model is not found"


# Knowledge base


class KBException(HTTPException):
    pass


class KBNotFound(KBException):
    status_code = 404

    def __init__(self, knowledge_base_id: int):
        self.detail = f"knowledge base #{knowledge_base_id} is not found"


class KBDataSourceNotFound(KBException):
    status_code = 404

    def __init__(self, kb_id: int, data_source_id: int):
        self.detail = (
            f"data source #{data_source_id} is not found in knowledge base #{kb_id}"
        )


class KBNoLLMConfigured(KBException):
    status_code = 500

    def __init__(self):
        self.detail = "must configured a LLM for knowledge base"


class KBNoEmbedModelConfigured(KBException):
    status_code = 500

    def __init__(self):
        self.detail = "must configured a embedding model for knowledge base"


class KBNoVectorIndexConfigured(KBException):
    status_code = 500

    def __init__(self):
        self.detail = "must configured vector index as one of the index method for knowledge base, which is required for now"


class KBNotAllowedUpdateEmbedModel(KBException):
    status_code = 500

    def __init__(self):
        self.detail = "update embedding model is not allowed once the knowledge base has been created"


class KBIsUsedByChatEngines(KBException):
    status_code = 500

    def __init__(self, kb_id, chat_engines_num: int):
        self.detail = f"knowledge base #{kb_id} is used by {chat_engines_num} chat engines, please unlink them before deleting"


# Document


class DocumentException(HTTPException):
    pass


class DocumentNotFound(DocumentException):
    status_code = 404

    def __init__(self, document_id: int):
        self.detail = f"document #{document_id} is not found"


# Chat engine


class ChatEngineException(HTTPException):
    pass


class ChatEngineNotFound(ChatEngineException):
    status_code = 404

    def __init__(self, chat_engine_id: int):
        self.detail = f"chat engine #{chat_engine_id} is not found"


class DefaultChatEngineCannotBeDeleted(ChatEngineException):
    status_code = 400

    def __init__(self, chat_engine_id: int):
        self.detail = f"default chat engine #{chat_engine_id} cannot be deleted"
