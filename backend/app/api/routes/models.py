from pydantic import BaseModel


class RequiredConfigStatus(BaseModel):
    default_llm: bool
    default_embedding_model: bool
    default_chat_engine: bool
    knowledge_base: bool


class OptionalConfigStatus(BaseModel):
    langfuse: bool
    default_reranker: bool


class NeedMigrationStatus(BaseModel):
    chat_engines_without_kb_configured: list[int]


class SystemConfigStatusResponse(BaseModel):
    required: RequiredConfigStatus
    optional: OptionalConfigStatus
    need_migration: NeedMigrationStatus
