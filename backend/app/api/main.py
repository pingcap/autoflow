from fastapi import APIRouter
from app.api.routes import (
    index,
    chat,
    user,
    api_key,
    feedback,
    document,
)
from app.api.admin_routes.knowledge_base.routes import (
    router as admin_knowledge_base_router,
)
from app.api.admin_routes.knowledge_base.graph.routes import (
    router as admin_kb_graph_router,
)
from app.api.admin_routes.knowledge_base.graph.entity.routes import (
    router as admin_kb_graph_entity_router,
)
from app.api.admin_routes.knowledge_base.graph.relationship.routes import (
    router as admin_kb_graph_relationship_router,
)
from app.api.admin_routes.knowledge_base.graph.knowledge.routes import (
    router as admin_kb_graph_knowledge_router,
)
from app.api.admin_routes.knowledge_base.data_source.routes import (
    router as admin_kb_data_source_router,
)
from app.api.admin_routes.knowledge_base.document.routes import (
    router as admin_kb_document_router,
)
from app.api.admin_routes.knowledge_base.chunk.routes import (
    router as admin_kb_chunk_router,
)
from app.api.admin_routes.document.routes import router as admin_document_router
from app.api.admin_routes.llm.routes import router as admin_llm_router
from app.api.admin_routes.embedding_model.routes import (
    router as admin_embedding_model_router,
)
from app.api.admin_routes.reranker_model.routes import (
    router as admin_reranker_model_router,
)
from app.api.admin_routes.chat.routes import router as admin_user_router
from app.api.admin_routes import (
    chat_engine as admin_chat_engine,
    feedback as admin_feedback,
    legacy_retrieve as admin_legacy_retrieve,
    site_setting as admin_site_settings,
    upload as admin_upload,
    stats as admin_stats,
    semantic_cache as admin_semantic_cache,
    langfuse as admin_langfuse,
    user as admin_user,
)
from app.api.admin_routes.evaluation import (
    evaluation_task as admin_evaluation_task,
    evaluation_dataset as admin_evaluation_dataset,
)
from app.api.routes.retrieve import (
    routes as retrieve_routes,
)

from app.auth.users import auth_backend, fastapi_users

api_router = APIRouter()
api_router.include_router(index.router, tags=["index"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(feedback.router, tags=["chat"])
api_router.include_router(user.router, tags=["user"])
api_router.include_router(api_key.router, tags=["auth"])
api_router.include_router(document.router, tags=["documents"])
api_router.include_router(retrieve_routes.router, tags=["retrieve"])
api_router.include_router(admin_user_router)
api_router.include_router(admin_chat_engine.router, tags=["admin/chat-engines"])
api_router.include_router(admin_document_router, tags=["admin/documents"])
api_router.include_router(admin_feedback.router)
api_router.include_router(admin_site_settings.router, tags=["admin/site_settings"])
api_router.include_router(admin_upload.router, tags=["admin/upload"])
api_router.include_router(admin_knowledge_base_router, tags=["admin/knowledge_base"])
api_router.include_router(admin_kb_graph_router, tags=["admin/knowledge_base/graph"])
api_router.include_router(
    admin_kb_graph_entity_router, tags=["admin/knowledge_base/graph/entity"]
)
api_router.include_router(
    admin_kb_graph_relationship_router, tags=["admin/knowledge_base/graph/relationship"]
)
api_router.include_router(
    admin_kb_graph_knowledge_router, tags=["admin/knowledge_base/graph/knowledge"]
)
api_router.include_router(
    admin_kb_data_source_router, tags=["admin/knowledge_base/data_source"]
)
api_router.include_router(
    admin_kb_document_router, tags=["admin/knowledge_base/document"]
)
api_router.include_router(admin_kb_chunk_router, tags=["admin/knowledge_base/chunk"])
api_router.include_router(admin_llm_router, tags=["admin/llm"])
api_router.include_router(admin_embedding_model_router, tags=["admin/embedding_model"])
api_router.include_router(admin_reranker_model_router, tags=["admin/reranker_model"])
api_router.include_router(admin_langfuse.router, tags=["admin/langfuse"])
api_router.include_router(admin_legacy_retrieve.router, tags=["admin/retrieve_old"])
api_router.include_router(admin_stats.router, tags=["admin/stats"])
api_router.include_router(admin_semantic_cache.router, tags=["admin/semantic_cache"])
api_router.include_router(admin_evaluation_task.router, tags=["admin/evaluation/task"])

api_router.include_router(
    admin_evaluation_dataset.router, tags=["admin/evaluation/dataset"]
)
api_router.include_router(admin_user.router)

api_router.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth", tags=["auth"]
)
