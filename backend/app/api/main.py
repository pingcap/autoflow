from fastapi import APIRouter, Depends


from app.api.routes import (
    index,
    chat,
    user,
    api_key,
    feedback,
)
from app.api.admin_routes.knowledge_base.routes import router as admin_knowledge_base_router
from app.api.admin_routes.knowledge_base.graph.routes import router as admin_knowledge_base_graph_router
from app.api.admin_routes.document.routes import router as admin_document_router
from app.api.admin_routes.embedding_model.routes import router as admin_embedding_model_router
from app.api.admin_routes import (
    rag_index as admin_rag_index,
    chat_engine as admin_chat_engine,
    feedback as admin_feedback,
    site_setting as admin_site_settings,
    graph_editor as admin_graph_editor,
    upload as admin_upload,
    data_source as admin_data_source,
    llm as admin_llm,
    retrieve as admin_retrieve,
    stats as admin_stats,
    semantic_cache as admin_semantic_cache,
)
from app.auth.users import auth_backend, fastapi_users
from app.api.deps import current_superuser

api_router = APIRouter()
api_router.include_router(index.router, tags=["index"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(feedback.router, tags=["chat"])
api_router.include_router(user.router, tags=["user"])
api_router.include_router(api_key.router, tags=["auth"])

api_router.include_router(admin_rag_index.router, tags=["admin/rag_index"])
api_router.include_router(admin_chat_engine.router, tags=["admin/chat_engine"])
api_router.include_router(admin_document_router, tags=["admin/documents"])
api_router.include_router(admin_feedback.router, tags=["admin/feedback"])
api_router.include_router(admin_site_settings.router, tags=["admin/site_settings"])
api_router.include_router(
    admin_graph_editor.router, tags=["admin/graph_editor"], dependencies=[Depends(current_superuser)]
)
api_router.include_router(admin_upload.router, tags=["admin/upload"])
api_router.include_router(admin_data_source.router, tags=["admin/data_source"])
api_router.include_router(admin_knowledge_base_router, tags=["admin/knowledge_base"])
api_router.include_router(admin_knowledge_base_graph_router, tags=["admin/knowledge_base/graph_editor"])
api_router.include_router(admin_llm.router, tags=["admin/llm"])

api_router.include_router(admin_embedding_model_router, tags=["admin/embedding_model"])
api_router.include_router(admin_retrieve.router, tags=["admin/retrieve"])
api_router.include_router(admin_stats.router, tags=["admin/stats"])
api_router.include_router(admin_semantic_cache.router, tags=["admin/semantic_cache"])

api_router.include_router(
    fastapi_users.get_auth_router(auth_backend), prefix="/auth", tags=["auth"]
)
# api_router.include_router(
#     fastapi_users.get_register_router(UserRead, UserCreate),
#     prefix="/auth",
#     tags=["auth"],
# )
# api_router.include_router(
#     fastapi_users.get_reset_password_router(),
#     prefix="/auth",
#     tags=["auth"],
# )
# api_router.include_router(
#     fastapi_users.get_verify_router(UserRead),
#     prefix="/auth",
#     tags=["auth"],
# )
