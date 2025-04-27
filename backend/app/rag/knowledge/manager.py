from .base import KnowledgeBase
from app.repositories import knowledge_base_repo


class KnowledgeBaseManager:
    def get(self, kb_id: str) -> KnowledgeBase:
        kb = knowledge_base_repo.must_get(kb_id)
        return KnowledgeBase(
            id=kb.id,
            name=kb.name,
            llm=kb.llm,
            embed_model=kb.embed_model,
            db_session=kb.db_session,
        )


kb_manager = KnowledgeBaseManager()
