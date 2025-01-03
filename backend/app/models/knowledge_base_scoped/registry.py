import logging
from typing import Optional, Dict

from sqlalchemy.orm.decl_api import RegistryType
from sqlmodel.main import default_registry

from app.models.knowledge_base import KnowledgeBase
from app.models.patch.sql_model import SQLModel as PatchSQLModel

logger = logging.getLogger(__name__)


class KBSQLModelContext:
    # Each Knowledge Base has an independent sqlalchemy registry (aka. namespace), to avoid conflicts
    # between SQLModels with the same class name.
    registry: RegistryType = None

    chunk_model: Optional[PatchSQLModel] = None
    entity_model: Optional[PatchSQLModel] = None
    relationship_model: Optional[PatchSQLModel] = None

    def __init__(self, registry: RegistryType = default_registry):
        self.registry = registry


kb_sql_model_contexts: Dict[str, KBSQLModelContext] = {}


def get_kb_scoped_registry(kb: KnowledgeBase) -> KBSQLModelContext:
    ns = f"knowledge_base_{kb.id}"
    if ns not in kb_sql_model_contexts:
        registry = RegistryType(
            metadata=default_registry.metadata,
            class_registry=default_registry._class_registry.copy(),
        )
        kb_sql_model_contexts[ns] = KBSQLModelContext(registry)
    return kb_sql_model_contexts[ns]
