import re

from app.models.knowledge_base import KnowledgeBase
from app.models.embed_model import DEFAULT_VECTOR_DIMENSION
from app.models.knowledge_base_scoped.registry import logger

DEFAULT_CHUNKS_TABLE_NAME = "chunks"
DEFAULT_ENTITIES_TABLE_NAME = "entities"
DEFAULT_RELATIONSHIPS_TABLE_NAME = "relationships"
CHUNKS_TABLE_PREFIX = "chunks_"
ENTITIES_TABLE_PREFIX = "entities_"
RELATIONSHIPS_TABLE_PREFIX = "relationships_"
KB_CHUNKS_TABLE_PATTERN = re.compile(r'^chunks_\d+$')
KB_ENTITIES_TABLE_PATTERN = re.compile(r'^entities_\d+$')
KB_RELATIONSHIPS_TABLE_PATTERN = re.compile(r'^relationships_\d+$')


def get_kb_chunks_table_name(knowledge_base: KnowledgeBase) -> str:
    return CHUNKS_TABLE_PREFIX + str(knowledge_base.id) if knowledge_base else DEFAULT_CHUNKS_TABLE_NAME


def get_kb_relationships_table_name(knowledge_base: KnowledgeBase) -> str:
    return RELATIONSHIPS_TABLE_PREFIX + str(knowledge_base.id) if knowledge_base else DEFAULT_RELATIONSHIPS_TABLE_NAME


def get_kb_entities_table_name(knowledge_base: KnowledgeBase) -> str:
    return ENTITIES_TABLE_PREFIX + str(knowledge_base.id) if knowledge_base else DEFAULT_ENTITIES_TABLE_NAME


def get_kb_vector_dims(kb: KnowledgeBase):
    vector_dimension = DEFAULT_VECTOR_DIMENSION
    if kb.embedding_model and kb.embedding_model.vector_dimension:
        vector_dimension = kb.embedding_model.vector_dimension
    else:
        logger.warning(f"This knowledge base doesn't configured a embedding model or this vector vector_dimension "
                       f"of the embedding model is miss.")
    return vector_dimension
