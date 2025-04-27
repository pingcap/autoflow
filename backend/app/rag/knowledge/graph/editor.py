from typing import Optional, Tuple, List, Type

from llama_index.core.embeddings import resolve_embed_model
from llama_index.core.embeddings.utils import EmbedType
from llama_index.embeddings.openai import OpenAIEmbedding, OpenAIEmbeddingModelType

from sqlmodel import Session, select, SQLModel
from sqlalchemy.orm import joinedload
from sqlalchemy.orm.attributes import flag_modified

from app.models import EntityType
from app.rag.knowledge.graph.extractor.schema import Relationship as RelationshipAIModel
from app.rag.storage.graph_stores import TiDBGraphStore
from app.rag.storage.graph_stores.helpers import (
    get_entity_description_embedding,
    get_relationship_description_embedding,
    get_entity_metadata_embedding,
    get_query_embedding,
)
from app.staff_action import create_staff_action_log


# TODO: CRUD operations should move to TiDBGraphStore
class TiDBGraphEditor:
    _entity_db_model: Type[SQLModel]
    _relationship_db_model: Type[SQLModel]

    def __init__(
        self,
        knowledge_base_id: int,
        entity_db_model: Type[SQLModel],
        relationship_db_model: Type[SQLModel],
        embed_model: Optional[EmbedType] = None,
    ):
        self.knowledge_base_id = knowledge_base_id
        self._entity_db_model = entity_db_model
        self._relationship_db_model = relationship_db_model

        if embed_model:
            self._embed_model = resolve_embed_model(embed_model)
        else:
            self._embed_model = OpenAIEmbedding(
                model=OpenAIEmbeddingModelType.TEXT_EMBED_3_SMALL
            )
