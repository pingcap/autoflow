from typing import Optional

from sqlalchemy import func
from sqlmodel import Session, select, SQLModel
from app.repositories.base_repo import BaseRepo

from app.models import (
    Relationship as DBRelationship,
)

class RelationshipRepo(BaseRepo):

    def __init__(self, relationship_model: Optional[SQLModel] = DBRelationship):
        self.model_cls = relationship_model

    def count(self, session: Session):
        return session.scalar(select(func.count(self.model_cls.id)))


relationship_repo = RelationshipRepo()
