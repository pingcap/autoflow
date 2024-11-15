from typing import Optional

from sqlalchemy import func
from sqlmodel import Session, select, SQLModel
from app.repositories.base_repo import BaseRepo

from app.models import (
    Entity as DBEntity,
)

class EntityRepo(BaseRepo):

    def __init__(self, entity_model: Optional[SQLModel] = DBEntity):
        self.model_cls = entity_model

    def count(self, session: Session):
        return session.scalar(select(func.count(self.model_cls.id)))


entity_repo = EntityRepo()
