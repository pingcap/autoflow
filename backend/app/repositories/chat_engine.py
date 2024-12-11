from typing import Optional
from datetime import datetime, UTC

from sqlalchemy import func
from sqlmodel import select, Session, update
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlalchemy.orm.attributes import flag_modified

from app.models import ChatEngine, ChatEngineUpdate
from app.repositories.base_repo import BaseRepo


class ChatEngineRepo(BaseRepo):
    model_cls = ChatEngine

    def get(self, session: Session, id: int) -> Optional[ChatEngine]:
        return session.exec(
            select(ChatEngine).where(ChatEngine.id == id, ChatEngine.deleted_at == None)
        ).first()

    def paginate(
        self,
        session: Session,
        params: Params | None = Params(),
    ) -> Page[ChatEngine]:
        query = select(ChatEngine).where(ChatEngine.deleted_at == None)
        # Make sure the default engine is always on top
        query = query.order_by(ChatEngine.is_default.desc(), ChatEngine.name)
        return paginate(session, query, params)

    def get_default_engine(self, session: Session) -> Optional[ChatEngine]:
        return session.exec(
            select(ChatEngine).where(
                ChatEngine.is_default == True, ChatEngine.deleted_at == None
            )
        ).first()

    def has_default(self, session: Session) -> bool:
        return (
            session.scalar(
                select(func.count(ChatEngine.id)).where(
                    ChatEngine.is_default == True, ChatEngine.deleted_at == None
                )
            )
            > 0
        )

    def get_engine_by_name(self, session: Session, name: str) -> Optional[ChatEngine]:
        return session.exec(
            select(ChatEngine).where(
                ChatEngine.name == name, ChatEngine.deleted_at == None
            )
        ).first()

    def create(self, session: Session, obj: ChatEngine):
        if obj.is_default:
            session.exec(
                update(ChatEngine)
                .where(ChatEngine.id != obj.id)
                .values(is_default=False)
            )
        session.add(obj)
        session.commit()
        session.refresh(obj)
        return obj

    def update(
        self,
        session: Session,
        chat_engine: ChatEngine,
        chat_engine_in: ChatEngineUpdate,
    ) -> ChatEngine:
        set_default = chat_engine_in.is_default
        for field, value in chat_engine_in.model_dump(exclude_unset=True).items():
            setattr(chat_engine, field, value)
            flag_modified(chat_engine, field)

        if set_default:
            session.exec(
                update(ChatEngine)
                .where(ChatEngine.id != chat_engine.id)
                .values(is_default=False)
            )
        session.commit()
        session.refresh(chat_engine)
        return chat_engine

    def delete(self, session: Session, chat_engine: ChatEngine) -> ChatEngine:
        chat_engine.deleted_at = datetime.now(UTC)
        session.commit()
        session.refresh(chat_engine)
        return chat_engine


chat_engine_repo = ChatEngineRepo()
