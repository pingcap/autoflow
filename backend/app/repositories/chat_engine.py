from typing import Optional
from datetime import datetime, UTC

from sqlalchemy import func
from sqlmodel import select, Session, update
from app.exceptions import ChatEngineNotFound
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate
from sqlalchemy.orm.attributes import flag_modified

from app.models.chat_engine import ChatEngine, ChatEngineUpdate
from app.repositories.base_repo import BaseRepo


class ChatEngineRepo(BaseRepo):
    model_cls = ChatEngine

    def get(
        self, session: Session, id: int, need_public: bool = False
    ) -> Optional[ChatEngine]:
        query = select(ChatEngine).where(
            ChatEngine.id == id, ChatEngine.deleted_at == None
        )
        if need_public:
            query = query.where(ChatEngine.is_public == True)
        return session.exec(query).first()

    def must_get(
        self, session: Session, id: int, need_public: bool = False
    ) -> ChatEngine:
        chat_engine = self.get(session, id, need_public)
        if chat_engine is None:
            raise ChatEngineNotFound(id)
        return chat_engine

    def paginate(
        self,
        session: Session,
        params: Params | None = Params(),
        need_public: bool = False,
    ) -> Page[ChatEngine]:
        query = select(ChatEngine).where(ChatEngine.deleted_at == None)
        if need_public:
            query = query.where(ChatEngine.is_public == True)
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
        chat_engine_update: ChatEngineUpdate,
    ) -> ChatEngine:
        set_default = chat_engine_update.is_default
        for field, value in chat_engine_update.model_dump(exclude_unset=True).items():
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
