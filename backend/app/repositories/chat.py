from sqlalchemy import and_, false
from sqlalchemy.orm import lazyload
import re
import enum
from uuid import UUID
from typing import Optional, List
from datetime import datetime, UTC, date, timedelta
from collections import defaultdict

from sqlmodel import select, Session, or_, func, case, desc, col
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate

from app.models import Chat, User, ChatMessage, ChatUpdate, ChatFilters, ChatOrigin
from app.repositories.base_repo import BaseRepo
from app.exceptions import ChatNotFound, ChatMessageNotFound


class ChatRepo(BaseRepo):
    model_cls = Chat

    def paginate(
        self,
        session: Session,
        user: User | None,
        browser_id: str | None,
        filters: ChatFilters,
        params: Params | None = Params(),
    ) -> Page[Chat]:
        query = select(Chat).where(Chat.deleted_at == None)
        if user:
            if not user.is_superuser:
                query = query.where(
                    or_(
                        Chat.user_id == user.id,
                        and_(Chat.user_id == None, Chat.browser_id == browser_id)
                        if browser_id
                        else false(),
                    )
                )
        else:
            query = (
                query.where(Chat.browser_id == browser_id, Chat.user_id == None)
                if browser_id
                else query.where(false())
            )

        # filters
        if filters.created_at_start:
            query = query.where(Chat.created_at >= filters.created_at_start)
        if filters.created_at_end:
            query = query.where(Chat.created_at <= filters.created_at_end)
        if filters.updated_at_start:
            query = query.where(Chat.updated_at >= filters.updated_at_start)
        if filters.updated_at_end:
            query = query.where(Chat.updated_at <= filters.updated_at_end)
        if filters.chat_origin:
            query = query.where(col(Chat.origin).contains(filters.chat_origin))
        # if filters.user_id:
        #     query = query.where(Chat.user_id == filters.user_id)
        if filters.engine_id:
            query = query.where(Chat.engine_id == filters.engine_id)

        query = query.order_by(Chat.created_at.desc())
        return paginate(session, query, params)

    def get(
        self,
        session: Session,
        chat_id: UUID,
    ) -> Optional[Chat]:
        return session.exec(
            select(Chat).where(Chat.id == chat_id, Chat.deleted_at == None)
        ).first()

    def must_get(
        self,
        session: Session,
        chat_id: UUID,
    ) -> Chat:
        chat = self.get(session, chat_id)
        if not chat:
            raise ChatNotFound(chat_id)
        return chat

    def update(
        self,
        session: Session,
        chat: Chat,
        chat_update: ChatUpdate,
    ) -> Chat:
        for field, value in chat_update.model_dump(exclude_unset=True).items():
            if isinstance(value, enum.Enum):
                value = value.value
            setattr(chat, field, value)
        session.commit()
        session.refresh(chat)
        return chat

    def delete(self, session: Session, chat: Chat):
        chat.deleted_at = datetime.now(UTC)
        session.add(chat)
        session.commit()

    def get_last_message(self, session: Session, chat: Chat) -> Optional[ChatMessage]:
        return session.exec(
            select(ChatMessage)
            .where(ChatMessage.chat_id == chat.id)
            .order_by(ChatMessage.ordinal.desc())
        ).first()

    def get_messages(
        self,
        session: Session,
        chat: Chat,
    ) -> List[ChatMessage]:
        return session.exec(
            select(ChatMessage)
            .where(ChatMessage.chat_id == chat.id)
            .order_by(ChatMessage.ordinal.asc())
        ).all()

    def get_context_messages(self, session: Session, chat: Chat, remaining_chars: int):
        """Fetch only bounded text for the latest model context, never source metadata."""
        remaining_chars = max(0, min(40000, remaining_chars))
        rows = session.exec(
            select(
                ChatMessage.role,
                func.substr(ChatMessage.content, -20000).label("content"),
            )
            .where(
                ChatMessage.chat_id == chat.id,
                ChatMessage.role.in_(["user", "assistant"]),
                ChatMessage.error.is_(None),
            )
            .order_by(ChatMessage.ordinal.desc(), ChatMessage.id.desc())
            .limit(19)
        ).all()
        result = []
        for role, content in rows:
            if remaining_chars == 0:
                break
            content = (content or "")[-remaining_chars:]
            if content:
                result.append((role, content))
                remaining_chars -= len(content)
        return result[::-1]

    def get_message(
        self,
        session: Session,
        chat_message_id: int,
    ) -> Optional[ChatMessage]:
        return session.exec(
            select(ChatMessage).where(
                ChatMessage.id == chat_message_id,
                ChatMessage.chat.has(Chat.deleted_at == None),
            )
        ).first()

    def must_get_message(
        self,
        session: Session,
        chat_message_id: int,
    ):
        msg = self.get_message(session, chat_message_id)
        if not msg:
            raise ChatMessageNotFound(chat_message_id)
        return msg

    def create_message(
        self,
        session: Session,
        chat: Chat,
        chat_message: ChatMessage,
    ) -> ChatMessage:
        if not chat_message.ordinal:
            last_message = self.get_last_message(session, chat)
            if last_message:
                ordinal = last_message.ordinal + 1
            else:
                ordinal = 1
            chat_message.ordinal = ordinal
        chat_message.chat_id = chat.id
        chat_message.user_id = chat.user_id
        session.add(chat_message)
        session.commit()
        session.refresh(chat_message)
        return chat_message

    def _answer_cache_query(self, chat: Chat, cache_context: str, days: int):
        """Reuse only this principal's completed first-turn answers in this context.

        `chat` must come from the authorized ChatFlow, never from request fields.
        Even administrators do not get a cross-owner cache bypass.
        """
        query = (
            select(ChatMessage)
            .options(lazyload("*"))
            .join(Chat, Chat.id == ChatMessage.chat_id)
            .where(
                Chat.deleted_at.is_(None),
                Chat.engine_id == chat.engine_id,
                Chat.visibility == chat.visibility,
                ChatMessage.role == "assistant",
                ChatMessage.ordinal == 2,
                ChatMessage.is_best_answer.is_(True),
                ChatMessage.finished_at.is_not(None),
                ChatMessage.created_at >= datetime.now(UTC) - timedelta(days=days),
                ChatMessage.meta["_answer_cache_context"].as_string() == cache_context,
            )
            .order_by(desc(ChatMessage.created_at), desc(ChatMessage.id))
            .limit(1)
        )
        if (
            chat.deleted_at
            or not chat.engine_id
            or not re.fullmatch(r"[0-9a-f]{64}", cache_context)
        ):
            return query.where(false())
        if chat.user_id:
            query = query.where(Chat.user_id == chat.user_id)
            return query.where(
                Chat.api_key_id == chat.api_key_id
                if chat.api_key_id is not None
                else Chat.api_key_id.is_(None)
            )
        if chat.browser_id and chat.api_key_id is None:
            return query.where(
                Chat.user_id.is_(None),
                Chat.api_key_id.is_(None),
                Chat.browser_id == chat.browser_id,
            )
        return query.where(false())

    def find_recent_assistant_messages_by_goal(
        self, session: Session, goal_key: str, *, chat: Chat, cache_context: str
    ) -> List[ChatMessage]:
        """Return at most one scoped answer matching the complete goal and format."""
        query = self._answer_cache_query(chat, cache_context, days=15).where(
            ChatMessage.meta["_answer_cache_goal"].as_string() == goal_key
        )
        return session.exec(query).all()

    def find_best_answer_for_question(
        self, session: Session, user_question: str, *, chat: Chat, cache_context: str
    ) -> List[ChatMessage]:
        """Find best answer messages for a specific user question.

        This method finds assistant messages that:
        1. Are marked as best answers
        2. Are responses (ordinal=2) to the exact user question
        3. Were created within the last 15 days

        Args:
            session: Database session
            user_question: The exact question text to search for

        Returns:
            List of matching assistant messages marked as best answers
        """
        # Question matching is data only; all answer reads use the scoped query.
        matching_chat_ids = select(ChatMessage.chat_id).where(
            ChatMessage.role == "user",
            ChatMessage.ordinal == 1,
            ChatMessage.content == user_question.strip(),
        )

        # Finally, get the best answers that correspond to the matching user questions
        query = self._answer_cache_query(chat, cache_context, days=15).where(
            ChatMessage.chat_id.in_(matching_chat_ids),
        )
        return session.exec(query).all()

    def chat_trend_by_user(
        self, session: Session, start_date: date, end_date: date
    ) -> List[dict]:
        start_at = datetime.combine(start_date, datetime.min.time(), UTC)
        end_at = datetime.combine(end_date, datetime.max.time(), UTC)
        query = (
            select(
                func.date(Chat.created_at).label("date"),
                func.sum(case((Chat.user_id.isnot(None), 1), else_=0)).label("user"),
                func.sum(case((Chat.user_id.is_(None), 1), else_=0)).label("anonymous"),
            )
            .where(Chat.created_at.between(start_at, end_at))
            .group_by(func.date(Chat.created_at))
            .order_by(func.date(Chat.created_at))
        )
        result = session.exec(query)
        return [
            {"date": row.date, "user": int(row.user), "anonymous": int(row.anonymous)}
            for row in result
        ]

    def chat_trend_by_origin(
        self, session: Session, start_date: date, end_date: date
    ) -> List[dict]:
        start_at = datetime.combine(start_date, datetime.min.time(), UTC)
        end_at = datetime.combine(end_date, datetime.max.time(), UTC)
        query = (
            select(
                func.count(Chat.id).label("count"),
                func.date(Chat.created_at).label("date"),
                Chat.origin,
            )
            .where(Chat.created_at.between(start_at, end_at))
            .group_by(func.date(Chat.created_at), Chat.origin)
            .order_by(func.date(Chat.created_at))
        )
        result = session.exec(query)

        date_origin_counts = defaultdict(lambda: defaultdict(int))
        origins = set()

        for row in result:
            date_origin_counts[row.date][row.origin] = row.count
            origins.add(row.origin)

        stats = []
        for d, origin_counts in date_origin_counts.items():
            stat = {"date": d}
            for origin in origins:
                stat[origin] = origin_counts[origin]
            stats.append(stat)

        stats.sort(key=lambda x: x["date"])
        return stats

    def list_chat_origins(
        self,
        db_session: Session,
        search: Optional[str] = None,
        params: Params = Params(),
    ) -> Page[ChatOrigin]:
        query = (
            select(Chat.origin, func.count(Chat.id).label("chats"))
            .where(Chat.deleted_at == None)
            .where(Chat.origin != None)
            .where(Chat.origin != "")
        )

        if search:
            query = query.where(Chat.origin.ilike(f"%{search}%"))

        query = query.group_by(Chat.origin).order_by(desc("chats"))

        return paginate(
            db_session,
            query,
            params,
            transformer=lambda chats: [
                ChatOrigin(origin=chat.origin, chats=chat.chats) for chat in chats
            ],
        )


chat_repo = ChatRepo()
