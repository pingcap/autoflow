from sqlmodel import select, Session, col
from fastapi_pagination import Params, Page
from fastapi_pagination.ext.sqlmodel import paginate

from app.models import Feedback, AdminFeedbackPublic, FeedbackFilters
from app.repositories.base_repo import BaseRepo

class FeedbackRepo(BaseRepo):
    model_cls = Feedback

    def paginate(
        self,
        session: Session,
        filters: FeedbackFilters,
        params: Params | None = Params(),
    ) -> Page[AdminFeedbackPublic]:
        # build the select statement via conditions
        stmt = select(Feedback)
        if filters.created_at_start:
            stmt = stmt.where(Feedback.created_at >= filters.created_at_start)
        if filters.created_at_end:
            stmt = stmt.where(Feedback.created_at <= filters.created_at_end)
        if filters.feedback_origin:
            stmt = stmt.where(col(Feedback.origin).contains(filters.feedback_origin))
        if filters.chat_id:
            stmt = stmt.where(Feedback.chat_id == filters.chat_id)
        if filters.feedback_type:
            stmt = stmt.where(Feedback.feedback_type == filters.feedback_type)
        if filters.user_id:
            stmt = stmt.where(Feedback.user_id == filters.user_id)
        
        stmt = stmt.order_by(Feedback.created_at.desc())
        return paginate(
            session, 
            stmt, 
            params,
            transformer=lambda items: [
                AdminFeedbackPublic(
                    **item.model_dump(),
                    chat_title=item.chat.title,
                    chat_origin=item.chat.origin,
                    chat_message_content=item.chat_message.content,
                    user_email=item.user.email if item.user else None,
                )
                for item in items
            ],
        )

feedback_repo = FeedbackRepo()
