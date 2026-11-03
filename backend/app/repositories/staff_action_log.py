from sqlmodel import Session

from app.models.staff_action_log import StaffActionLog
from app.repositories.base_repo import BaseRepo
from app.staff_action import create_staff_action_log


class StaffActionRepo(BaseRepo):
    model_cls = StaffActionLog

    def create_staff_action_log(
        self,
        session: Session,
        action,
        target_type,
        before,
        after,
        commit=True,
        *,
        target_id=None,
        knowledge_base_id=None,
    ) -> StaffActionLog:
        if target_id is None:
            target_id = after.get("id", before.get("id"))
        if target_id is None:
            raise ValueError("Audit target identifier is required")
        return create_staff_action_log(
            session,
            action,
            target_type,
            target_id,
            before,
            after,
            commit,
            knowledge_base_id=knowledge_base_id,
        )


staff_action_repo = StaffActionRepo()
