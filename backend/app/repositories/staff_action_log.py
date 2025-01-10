from sqlmodel import Session

from app.models.staff_action_log import StaffActionLog
from app.repositories.base_repo import BaseRepo


class StaffActionRepo(BaseRepo):
    model_cls = StaffActionLog

    def create_staff_action_log(
        self, session: Session, action, target_type, before, after, commit=True
    ) -> StaffActionLog:
        staff_action_log = StaffActionLog(
            action=action, target_type=target_type, before=before, after=after
        )
        session.add(staff_action_log)
        if commit:
            session.commit()
        return staff_action_log


staff_action_repo = StaffActionRepo()
