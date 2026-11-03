from sqlmodel import Session

from app.security.audit import make_audit_record


def create_staff_action_log(
    session: Session,
    action,
    target_type,
    target_id,
    before,
    after,
    commit=True,
    *,
    knowledge_base_id=None,
):
    fields = {
        key
        for key in set(before) | set(after)
        if key not in before or key not in after or before[key] != after[key]
    }
    staff_action_log = make_audit_record(
        action, target_type, target_id, fields, knowledge_base_id
    )
    session.add(staff_action_log)
    if commit:
        session.commit()
    return staff_action_log
