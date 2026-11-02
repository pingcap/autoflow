from hmac import compare_digest

from app.models import Chat, ChatVisibility, User


def user_can_use_engine(engine, user: User | None) -> bool:
    return bool(
        engine
        and not engine.deleted_at
        and (engine.is_public or (user and user.is_superuser))
    )


def user_can_edit_chat(
    chat: Chat, user: User | None, browser_id: str | None = None
) -> bool:
    if chat.deleted_at:
        return False
    if user and user.is_superuser:
        return True
    if chat.user_id:
        return user is not None and chat.user_id == user.id
    return bool(
        browser_id
        and chat.browser_id
        and compare_digest(str(browser_id).encode(), str(chat.browser_id).encode())
    )


def user_can_view_chat(
    chat: Chat, user: User | None, browser_id: str | None = None
) -> bool:
    if chat.deleted_at:
        return False
    return chat.visibility == ChatVisibility.PUBLIC or user_can_edit_chat(
        chat, user, browser_id
    )
