from http.client import HTTPException
from typing import TypeVar, Type

from fastapi import status, HTTPException
from sqlmodel import SQLModel, Session

T = TypeVar("T", bound=SQLModel)


def must_get(session: Session, model: Type[T], item_id: int) -> T:
    item = session.get(model, item_id)
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"{model.__name__} with ID {item_id} not found",
        )
    return item


def must_get_and_belong(
    session: Session, model: Type[T], item_id: int, user_id: int
) -> T:
    item = must_get(session, model, item_id)

    if not hasattr(item, "user_id"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{model.__name__} does not have a 'user_id' field",
        )

    if item.user_id != user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"{model.__name__} with ID {item_id} does not belong to user {user_id}",
        )

    return item
