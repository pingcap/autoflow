from typing import Optional
from uuid import UUID
from datetime import datetime

from pydantic import EmailStr
from sqlmodel import (
    Field,
    SQLModel,
    DateTime,
    func,
    Relationship as SQLRelationship,
)

from app.models.base import UpdatableBaseModel, UUIDBaseModel


class User(UUIDBaseModel, UpdatableBaseModel, table=True):
    email: EmailStr = Field(index=True, unique=True, nullable=False)
    hashed_password: str
    auth_version: int = Field(
        default=0, nullable=False, sa_column_kwargs={"server_default": "0"}
    )
    is_active: bool = Field(True, nullable=False)
    is_superuser: bool = Field(False, nullable=False)
    is_verified: bool = Field(False, nullable=False)

    __tablename__ = "users"


class UserSession(SQLModel, table=True):
    token: str = Field(max_length=43, primary_key=True)
    auth_version: int = Field(
        default=0, nullable=False, sa_column_kwargs={"server_default": "0"}
    )
    created_at: Optional[datetime] = Field(
        default=None,
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={"server_default": func.now()},
    )
    user_id: UUID = Field(foreign_key="users.id", nullable=False)
    user: User = SQLRelationship(
        sa_relationship_kwargs={
            "lazy": "joined",
            "primaryjoin": "UserSession.user_id == User.id",
        },
    )

    __tablename__ = "user_sessions"
