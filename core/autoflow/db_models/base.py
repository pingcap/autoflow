from uuid import UUID
from sqlmodel import Field, SQLModel
from sqlalchemy.types import TypeDecorator, Integer

from autoflow.utils.uuid6 import uuid7


class UUIDBaseModel(SQLModel):
    id: UUID = Field(
        default_factory=uuid7,
        primary_key=True,
        index=True,
        nullable=False,
        sa_column_kwargs={"sort_order": -1},
    )


class IntEnumType(TypeDecorator):
    """
    IntEnumType is a custom TypeDecorator that handles conversion between
    integer values in the database and Enum types in Python.

    This replaces the previous SmallInteger implementation to resolve Pydantic
    serialization warnings. When using SmallInteger, SQLAlchemy would return raw
    integers from the database (e.g., 0 or 1), causing Pydantic validation warnings
    since it expects proper Enum types.
    """

    impl = Integer

    def __init__(self, enum_class, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.enum_class = enum_class

    def process_bind_param(self, value, dialect):
        # enum -> int
        if isinstance(value, self.enum_class):
            return value.value
        elif value is None:
            return None
        raise ValueError(f"Invalid value for {self.enum_class}: {value}")

    def process_result_value(self, value, dialect):
        # int -> enum
        if value is not None:
            return self.enum_class(value)
        return None
