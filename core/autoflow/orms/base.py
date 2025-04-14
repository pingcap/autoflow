import uuid
from datetime import datetime
from typing import Optional

from pytidb.schema import TableModel, Field
from pytidb.datatype import DateTime
from pytidb.sql import func

from autoflow.utils import uuid6


class UUIDBaseModel(TableModel):
    id: uuid.UUID = Field(default_factory=uuid6.uuid7, primary_key=True)
    # Use sa_type instead of sa_column, refer to https://github.com/tiangolo/sqlmodel/discussions/743
    created_at: Optional[datetime] = Field(
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={"server_default": func.now()},
    )
    updated_at: Optional[datetime] = Field(
        sa_type=DateTime(timezone=True),
        sa_column_kwargs={
            "server_default": func.now(),
            "onupdate": func.now(),
        },
    )
