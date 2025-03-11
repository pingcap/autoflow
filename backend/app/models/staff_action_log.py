from typing import Optional, Dict
from datetime import datetime

from sqlmodel import SQLModel, Field, Column, JSON, DateTime, func


class StaffActionLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    action: str
    action_time: datetime = Field(sa_column=Column(DateTime, server_default=func.now()))
    # TODO: Add knowledge base ID.
    target_type: str
    target_id: int
    before: Dict = Field(default_factory=dict, sa_column=Column(JSON))
    after: Dict = Field(default_factory=dict, sa_column=Column(JSON))

    __tablename__ = "staff_action_logs"
