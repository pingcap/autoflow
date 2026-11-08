from typing import Optional, Dict
from datetime import datetime

from sqlmodel import SQLModel, Field, Column, JSON, DateTime, func


class StaffActionLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    action: str
    action_time: datetime = Field(sa_column=Column(DateTime, server_default=func.now()))
    target_type: str
    target_id: int
    actor_id: Optional[str] = Field(default=None, max_length=36, index=True)
    actor_type: Optional[str] = Field(default=None, max_length=16)
    api_key_id: Optional[int] = None
    request_id: Optional[str] = Field(default=None, max_length=36, index=True)
    knowledge_base_id: Optional[int] = Field(default=None, index=True)
    target_identifier: Optional[str] = Field(default=None, max_length=128)
    changed_fields: Optional[list[str]] = Field(default=None, sa_column=Column(JSON))
    before: Dict = Field(default_factory=dict, sa_column=Column(JSON))
    after: Dict = Field(default_factory=dict, sa_column=Column(JSON))

    __tablename__ = "staff_action_logs"
