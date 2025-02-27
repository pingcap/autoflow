from typing import Optional, List

from pydantic import BaseModel


class QueryBundle(BaseModel):
    query_str: Optional[str] = None
    query_embedding: Optional[List[float]] = None
