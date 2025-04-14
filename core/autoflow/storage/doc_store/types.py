from uuid import UUID
from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, Field, computed_field

from autoflow.data_types import DataType
from autoflow.utils import uuid6
from autoflow.utils.hash import sha256


# Chunk


class Chunk(BaseModel):
    id: Optional[UUID] = Field(default_factory=uuid6.uuid7)
    text: str = Field(description="The text of the chunk.")
    text_vec: Optional[list[float]] = Field(
        default=None, description="The vector of text vectors."
    )
    meta: Optional[dict] = Field(
        default_factory=dict, description="The metadata of the chunk."
    )
    document_id: Optional[UUID] = Field(
        default=None, description="The id of the document that the chunk belongs to."
    )
    created_at: datetime = Field(default=None, description="The created time")
    updated_at: datetime = Field(default=None, description="The updated time")

    @computed_field
    @property
    def hash(self) -> Optional[str]:
        return sha256(self.text)


class RetrievedChunk(Chunk):
    score: Optional[float] = Field(description="The score of the chunk.", default=None)
    similarity_score: Optional[float] = Field(
        default=None, description="The similarity score of the chunk."
    )


# Document


class Document(BaseModel):
    id: Optional[UUID] = Field(default_factory=uuid6.uuid7)
    name: Optional[str] = Field(None, description="The name of the document.")
    content: str = Field(description="The content of the document.")
    data_type: Optional[DataType] = Field(
        default=None, description="The data type of the document."
    )
    meta: Optional[dict] = Field(
        default_factory=dict, description="The metadata of the document."
    )
    created_at: Optional[datetime] = Field(default=None, description="The created time")
    updated_at: Optional[datetime] = Field(default=None, description="The updated time")
    chunks: Optional[List[Chunk]] = Field(
        default_factory=list, description="The chunks of the document."
    )

    @computed_field
    @property
    def hash(self) -> Optional[str]:
        return sha256(self.content)


class DocumentDescriptor(BaseModel):
    id: UUID
    name: str
    # source_uri: str


class DocumentSearchResult(BaseModel):
    chunks: List[RetrievedChunk] = Field(
        default_factory=list, description="The chunks of the search result."
    )
    documents: List[DocumentDescriptor | Document] = Field(
        default_factory=list,
        description="The aggregated documents of the search result.",
    )
