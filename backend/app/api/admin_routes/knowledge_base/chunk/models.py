from pydantic import BaseModel


class RetrieveChunkRequest(BaseModel):
    query: str
    top_k: int = 10
    similarity_top_k: int = None
    oversampling_factor: int = 5
    enable_kg_enhance_query_refine: bool = True
