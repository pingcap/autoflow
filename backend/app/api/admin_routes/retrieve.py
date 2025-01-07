from typing import Optional, List

from fastapi import APIRouter
from app.models import Document
from app.api.admin_routes.models import RetrieveRequest
from app.api.deps import SessionDep, CurrentSuperuserDep
from app.rag.retrieve import RetrieveService
from llama_index.core.schema import NodeWithScore
from app.rag.retrieve import retrieve_service

router = APIRouter()


@router.get("/admin/retrieve/documents")
async def retrieve_documents(
    session: SessionDep,
    user: CurrentSuperuserDep,
    question: str,
    chat_engine: str = "default",
    top_k: Optional[int] = 5,
) -> List[Document]:
    return retrieve_service.chat_engine_retrieve_documents(
        session, question, top_k, chat_engine
    )


@router.get("/admin/retrieve/chunks")
async def retrieve_chunks(
    session: SessionDep,
    user: CurrentSuperuserDep,
    request: RetrieveRequest,
) -> List[NodeWithScore]:
    retrieve_service = RetrieveService(session, request.chat_engine)
    return retrieve_service.retrieve_chunks(request.query, top_k=request.top_k)
