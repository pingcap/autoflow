from typing import Optional, List, Any
from uuid import UUID

from llama_index.core.workflow import Workflow

from app.models import ChatMessage, Chat


class AppChatEngine:
    workflow: Workflow

    def __init__(self, config: Any):
        self.config = config
        self.workflow = Workflow()

    async def chat(
        self,
        *,
        db_chat: Chat,
        user_question: Optional[UUID] = None,
        chat_history: List[ChatMessage],
    ):
        result = await self.workflow.run(
            user_question=user_question,
            chat_history=chat_history,
        )
        return result
