from typing import List

from llama_index.core import QueryBundle

from app.rag.types import ChatMessage


def get_query_bundle_from_chat(
    user_question, chat_history: List[ChatMessage]
) -> QueryBundle:
    query_str = user_question
    if len(chat_history) > 0:
        chat_messages = [
            f"{message.role.value}: {message.content}" for message in chat_history
        ]
        query_with_history = (
            "++++ Chat History ++++\n"
            + "\n".join(chat_messages)
            + "++++ Chat History ++++\n"
        )
        query_str = query_with_history + "\n\nThen the user asks:\n" + user_question
    return QueryBundle(query_str=query_str)
