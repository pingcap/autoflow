import dspy

from autoflow.llms.chat_models import ChatModel


def get_dspy_lm_by_chat_model(chat_model: ChatModel) -> dspy.LM:
    return dspy.LM(
        model=chat_model.model,
        max_tokens=chat_model.max_tokens,
        **chat_model.additional_kwargs,
    )
