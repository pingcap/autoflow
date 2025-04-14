import logging
from litellm import verbose_logger
from llama_index.llms.litellm import LiteLLM

verbose_logger.setLevel(logging.WARN)

LiteLLM = LiteLLM
