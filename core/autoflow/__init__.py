import os
from .main import Autoflow

if os.getenv("LITELLM_LOCAL_MODEL_COST_MAP") is None:
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"

__all__ = [
    "Autoflow",
]
