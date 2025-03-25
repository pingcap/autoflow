from typing import List, Any, Optional
from pydantic import Field

from autoflow.storage.tidb.embeddings.base import BaseEmbeddingFunction


def get_embeddings(
    api_key: str,
    api_base: str,
    model_name: str,
    input: List[str],
    timeout: int = 60,
    **kwargs: Any,
) -> List[List[float]]:
    """
    Retrieve embeddings for a given list of input strings using the specified model.

    Args:
        api_key (str): The API key for authentication.
        api_base (str): The base URL of the LiteLLM proxy server.
        model_name (str): The name of the model to use for generating embeddings.
        input (List[str]): A list of input strings for which embeddings are to be generated.
        timeout (float): The timeout value for the API call, default 60 secs.
        **kwargs (Any): Additional keyword arguments to be passed to the embedding function.

    Returns:
        List[List[float]]: A list of embeddings, where each embedding corresponds to an input string.
    """
    from litellm import embedding

    response = embedding(
        api_key=api_key,
        api_base=api_base,
        model=model_name,
        input=input,
        timeout=timeout,
        **kwargs,
    )
    return [result["embedding"] for result in response.data]


class LiteLLMEmbeddingFunction(BaseEmbeddingFunction):
    api_key: Optional[str] = Field(None, description="The API key for authentication.")
    api_base: Optional[str] = Field(
        None, description="The base URL of the model provider."
    )
    timeout: Optional[int] = Field(
        None, description="The timeout value for the API call."
    )

    def __init__(
        self,
        model_name: str,
        dimensions: Optional[int] = None,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        timeout: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(
            model_name=model_name,
            dimensions=dimensions,
            api_key=api_key,
            api_base=api_base,
            timeout=timeout,
            **kwargs,
        )
        if dimensions is None:
            self.dimensions = len(self.get_query_embedding("test"))

    def get_query_embedding(self, query: str) -> list[float]:
        embeddings = get_embeddings(
            api_key=self.api_key,
            api_base=self.api_base,
            model_name=self.model_name,
            dimensions=self.dimensions,
            timeout=self.timeout,
            input=[query],
        )
        return embeddings[0]

    def get_source_embedding(self, source: str) -> list[float]:
        embeddings = get_embeddings(
            api_key=self.api_key,
            api_base=self.api_base,
            model_name=self.model_name,
            dimensions=self.dimensions,
            timeout=self.timeout,
            input=[source],
        )
        return embeddings[0]

    def get_source_embeddings(self, sources: List[str]) -> list[list[float]]:
        embeddings = get_embeddings(
            api_key=self.api_key,
            api_base=self.api_base,
            model_name=self.model_name,
            dimensions=self.dimensions,
            timeout=self.timeout,
            input=sources,
        )
        return embeddings
