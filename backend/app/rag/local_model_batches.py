"""Keep ordinary embedding/reranking batches within the sidecar's fixed budget."""


def local_model_batches(texts, *, query=""):
    if not isinstance(query, str) or len(query) > 8192:
        raise ValueError("Local model query is too large")
    batch, size = [], len(query)
    for text in texts:
        if not isinstance(text, str) or not 1 <= len(text) <= 8192:
            raise ValueError("Local model passages must contain 1 to 8192 characters")
        if len(batch) >= 32 or size + len(text) > 65536:
            yield batch
            batch, size = [], len(query)
        batch.append(text)
        size += len(text)
    if batch:
        yield batch
