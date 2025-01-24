from typing import List

from llama_index.core.schema import NodeWithScore

from app.rag.retrievers.chunk.schema import RetrievedChunk


def map_nodes_to_chunks(nodes_with_score: List[NodeWithScore]) -> List[RetrievedChunk]:
    return [
        RetrievedChunk(
            id=ns.node.node_id,
            text=ns.node.text,
            metadata=ns.node.metadata,
            document_id=ns.node.metadata["document_id"],
            score=ns.score,
        )
        for ns in nodes_with_score
    ]
