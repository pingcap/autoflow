import logging
from typing import Optional

import dspy

from autoflow.knowledge_graph.extractors.simple import SimpleKGExtractor
from autoflow.knowledge_graph.retrievers.weighted import WeightedGraphRetriever
from autoflow.knowledge_graph.types import (
    GeneratedKnowledgeGraph,
    RetrievedKnowledgeGraph,
)
from autoflow.storage.doc_store.types import Chunk
from autoflow.storage.graph_store.base import GraphStore
from autoflow.storage.graph_store.types import EntityType, KnowledgeGraph
from autoflow.types import BaseComponent


logger = logging.getLogger(__name__)


class KnowledgeGraphIndex(BaseComponent):
    def __init__(self, graph_store: GraphStore, dspy_lm: dspy.LM):
        super().__init__()
        self._graph_store = graph_store
        self._dspy_lm = dspy_lm
        self._extractor = SimpleKGExtractor(self._dspy_lm)

    def add_from_text(self, text: str) -> Optional[KnowledgeGraph]:
        knowledge_graph = self._extractor.extract(text)
        return self.add(knowledge_graph)

    def add_from_chunk(self, chunk: Chunk) -> Optional[KnowledgeGraph]:
        # Check if the chunk has been added.
        exists_relationships = self._graph_store.list_relationships(chunk_id=chunk.id)
        if len(exists_relationships) > 0:
            logger.warning(
                "The subgraph of chunk %s has already been added, skip.", chunk.id
            )
            return None

        knowledge_graph = self._extractor.extract(chunk)
        return self.add(knowledge_graph)

    def add(self, knowledge_graph: GeneratedKnowledgeGraph) -> Optional[KnowledgeGraph]:
        if (
            len(knowledge_graph.entities) == 0
            or len(knowledge_graph.relationships) == 0
        ):
            logger.warning(
                "Entities or relationships are empty, skip saving to the database"
            )
            return None

        with self._db.session():
            # Create or find entities
            entity_map = {}
            for entity in knowledge_graph.entities:
                created_entity = self._graph_store.find_or_create_entity(
                    entity_type=EntityType.original,
                    name=entity.name,
                    description=entity.description,
                    meta=entity.meta,
                )
                entity_map[entity.name] = created_entity

            # Create relationships
            relationships = []
            for rel in knowledge_graph.relationships:
                logger.info("Saving relationship: %s", rel.description)
                source_entity = entity_map.get(rel.source_entity_name)
                if not source_entity:
                    logger.warning(
                        "Source entity not found for relationship: %s", str(rel)
                    )
                    continue

                target_entity = entity_map.get(rel.target_entity_name)
                if not target_entity:
                    logger.warning(
                        "Target entity not found for relationship: %s", str(rel)
                    )
                    continue

                relationship = self._graph_store.create_relationship(
                    source_entity=source_entity,
                    target_entity=target_entity,
                    description=rel.description,
                    meta=rel.meta,
                )
                relationships.append(relationship)

        return KnowledgeGraph(
            entities=list(entity_map.values()), relationships=relationships
        )

    def retrieve(
        self,
        query: str,
        depth: int = 2,
        metadata_filters: Optional[dict] = None,
        **kwargs,
    ) -> RetrievedKnowledgeGraph:
        retriever = WeightedGraphRetriever(self._graph_store, **kwargs)
        return retriever.retrieve(
            query=query,
            depth=depth,
            metadata_filters=metadata_filters,
        )
