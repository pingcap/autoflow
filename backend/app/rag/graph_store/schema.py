from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from sqlmodel import Session


class KnowledgeGraphStore(ABC):
    @abstractmethod
    def save(self, entities_df, relationships_df) -> None:
        """Upsert entities and relationships to the graph store."""
        pass

    @abstractmethod
    def retrieve_with_weight(
        self,
        query: str,
        embedding: list,
        depth: int = 2,
        include_meta: bool = False,
        with_degree: bool = False,
        with_chunks: bool = True,
        relationship_meta_filters: Dict = {},
        session: Optional[Session] = None,
    ) -> Tuple[list, list, list]:
        """Retrieve nodes and relationships with weights."""
        pass
