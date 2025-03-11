from dotenv import load_dotenv
from sqlmodel import Session

from app.core.db import engine
from app.models import (
    Chunk as DBChunk,
    Entity as DBEntity,
    Relationship as DBRelationship,
)
from app.models.knowledge_base import KnowledgeBase
from app.rag.indices.knowledge_graph.graph_store.helpers import get_default_embed_model
from app.rag.indices.knowledge_graph.graph_store.tidb_graph_store import TiDBGraphStore
from app.rag.indices.knowledge_graph.schema import (
    EntityCreate,
    EntityDegree,
)

load_dotenv()


class TestGraphStore:
    @classmethod
    def setup_class(cls):
        """Set up test fixtures before running any tests in the class"""
        cls.db_session = Session(engine)
        # Create a test knowledge base
        cls.kb = KnowledgeBase(name="test_kb")
        cls.db_session.add(cls.kb)
        cls.db_session.commit()

        cls.graph_store = TiDBGraphStore(
            db_session=cls.db_session,
            knowledge_base=cls.kb,
            embed_model=get_default_embed_model(),
            entity_model=DBEntity,
            relationship_model=DBRelationship,
            chunk_model=DBChunk,
        )

    @classmethod
    def teardown_class(cls):
        """Clean up after all tests in the class have run"""
        # Clean up the test data
        cls.db_session.delete(cls.kb)
        cls.db_session.commit()
        cls.db_session.close()

    def test_calc_entity_degrees(self):
        tidb_entity = self.graph_store.create_entity(
            EntityCreate(
                name="TiDB",
            )
        )
        tikv_entity = self.graph_store.create_entity(
            EntityCreate(
                name="TiKV",
            )
        )
        ticdc_entity = self.graph_store.create_entity(
            EntityCreate(
                name="TiCDC",
            )
        )
        self.graph_store.create_relationship(
            source_entity=tidb_entity,
            target_entity=tikv_entity,
            description="TiDB has a component named TiKV",
        )
        self.graph_store.create_relationship(
            source_entity=tidb_entity,
            target_entity=ticdc_entity,
            description="TiDB has a tool named TiCDC",
        )

        out_degree = self.graph_store.calc_entity_out_degree(tidb_entity.id)
        assert out_degree == 2

        in_degree = self.graph_store.calc_entity_in_degree(tikv_entity.id)
        assert in_degree == 1

        degrees = self.graph_store.calc_entities_degrees(
            [tidb_entity.id, tikv_entity.id, ticdc_entity.id]
        )
        print(degrees)
        assert degrees == [
            EntityDegree(
                entity_id=tidb_entity.id, in_degree=0, out_degree=2, degrees=2
            ),
            EntityDegree(
                entity_id=tikv_entity.id, in_degree=1, out_degree=0, degrees=1
            ),
            EntityDegree(
                entity_id=ticdc_entity.id, in_degree=1, out_degree=0, degrees=1
            ),
        ]

    def test_another_method(self):
        pass
