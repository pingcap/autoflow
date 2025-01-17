from hashlib import sha256

from llama_index.core.schema import BaseNode, MetadataMode
from pydantic import Field
from typing import Any, List
from app.models.relationship import Relationship
from app.rag.chat import get_prompt_by_jinja2_template
from app.rag.indexes.knowledge_graph.schema import Entity, KnowledgeGraph


DEFAULT_ENTITY_TEMPL = """
{% for entity in entities %}

- Name: {{ entity.name }}
- Description: {{ entity.description }}

{% endfor %}
"""

DEFAULT_RELATIONSHIP_TEMPL = """
{% for relationship in relationships %}

- Description: {{ relationship.rag_description }}
- Weight: {{ relationship.weight }}
- Last Modified At: {{ relationship.last_modified_at }}
- Meta: {{ relationship.meta | tojson(indent=2) }}

{% endfor %}
"""


class KnowledgeGraphNode(BaseNode):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

    query: str = Field(description="Query of the knowledge graph")
    entities: List[Entity] = Field(
        default_factory=list, description="The list of entities in the knowledge graph"
    )
    relationships: List[Relationship] = Field(
        default_factory=list,
        description="The list of relationships in the knowledge graph",
    )
    entity_template: str = Field(
        default=DEFAULT_ENTITY_TEMPL,
        description="The template to render the entity list as string",
    )
    relationship_template: str = Field(
        default=DEFAULT_RELATIONSHIP_TEMPL,
        description="The template to render the relationship list as string",
    )

    @classmethod
    def get_type(cls) -> str:
        return "KnowledgeGraphNode"

    def get_content(self, metadata_mode: MetadataMode = MetadataMode.ALL) -> str:
        return f"""
        Entities:
        ------
        {self._get_entities_str()}
        
        Relationships:
        ------
        {self._get_relationships_str()}
        """

    def _get_entities_str(self):
        return get_prompt_by_jinja2_template(
            self.entity_template, entities=self.entities
        )

    def _get_relationships_str(self):
        return get_prompt_by_jinja2_template(
            self.relationship_template, relationships=self.relationships
        )

    def set_content(self, value: KnowledgeGraph) -> None:
        self.entities = value.entities
        self.relationships = value.relationships

    @property
    def hash(self) -> str:
        knowledge_graph_identity = (
            self._get_entities_str() + self._get_relationships_str()
        )
        return str(sha256(knowledge_graph_identity).hexdigest())
