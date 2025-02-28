import re
from typing import Any, Dict, List, Optional, Sequence, Callable

from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.node_parser.interface import NodeParser
from llama_index.core.node_parser.node_utils import build_nodes_from_splits
from llama_index.core.schema import BaseNode, MetadataMode, TextNode
from llama_index.core.utils import get_tqdm_iterable, get_tokenizer
from llama_index.core.bridge.pydantic import Field, PrivateAttr


DEFAULT_CHUNK_HEADER_LEVEL = 2
DEFAULT_CHUNK_SIZE = 1200


class MarkdownNodeParser(NodeParser):
    """Markdown node parser.
    Splits a document into Nodes using custom Markdown splitting logic.
    Args:
        include_metadata (bool): whether to include metadata in nodes
        include_prev_next_rel (bool): whether to include prev/next relationships
    """

    chunk_size: int = Field(
        default=DEFAULT_CHUNK_SIZE,
        description="The token chunk size for each chunk.",
        gt=0,
    )
    chunk_header_level: int = Field(
        default=DEFAULT_CHUNK_HEADER_LEVEL,
        description="The header level to split on",
        ge=1,
        le=6,
    )
    _tokenizer: Callable = PrivateAttr()

    def __init__(
        self,
        chunk_header_level: int = DEFAULT_CHUNK_HEADER_LEVEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        tokenizer: Optional[Callable] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ):
        super().__init__(
            chunk_header_level=chunk_header_level,
            chunk_size=chunk_size,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )
        self.callback_manager = callback_manager or CallbackManager([])
        self._tokenizer = tokenizer or get_tokenizer()

    @classmethod
    def from_defaults(
        cls,
        chunk_header_level: int = DEFAULT_CHUNK_HEADER_LEVEL,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        tokenizer: Optional[Callable] = None,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
        callback_manager: Optional[CallbackManager] = None,
    ) -> "MarkdownNodeParser":
        callback_manager = callback_manager or CallbackManager([])
        tokenizer = tokenizer or get_tokenizer()
        return cls(
            chunk_header_level=chunk_header_level,
            chunk_size=chunk_size,
            tokenizer=tokenizer,
            include_metadata=include_metadata,
            include_prev_next_rel=include_prev_next_rel,
            callback_manager=callback_manager,
        )

    @classmethod
    def class_name(cls) -> str:
        """Get class name."""
        return "MarkdownNodeParser"

    def _parse_nodes(
        self,
        nodes: Sequence[BaseNode],
        show_progress: bool = False,
        **kwargs: Any,
    ) -> List[BaseNode]:
        all_nodes: List[BaseNode] = []
        nodes_with_progress = get_tqdm_iterable(nodes, show_progress, "Parsing nodes")

        for node in nodes_with_progress:
            splitted_nodes = self.get_nodes_from_node(
                node,
                self.chunk_header_level,
                self.chunk_size * 0.7,
                self.chunk_size * 1.1,
            )
            nodes = []
            for sn in splitted_nodes:
                header_level = sn.metadata.get("Header_Level")
                if header_level:
                    for _hl in range(1, header_level + 1)[::-1]:
                        if (
                            f"Header_{_hl}" in sn.metadata
                            and sn.metadata[f"Header_{_hl}"] not in sn.text
                        ):
                            sn.text = (
                                f'{"#" * _hl} {sn.metadata[f"Header_{_hl}"]}\n\n'
                                + sn.text
                            )
                n = build_nodes_from_splits([sn.text], node, id_func=self.id_func)[0]
                if self.include_metadata:
                    n.metadata = {**node.metadata, **sn.metadata}
                nodes.append(n)
            all_nodes.extend(nodes)

        return all_nodes

    def get_nodes_from_node(
        self,
        node: BaseNode,
        chunk_header_level: int,
        chunk_size_small_threshold: float,
        chunk_size_large_threshold: float,
    ) -> List[TextNode]:
        # print(chunk_header_level, chunk_size_small_threshold, chunk_size_large_threshold)
        """Get nodes from document."""
        text = node.get_content(metadata_mode=MetadataMode.NONE)
        markdown_nodes = []
        lines = text.split("\n")
        metadata: Dict[str, str] = node.metadata
        code_block = False
        current_section = ""
        first_header = True

        for line in lines:
            if line.lstrip().startswith("```"):
                code_block = not code_block
            header_match = re.match(r"^(#+)\s(.*)", line)
            if header_match and not code_block:
                current_header_level = len(header_match.group(1).strip())
                if current_section != "" and current_header_level == chunk_header_level:
                    if first_header:
                        # skip the first header, merge it with the first section (usually the title of the document)
                        first_header = False
                    else:
                        markdown_nodes.append(
                            self._build_node_from_split(
                                current_section.strip(), node, metadata
                            )
                        )
                        current_section = ""
                if current_header_level <= chunk_header_level:
                    metadata = self._update_metadata(
                        metadata, header_match.group(2), current_header_level
                    )
                current_section += line + "\n"
            else:
                current_section += line + "\n"

        markdown_nodes.append(
            self._build_node_from_split(current_section.strip(), node, metadata)
        )
        return self._normalize_node_sizes(
            markdown_nodes, chunk_size_small_threshold, chunk_size_large_threshold
        )

    def _normalize_node_sizes(
        self,
        nodes: List[TextNode],
        chunk_size_small_threshold: float,
        chunk_size_large_threshold: float,
    ) -> List[TextNode]:
        # 1. Split the big node into multiple small nodes
        # 2. Merge the small nodes into a big node if they are too small
        # 3. Make all the nodes as much as possible close to the chunk size
        nodes_token_size = [self._token_size(node.text) for node in nodes]
        normalized_nodes = []
        buffer = []
        node_count = len(nodes)
        i = 0

        while i < node_count:
            node = nodes[i]
            this_chunk_size = nodes_token_size[i]
            if this_chunk_size < chunk_size_small_threshold:
                # if the last node is too small, merge it with the previous one
                if (
                    not buffer
                    and i == (node_count - 1)
                    and i > 0
                    and nodes_token_size[i - 1] + this_chunk_size
                    < chunk_size_large_threshold
                ):
                    normalized_nodes[-1].text += "\n\n" + node.text
                    i += 1
                    continue
                buffer.append(this_chunk_size)
                total = sum(buffer)
                while (
                    (i + 1) < node_count
                    and nodes_token_size[i + 1] < self.chunk_size
                    and total + nodes_token_size[i + 1] <= chunk_size_large_threshold
                ):
                    i += 1
                    buffer.append(nodes_token_size[i])
                    total += nodes_token_size[i]
                # output the sum of the buffer
                buffer_nodes = nodes[i - len(buffer) + 1 : i + 1]
                normalized_nodes.append(
                    TextNode(
                        text="\n\n".join([node.text for node in buffer_nodes]),
                        metadata=buffer_nodes[0].metadata,
                    )
                )
                i += 1
                buffer.clear()
            elif this_chunk_size > chunk_size_large_threshold:
                # split into multiple nodes with next header level and bigger chunk size
                md_splitted_nodes = self.get_nodes_from_node(
                    node,
                    self.chunk_header_level + 1,
                    chunk_size_small_threshold,
                    chunk_size_large_threshold * 1.1,
                )
                for n in md_splitted_nodes:
                    _chunk_size = self._token_size(n.text)
                    if _chunk_size > chunk_size_large_threshold * 1.1:
                        # using sentence splitter to split the node if it's still too large
                        sentence_splitted_nodes = SentenceSplitter(
                            chunk_size=int(chunk_size_large_threshold), separator="\n\n"
                        ).get_nodes_from_documents([n])
                        normalized_nodes.extend(sentence_splitted_nodes)
                    else:
                        normalized_nodes.append(n)
                i += 1
            else:
                normalized_nodes.append(node)
                i += 1
        return normalized_nodes

    def _update_metadata(
        self, headers_metadata: dict, new_header: str, new_header_level: int
    ) -> dict:
        """Update the markdown headers for metadata.
        Removes all headers that are equal or less than the level
        of the newly found header
        """
        updated_headers = {}

        for i in range(1, new_header_level):
            key = f"Header_{i}"
            if key in headers_metadata:
                updated_headers[key] = headers_metadata[key]

        updated_headers[f"Header_{new_header_level}"] = new_header
        updated_headers["Header_Level"] = new_header_level
        return updated_headers

    def _build_node_from_split(
        self,
        text_split: str,
        node: BaseNode,
        metadata: dict,
    ) -> TextNode:
        """Build node from single text split."""
        node = build_nodes_from_splits([text_split], node, id_func=self.id_func)[0]

        if self.include_metadata:
            node.metadata = {**node.metadata, **metadata}

        return node

    def _token_size(self, text: str) -> int:
        return len(self._tokenizer(text))
