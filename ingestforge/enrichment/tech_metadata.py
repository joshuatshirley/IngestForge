"""
Technical metadata extraction for code chunks.

Extracts programming language specific metadata like imports, exports,
and dependencies using Tree-sitter.
"""

import logging
from typing import List
from tree_sitter_languages import get_language, get_parser

from ingestforge.chunking.semantic_chunker import ChunkRecord

logger = logging.getLogger(__name__)


class TechMetadataRefiner:
    """
    Enriches code chunks with technical metadata.
    """

    IMPORT_QUERIES = {
        "python": """
            (import_statement) @import
            (import_from_statement) @import
        """,
        "javascript": """
            (import_statement) @import
        """,
        "typescript": """
            (import_statement) @import
        """,
        "go": """
            (import_declaration) @import
        """,
        "rust": """
            (use_declaration) @import
        """,
    }

    def enrich(self, chunk: ChunkRecord) -> ChunkRecord:
        """Enrich a chunk with technical metadata if it's a code chunk."""
        if not chunk.chunk_type.startswith("code_"):
            return chunk

        language_name = self._get_language_from_chunk(chunk)
        if not language_name or language_name == "unknown":
            return chunk

        imports = self._extract_imports(chunk.content, language_name)

        if chunk.metadata is None:
            chunk.metadata = {}

        chunk.metadata["language"] = language_name
        chunk.metadata["imports"] = imports

        return chunk

    def _get_language_from_chunk(self, chunk: ChunkRecord) -> str:
        # Check metadata first
        if chunk.metadata and "language" in chunk.metadata:
            return chunk.metadata["language"]

        # Infer from source file extension
        from pathlib import Path

        source_file = chunk.source_file
        if not source_file:
            return "unknown"

        ext = Path(source_file).suffix.lower()
        from ingestforge.chunking.tree_sitter_chunker import TreeSitterCodeChunker

        return TreeSitterCodeChunker.LANGUAGE_MAP.get(ext, "unknown")

    def _extract_imports(self, code: str, language_name: str) -> List[str]:
        if language_name not in self.IMPORT_QUERIES:
            return []

        try:
            language = get_language(language_name)
            parser = get_parser(language_name)
            tree = parser.parse(bytes(code, "utf8"))
            query = language.query(self.IMPORT_QUERIES[language_name])
            captures = query.captures(tree.root_node)

            return [node.text.decode("utf8").strip() for node, _ in captures]
        except Exception as e:
            logger.error(f"Failed to extract imports for {language_name}: {e}")
            return []
