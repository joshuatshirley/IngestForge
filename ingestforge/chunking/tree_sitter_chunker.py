"""
Tree-sitter based code chunking.

Provides high-fidelity code chunking using Tree-sitter for AST parsing,
supporting multiple languages with precise boundary detection.
"""

from typing import List, Optional, Tuple
from pathlib import Path
import logging

from tree_sitter_languages import get_language, get_parser
from tree_sitter import Node

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.core.config import Config
from ingestforge.chunking.code_chunker import CodeUnit

logger = logging.getLogger(__name__)


class TreeSitterCodeChunker:
    """
    Advanced code chunker using Tree-sitter.

    Uses S-expression queries to find semantic units (functions, classes)
    across different programming languages.
    """

    LANGUAGE_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".go": "go",
        ".rs": "rust",
        ".java": "java",
        ".cpp": "cpp",
        ".c": "c",
    }

    # S-expression queries for finding definitions
    QUERIES = {
        "python": """
            (class_definition name: (identifier) @class.name) @class.def
            (function_definition name: (identifier) @func.name) @func.def
        """,
        "javascript": """
            (class_declaration name: (identifier) @class.name) @class.def
            (function_declaration name: (identifier) @func.name) @func.def
            (method_definition name: (property_identifier) @func.name) @func.def
            (variable_declarator name: (identifier) @func.name value: (arrow_function)) @func.def
        """,
        "typescript": """
            (class_declaration name: (type_identifier) @class.name) @class.def
            (interface_declaration name: (type_identifier) @class.name) @class.def
            (function_declaration name: (identifier) @func.name) @func.def
            (method_definition name: (property_identifier) @func.name) @func.def
        """,
        "go": """
            (type_declaration (type_spec name: (type_identifier) @class.name type: (struct_type))) @class.def
            (function_declaration name: (identifier) @func.name) @func.def
            (method_declaration name: (identifier) @func.name) @func.def
        """,
        "rust": """
            (struct_item name: (type_identifier) @class.name) @class.def
            (enum_item name: (type_identifier) @class.name) @class.def
            (trait_item name: (type_identifier) @class.name) @class.def
            (function_item name: (identifier) @func.name) @func.def
            (impl_item type: (type_identifier) @class.name) @class.def
        """,
    }

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.max_chunk_size = 4000

    def chunk(
        self,
        code: str,
        document_id: str,
        source_file: str = "",
    ) -> List[ChunkRecord]:
        """Chunk code using tree-sitter queries."""
        language_name = self._detect_language(source_file)
        if language_name not in self.QUERIES:
            # Fallback to module-level chunking if no query defined
            return [self._create_module_record(code, document_id, source_file)]

        try:
            language = get_language(language_name)
            parser = get_parser(language_name)
            tree = parser.parse(bytes(code, "utf8"))
            query = language.query(self.QUERIES[language_name])
            captures = query.captures(tree.root_node)
        except Exception as e:
            logger.error(f"Tree-sitter parsing failed for {source_file}: {e}")
            return [self._create_module_record(code, document_id, source_file)]

        units = self._process_captures(code, captures)

        # Add a header unit if there's significant content before the first unit
        if units and units[0].start_line > 1:
            header_content = "\n".join(code.splitlines()[: units[0].start_line - 1])
            if header_content.strip():
                units.insert(
                    0,
                    CodeUnit(
                        name="header",
                        kind="imports",
                        content=header_content,
                        start_line=1,
                        end_line=units[0].start_line - 1,
                    ),
                )

        return [
            self._create_record(unit, i, len(units), document_id, source_file)
            for i, unit in enumerate(units)
        ]

    def _detect_language(self, source_file: str) -> str:
        ext = Path(source_file).suffix.lower()
        return self.LANGUAGE_MAP.get(ext, "unknown")

    def _process_captures(
        self, code: str, captures: List[Tuple[Node, str]]
    ) -> List[CodeUnit]:
        units = []
        lines = code.splitlines()

        # We group captures by definition node
        # Query returns (node, tag)
        # e.g., (node_func, "func.def"), (node_id, "func.name")

        current_unit_node = None
        current_name = "unknown"
        current_kind = "unknown"

        for node, tag in captures:
            if tag.endswith(".def"):
                # If we were already tracking a unit, save it?
                # No, Tree-sitter might return nested captures.
                # Usually we want the top-level definitions.

                # Check if this node is already covered by a previous unit (nested)
                if any(
                    u.start_line <= node.start_point[0] + 1
                    and u.end_line >= node.end_point[0] + 1
                    for u in units
                ):
                    continue

                kind = "class" if tag.startswith("class") else "function"

                # Extract content
                start_byte = node.start_byte
                end_byte = node.end_byte
                content = code[start_byte:end_byte]

                # Find name in subsequent captures if available, or extract from node
                name = self._find_name_for_node(node, captures)

                units.append(
                    CodeUnit(
                        name=name,
                        kind=kind,
                        content=content,
                        start_line=node.start_point[0] + 1,
                        end_line=node.end_point[0] + 1,
                    )
                )

        return units

    def _find_name_for_node(
        self, def_node: Node, captures: List[Tuple[Node, str]]
    ) -> str:
        """Find the identifier associated with a definition node."""
        for node, tag in captures:
            if tag.endswith(".name") and node.parent == def_node:
                return node.text.decode("utf8")
        return "anonymous"

    def _create_record(
        self, unit: CodeUnit, index: int, total: int, doc_id: str, source: str
    ) -> ChunkRecord:
        import hashlib

        content_hash = hashlib.md5(unit.content.encode()).hexdigest()[:8]
        chunk_id = f"{doc_id}_ts_{index:04d}_{content_hash}"

        return ChunkRecord(
            chunk_id=chunk_id,
            document_id=doc_id,
            content=unit.content,
            chunk_type=f"code_{unit.kind}",
            section_title=f"{unit.kind} {unit.name}",
            source_file=source,
            chunk_index=index,
            total_chunks=total,
            char_count=len(unit.content),
        )

    def _create_module_record(self, code: str, doc_id: str, source: str) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=f"{doc_id}_module",
            document_id=doc_id,
            content=code,
            chunk_type="code_module",
            section_title="module",
            source_file=source,
            chunk_index=0,
            total_chunks=1,
            char_count=len(code),
        )
