"""
Tree-sitter Code Extractor.

High-fidelity code parsing into IFCodeArtifact.
Extracts symbols, imports, exports using tree-sitter AST parsing.

NASA JPL Power of Ten Rules:
- Rule #2: Bounded recursion depth (MAX_AST_DEPTH)
- Rule #4: AST traversal loop < 60 lines
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# JPL RULE #2: FIXED UPPER BOUNDS
# =============================================================================

MAX_AST_DEPTH = 50
MAX_SYMBOLS = 5000
MAX_IMPORTS = 500
MAX_FILE_SIZE_KB = 1024

# Language configuration
LANGUAGE_MAP: Dict[str, str] = {
    ".py": "python",
    ".pyw": "python",
    ".js": "javascript",
    ".mjs": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
}


# =============================================================================
# EXTRACTION RESULT TYPES
# =============================================================================


@dataclass
class ExtractedSymbol:
    """A symbol extracted from source code."""

    name: str
    kind: str
    line_start: int
    line_end: int
    column_start: int
    column_end: int
    signature: Optional[str] = None
    docstring: Optional[str] = None
    parent: Optional[str] = None
    visibility: str = "public"
    is_async: bool = False
    decorators: List[str] = None

    def __post_init__(self) -> None:
        if self.decorators is None:
            self.decorators = []


@dataclass
class ExtractedImport:
    """An import statement extracted from source code."""

    module: str
    names: List[str]
    alias: Optional[str] = None
    is_relative: bool = False
    line_number: int = 0


@dataclass
class ExtractedExport:
    """An export extracted from source code."""

    name: str
    kind: str
    is_default: bool = False


@dataclass
class ExtractionResult:
    """Result of code extraction."""

    symbols: List[ExtractedSymbol]
    imports: List[ExtractedImport]
    exports: List[ExtractedExport]
    language: str
    line_count: int
    success: bool
    error: str = ""


# =============================================================================
# TREE-SITTER EXTRACTOR
# =============================================================================


class TreeSitterExtractor:
    """
    Tree-sitter based code symbol extractor.

    Integration with tree-sitter for Python and TypeScript.
    Rule #4: AST traversal loop < 60 lines.
    """

    # S-expression queries for symbol extraction
    SYMBOL_QUERIES: Dict[str, str] = {
        "python": """
            (class_definition
                name: (identifier) @class.name
                body: (block) @class.body) @class.def
            (function_definition
                name: (identifier) @func.name
                parameters: (parameters) @func.params) @func.def
            (decorated_definition
                (decorator) @decorator
                definition: (_) @decorated) @decorated.def
        """,
        "typescript": """
            (class_declaration
                name: (type_identifier) @class.name) @class.def
            (interface_declaration
                name: (type_identifier) @interface.name) @interface.def
            (function_declaration
                name: (identifier) @func.name) @func.def
            (method_definition
                name: (property_identifier) @method.name) @method.def
            (arrow_function) @arrow.def
            (export_statement) @export.stmt
        """,
    }

    IMPORT_QUERIES: Dict[str, str] = {
        "python": """
            (import_statement
                name: (dotted_name) @import.module) @import.stmt
            (import_from_statement
                module_name: (dotted_name) @import.module
                name: (dotted_name) @import.name) @import.from
        """,
        "typescript": """
            (import_statement
                source: (string) @import.source) @import.stmt
        """,
    }

    def __init__(self) -> None:
        """Initialize extractor."""
        self._parsers: Dict[str, Any] = {}
        self._languages: Dict[str, Any] = {}

    def is_available(self) -> bool:
        """Check if tree-sitter is available."""
        try:
            from tree_sitter_languages import get_parser  # noqa: F401

            return True
        except ImportError:
            return False

    def extract(self, code: str, language: str) -> ExtractionResult:
        """
        Extract symbols, imports, exports from source code.

        AC: Supports Python and TypeScript.
        Rule #4: Delegates to focused helper methods.
        Rule #7: Returns error result on failure.

        Args:
            code: Source code to parse.
            language: Programming language name.

        Returns:
            ExtractionResult with symbols, imports, exports.
        """
        if not self.is_available():
            return ExtractionResult(
                symbols=[],
                imports=[],
                exports=[],
                language=language,
                line_count=len(code.splitlines()),
                success=False,
                error="tree-sitter-languages not installed",
            )

        if language not in self.SYMBOL_QUERIES:
            return ExtractionResult(
                symbols=[],
                imports=[],
                exports=[],
                language=language,
                line_count=len(code.splitlines()),
                success=False,
                error=f"Unsupported language: {language}",
            )

        try:
            tree = self._parse(code, language)
            if tree is None:
                return ExtractionResult(
                    symbols=[],
                    imports=[],
                    exports=[],
                    language=language,
                    line_count=len(code.splitlines()),
                    success=False,
                    error="Parsing failed",
                )

            symbols = self._extract_symbols(code, tree, language)
            imports = self._extract_imports(code, tree, language)
            exports = self._extract_exports(code, tree, language)

            return ExtractionResult(
                symbols=symbols[:MAX_SYMBOLS],
                imports=imports[:MAX_IMPORTS],
                exports=exports,
                language=language,
                line_count=len(code.splitlines()),
                success=True,
            )
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return ExtractionResult(
                symbols=[],
                imports=[],
                exports=[],
                language=language,
                line_count=len(code.splitlines()),
                success=False,
                error=str(e),
            )

    def _parse(self, code: str, language: str) -> Optional[Any]:
        """
        Parse code into AST.

        Rule #7: Returns None on error.
        """
        try:
            from tree_sitter_languages import get_parser, get_language

            if language not in self._parsers:
                self._parsers[language] = get_parser(language)
                self._languages[language] = get_language(language)

            parser = self._parsers[language]
            return parser.parse(bytes(code, "utf-8"))
        except Exception as e:
            logger.error(f"Parse error for {language}: {e}")
            return None

    def _extract_symbols(
        self, code: str, tree: Any, language: str
    ) -> List[ExtractedSymbol]:
        """
        Extract symbols from AST.

        AC / JPL Rule #4: Loop < 60 lines.
        """
        symbols: List[ExtractedSymbol] = []
        query_str = self.SYMBOL_QUERIES.get(language, "")
        if not query_str:
            return symbols

        try:
            lang = self._languages[language]
            query = lang.query(query_str)
            captures = query.captures(tree.root_node)

            current_def = None
            current_name = "anonymous"
            decorators: List[str] = []

            for node, tag in captures:
                if tag == "decorator":
                    decorators.append(node.text.decode("utf-8"))
                    continue

                if tag.endswith(".def"):
                    kind = self._tag_to_kind(tag)
                    current_def = node
                    # Reset decorators after applying
                    applied_decorators = decorators[:]
                    decorators = []

                    symbols.append(
                        ExtractedSymbol(
                            name=current_name,
                            kind=kind,
                            line_start=node.start_point[0] + 1,
                            line_end=node.end_point[0] + 1,
                            column_start=node.start_point[1],
                            column_end=node.end_point[1],
                            decorators=applied_decorators,
                        )
                    )

                elif tag.endswith(".name") and symbols:
                    # Update the last symbol's name
                    symbols[-1].name = node.text.decode("utf-8")

                elif tag.endswith(".params") and symbols:
                    # Extract signature
                    symbols[-1].signature = node.text.decode("utf-8")

                if len(symbols) >= MAX_SYMBOLS:
                    break

            return symbols
        except Exception as e:
            logger.error(f"Symbol extraction failed: {e}")
            return symbols

    def _extract_imports(
        self, code: str, tree: Any, language: str
    ) -> List[ExtractedImport]:
        """
        Extract import statements.

        Rule #4: < 60 lines.
        """
        imports: List[ExtractedImport] = []
        query_str = self.IMPORT_QUERIES.get(language, "")
        if not query_str:
            return imports

        try:
            lang = self._languages[language]
            query = lang.query(query_str)
            captures = query.captures(tree.root_node)

            current_module = ""
            current_names: List[str] = []

            for node, tag in captures:
                if tag == "import.module":
                    current_module = node.text.decode("utf-8")
                elif tag == "import.name":
                    current_names.append(node.text.decode("utf-8"))
                elif tag == "import.source":
                    # TypeScript/JavaScript string import
                    current_module = node.text.decode("utf-8").strip("'\"")
                elif tag.endswith(".stmt") or tag.endswith(".from"):
                    if current_module:
                        imports.append(
                            ExtractedImport(
                                module=current_module,
                                names=current_names[:],
                                line_number=node.start_point[0] + 1,
                                is_relative=current_module.startswith("."),
                            )
                        )
                    current_module = ""
                    current_names = []

                if len(imports) >= MAX_IMPORTS:
                    break

            return imports
        except Exception as e:
            logger.error(f"Import extraction failed: {e}")
            return imports

    def _extract_exports(
        self, code: str, tree: Any, language: str
    ) -> List[ExtractedExport]:
        """
        Extract export statements (TypeScript/JavaScript).

        Rule #4: < 60 lines.
        """
        exports: List[ExtractedExport] = []

        if language not in ("typescript", "javascript", "tsx"):
            # Python uses module-level visibility
            return exports

        try:
            # Simple heuristic: find export keywords
            for i, line in enumerate(code.splitlines()):
                line_stripped = line.strip()
                if line_stripped.startswith("export "):
                    if "default" in line_stripped:
                        exports.append(
                            ExtractedExport(
                                name="default",
                                kind="default",
                                is_default=True,
                            )
                        )
                    elif "class " in line_stripped:
                        name = self._extract_name_after(line_stripped, "class ")
                        exports.append(ExtractedExport(name=name, kind="class"))
                    elif "function " in line_stripped:
                        name = self._extract_name_after(line_stripped, "function ")
                        exports.append(ExtractedExport(name=name, kind="function"))
                    elif "const " in line_stripped or "let " in line_stripped:
                        name = self._extract_name_after(
                            line_stripped,
                            "const " if "const " in line_stripped else "let ",
                        )
                        exports.append(ExtractedExport(name=name, kind="variable"))

            return exports
        except Exception as e:
            logger.error(f"Export extraction failed: {e}")
            return exports

    def _tag_to_kind(self, tag: str) -> str:
        """Convert capture tag to symbol kind."""
        if tag.startswith("class"):
            return "class"
        if tag.startswith("interface"):
            return "interface"
        if tag.startswith("method"):
            return "method"
        if tag.startswith("func") or tag.startswith("arrow"):
            return "function"
        if tag.startswith("decorated"):
            return "function"
        return "unknown"

    def _extract_name_after(self, line: str, keyword: str) -> str:
        """Extract identifier after keyword."""
        idx = line.find(keyword)
        if idx == -1:
            return "anonymous"
        rest = line[idx + len(keyword) :].strip()
        # Take first word
        name = rest.split()[0] if rest.split() else "anonymous"
        # Remove trailing punctuation
        return name.rstrip("({<:")


# =============================================================================
# ARTIFACT CONVERSION
# =============================================================================


def extract_to_artifact(
    file_path: Path,
    content: Optional[str] = None,
) -> "IFCodeArtifact":
    """
    Extract code and create IFCodeArtifact.

    High-fidelity code parsing into Knowledge Graph.
    Rule #7: Returns artifact with error metadata on failure.

    Args:
        file_path: Path to source file.
        content: Optional pre-loaded content.

    Returns:
        IFCodeArtifact with extracted symbols.
    """
    from ingestforge.core.pipeline.artifacts import (
        IFCodeArtifact,
        CodeSymbol,
        ImportInfo,
        ExportInfo,
    )

    # Detect language
    suffix = file_path.suffix.lower()
    language = LANGUAGE_MAP.get(suffix, "unknown")

    # Load content if not provided
    if content is None:
        if not file_path.exists():
            return IFCodeArtifact(
                file_path=str(file_path),
                language=language,
                content="",
                metadata={"error": "File not found"},
            )
        try:
            content = file_path.read_text(encoding="utf-8")
        except Exception as e:
            return IFCodeArtifact(
                file_path=str(file_path),
                language=language,
                content="",
                metadata={"error": str(e)},
            )

    # Check file size
    if len(content) > MAX_FILE_SIZE_KB * 1024:
        return IFCodeArtifact(
            file_path=str(file_path),
            language=language,
            content=content[:1000] + "\n... (truncated)",
            metadata={"error": f"File exceeds {MAX_FILE_SIZE_KB}KB limit"},
        )

    # Extract with tree-sitter
    extractor = TreeSitterExtractor()
    result = extractor.extract(content, language)

    # Convert to IFCodeArtifact types
    symbols = [
        CodeSymbol(
            name=s.name,
            kind=_map_kind(s.kind),
            line_start=s.line_start,
            line_end=s.line_end,
            column_start=s.column_start,
            column_end=s.column_end,
            signature=s.signature,
            docstring=s.docstring,
            parent_symbol=s.parent,
            visibility=s.visibility,
            is_async=s.is_async,
            decorators=s.decorators or [],
        )
        for s in result.symbols
    ]

    imports = [
        ImportInfo(
            module=i.module,
            names=i.names,
            alias=i.alias,
            is_relative=i.is_relative,
            line_number=i.line_number,
        )
        for i in result.imports
    ]

    exports = [
        ExportInfo(
            name=e.name,
            kind=_map_kind(e.kind),
            is_default=e.is_default,
        )
        for e in result.exports
    ]

    import uuid

    return IFCodeArtifact(
        artifact_id=str(uuid.uuid4()),
        file_path=str(file_path),
        language=language,
        content=content,
        imports=imports,
        exports=exports,
        symbols=symbols,
        line_count=result.line_count,
        metadata={
            "extraction_success": result.success,
            "extraction_error": result.error if not result.success else None,
        },
    )


def _map_kind(kind_str: str) -> "SymbolKind":
    """Map string kind to SymbolKind enum."""
    from ingestforge.core.pipeline.artifacts import SymbolKind

    kind_map = {
        "class": SymbolKind.CLASS,
        "function": SymbolKind.FUNCTION,
        "method": SymbolKind.METHOD,
        "interface": SymbolKind.INTERFACE,
        "variable": SymbolKind.VARIABLE,
        "constant": SymbolKind.CONSTANT,
        "enum": SymbolKind.ENUM,
        "module": SymbolKind.MODULE,
        "property": SymbolKind.PROPERTY,
        "default": SymbolKind.MODULE,
        "unknown": SymbolKind.VARIABLE,
    }
    return kind_map.get(kind_str, SymbolKind.VARIABLE)
