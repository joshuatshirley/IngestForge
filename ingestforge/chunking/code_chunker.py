"""
AST-based code chunking.

Chunks code files by function/class definitions using AST parsing,
preserving semantic context for better code understanding.
"""

import ast
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from pathlib import Path

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.core.config import Config


@dataclass
class CodeUnit:
    """A semantic unit of code (function, class, method, etc.)."""

    name: str
    kind: str  # function, class, method, module
    content: str
    start_line: int
    end_line: int
    docstring: Optional[str] = None
    parent: Optional[str] = None  # Parent class/module name
    signature: Optional[str] = None
    decorators: List[str] = None

    def __post_init__(self) -> None:
        if self.decorators is None:
            self.decorators = []


# Alias for backwards compatibility
CodeChunk = CodeUnit


class CodeChunker:
    """
    AST-based code chunker.

    Parses code using AST and chunks by semantic units:
    - Classes (with methods grouped together)
    - Standalone functions
    - Module-level code
    - Import blocks

    Supports:
    - Python (via ast module)
    - JavaScript/TypeScript (via regex fallback)
    - Other languages (via regex patterns)
    """

    # Language detection by extension
    LANGUAGE_MAP = {
        ".py": "python",
        ".pyw": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".java": "java",
        ".kt": "kotlin",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".cpp": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
    }

    # Regex patterns for languages without AST support
    FUNCTION_PATTERNS = {
        "javascript": [
            # function name() {}
            r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\([^)]*\)",
            # const name = () => {}
            r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
            # const name = function() {}
            r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?function",
        ],
        "typescript": [
            r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*(?:<[^>]*>)?\s*\([^)]*\)",
            r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\([^)]*\)\s*=>",
        ],
        "java": [
            r"^\s*(?:public|private|protected)?\s*(?:static)?\s*\w+(?:<[^>]*>)?\s+(\w+)\s*\([^)]*\)",
        ],
        "go": [
            r"^func\s+(?:\([^)]+\)\s+)?(\w+)\s*\([^)]*\)",
        ],
        "rust": [
            r"^(?:pub\s+)?(?:async\s+)?fn\s+(\w+)",
        ],
    }

    CLASS_PATTERNS = {
        "javascript": [
            r"^(?:export\s+)?class\s+(\w+)",
        ],
        "typescript": [
            r"^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)",
            r"^(?:export\s+)?interface\s+(\w+)",
        ],
        "java": [
            r"^(?:public\s+)?(?:abstract\s+)?class\s+(\w+)",
            r"^(?:public\s+)?interface\s+(\w+)",
        ],
        "go": [
            r"^type\s+(\w+)\s+struct",
            r"^type\s+(\w+)\s+interface",
        ],
        "rust": [
            r"^(?:pub\s+)?struct\s+(\w+)",
            r"^(?:pub\s+)?enum\s+(\w+)",
            r"^(?:pub\s+)?trait\s+(\w+)",
            r"^impl(?:<[^>]*>)?\s+(?:\w+\s+for\s+)?(\w+)",
        ],
    }

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self.max_chunk_size = 3000  # Max chars per chunk
        self.include_imports = True
        self.include_docstrings = True

    def chunk(
        self,
        code: str,
        document_id: str,
        source_file: str = "",
        metadata: Optional[Dict] = None,
    ) -> List[ChunkRecord]:
        """
        Chunk code by AST structure.

        Rule #1: Reduced nesting from 4 → 2 levels
        Rule #4: Reduced from 67 → 28 lines via extraction

        Args:
            code: Source code text
            document_id: Parent document ID
            source_file: Source file path
            metadata: Additional metadata

        Returns:
            List of ChunkRecord objects with code units
        """
        if not code.strip():
            return []

        language = self._detect_language(source_file)
        units = (
            self._parse_python(code)
            if language == "python"
            else self._parse_generic(code, language)
        )
        return [
            self._create_code_chunk_record(
                unit, i, len(units), document_id, source_file
            )
            for i, unit in enumerate(units)
        ]

    def _build_section_title(self, unit: Any) -> str:
        """
        Build section title for code unit.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        if unit.kind == "class":
            return f"class {unit.name}"
        if unit.kind == "method":
            return f"{unit.parent}.{unit.name}()" if unit.parent else f"{unit.name}()"
        if unit.kind == "function":
            return f"{unit.name}()"
        return unit.name

    def _create_code_chunk_record(
        self, unit: Any, index: int, total: int, document_id: str, source_file: str
    ) -> ChunkRecord:
        """
        Create ChunkRecord from code unit.

        Rule #1: Extracted to reduce nesting (max 1 level)
        Rule #4: Function <60 lines

        Args:
            unit: Code unit to convert
            index: Chunk index
            total: Total chunk count
            document_id: Parent document ID
            source_file: Source file path

        Returns:
            ChunkRecord object
        """
        chunk_id = self._generate_chunk_id(document_id, index, unit.content)
        hierarchy = [unit.parent] if unit.parent else []
        section_title = self._build_section_title(unit)

        return ChunkRecord(
            chunk_id=chunk_id,
            document_id=document_id,
            content=unit.content,
            chunk_type=f"code_{unit.kind}",
            section_title=section_title,
            section_hierarchy=hierarchy,
            source_file=source_file,
            word_count=len(unit.content.split()),
            char_count=len(unit.content),
            chunk_index=index,
            total_chunks=total,
        )

    def _detect_language(self, source_file: str) -> str:
        """Detect programming language from file extension."""
        if not source_file:
            return "unknown"
        ext = Path(source_file).suffix.lower()
        return self.LANGUAGE_MAP.get(ext, "unknown")

    def _parse_python(self, code: str) -> List[CodeUnit]:
        """Parse Python code using the ast module."""
        units = []
        lines = code.split("\n")

        try:
            tree = ast.parse(code)
        except SyntaxError:
            # Fall back to regex parsing
            return self._parse_generic(code, "python")

        # Extract module-level docstring
        module_docstring = ast.get_docstring(tree)

        # Collect import statements
        imports = []
        import_end = 0
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                imports.append(ast.get_source_segment(code, node))
                import_end = max(import_end, node.end_lineno or 0)

        # Add imports as a unit if present
        if imports and self.include_imports:
            import_content = "\n".join(imports)
            units.append(
                CodeUnit(
                    name="imports",
                    kind="imports",
                    content=import_content,
                    start_line=1,
                    end_line=import_end,
                )
            )

        # Extract classes and functions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                units.extend(self._extract_class(node, code, lines))
            elif isinstance(node, ast.FunctionDef) or isinstance(
                node, ast.AsyncFunctionDef
            ):
                units.append(self._extract_function(node, code, lines))

        return units

    def _extract_class(
        self, node: ast.ClassDef, code: str, lines: List[str]
    ) -> List[CodeUnit]:
        """Extract a class and its methods."""
        units = []

        start = node.lineno - 1
        end = node.end_lineno or start + 1
        class_lines = lines[start:end]
        class_content = "\n".join(class_lines)

        docstring = ast.get_docstring(node)
        decorators = self._extract_decorators(node.decorator_list, code)

        if len(class_content) <= self.max_chunk_size:
            units.append(
                self._create_whole_class_unit(
                    node, class_content, start, end, docstring, decorators
                )
            )
        else:
            header = self._create_class_header(
                node, class_lines, start, docstring, decorators
            )
            if header:
                units.append(header)

            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    method_unit = self._extract_function(
                        child, code, lines, parent=node.name
                    )
                    units.append(method_unit)

        return units

    def _extract_decorators(self, decorator_list: List, code: str) -> List[str]:
        """Extract decorator strings from AST nodes.

        Rule #1: Reduced nesting via helper extraction
        """
        decorators = []
        for dec in decorator_list:
            decorator_str = self._format_decorator(dec, code)
            if decorator_str:
                decorators.append(decorator_str)
        return decorators

    def _format_decorator(self, dec: ast.AST, code: str) -> Optional[str]:
        """Format a single decorator based on its type.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        if isinstance(dec, ast.Name):
            return f"@{dec.id}"
        if isinstance(dec, ast.Attribute):
            return f"@{ast.get_source_segment(code, dec)}"
        if isinstance(dec, ast.Call):
            return f"@{ast.get_source_segment(code, dec)}"
        return None

    def _create_whole_class_unit(
        self,
        node: ast.ClassDef,
        content: str,
        start: int,
        end: int,
        docstring: Optional[str],
        decorators: List[str],
    ) -> CodeUnit:
        """Create CodeUnit for a complete class."""
        return CodeUnit(
            name=node.name,
            kind="class",
            content=content,
            start_line=start + 1,
            end_line=end,
            docstring=docstring,
            decorators=decorators,
        )

    def _create_class_header(
        self,
        node: ast.ClassDef,
        class_lines: List[str],
        start: int,
        docstring: Optional[str],
        decorators: List[str],
    ) -> Optional[CodeUnit]:
        """Create CodeUnit for class header (before first method)."""
        first_method_line = None
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                first_method_line = child.lineno
                break

        if not first_method_line:
            return None

        header_end = first_method_line - start - 1
        header_content = "\n".join(class_lines[:header_end])
        if not header_content.strip():
            return None

        return CodeUnit(
            name=f"{node.name} (header)",
            kind="class_header",
            content=header_content,
            start_line=start + 1,
            end_line=start + header_end,
            docstring=docstring,
            decorators=decorators,
        )

    def _extract_function(
        self,
        node: ast.FunctionDef,
        code: str,
        lines: List[str],
        parent: Optional[str] = None,
    ) -> CodeUnit:
        """Extract a function/method."""
        start = node.lineno - 1
        end = node.end_lineno or start + 1

        # Include decorators
        if node.decorator_list:
            first_dec = node.decorator_list[0]
            start = first_dec.lineno - 1

        func_lines = lines[start:end]
        func_content = "\n".join(func_lines)

        # Get docstring
        docstring = ast.get_docstring(node)

        # Get decorators
        decorators = []
        for dec in node.decorator_list:
            if isinstance(dec, ast.Name):
                decorators.append(f"@{dec.id}")
            elif isinstance(dec, ast.Attribute):
                decorators.append(f"@{ast.get_source_segment(code, dec)}")

        # Build signature
        args = []
        for arg in node.args.args:
            arg_str = arg.arg
            if arg.annotation:
                arg_str += f": {ast.get_source_segment(code, arg.annotation)}"
            args.append(arg_str)
        signature = f"{node.name}({', '.join(args)})"

        kind = "method" if parent else "function"

        return CodeUnit(
            name=node.name,
            kind=kind,
            content=func_content,
            start_line=start + 1,
            end_line=end,
            docstring=docstring,
            parent=parent,
            signature=signature,
            decorators=decorators,
        )

    def _parse_generic(self, code: str, language: str) -> List[CodeUnit]:
        """Parse code using regex patterns for non-Python languages."""
        lines = code.split("\n")
        func_patterns = self.FUNCTION_PATTERNS.get(language, [])
        class_patterns = self.CLASS_PATTERNS.get(language, [])

        if not func_patterns and not class_patterns:
            return [self._create_module_unit(code, lines)]

        func_regexes = [re.compile(p, re.MULTILINE) for p in func_patterns]
        class_regexes = [re.compile(p, re.MULTILINE) for p in class_patterns]

        definitions = self._find_code_definitions(lines, func_regexes, class_regexes)

        if not definitions:
            return [self._create_module_unit(code, lines)]

        units = self._extract_definition_units(lines, definitions)
        self._add_header_unit(units, lines, definitions[0]["line"])

        return units

    def _create_module_unit(self, code: str, lines: List[str]) -> CodeUnit:
        """Create a single module unit for unparseable code."""
        return CodeUnit(
            name="module",
            kind="module",
            content=code,
            start_line=1,
            end_line=len(lines),
        )

    def _find_code_definitions(
        self, lines: List[str], func_regexes: List, class_regexes: List
    ) -> List[Dict]:
        """Find all class and function definitions in code."""
        definitions = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check for class definitions
            for regex in class_regexes:
                match = regex.match(stripped)
                if match:
                    definitions.append(
                        {"name": match.group(1), "kind": "class", "line": i}
                    )
                    break

            # Check for function definitions
            for regex in func_regexes:
                match = regex.match(stripped)
                if match:
                    definitions.append(
                        {"name": match.group(1), "kind": "function", "line": i}
                    )
                    break

        return definitions

    def _extract_definition_units(
        self, lines: List[str], definitions: List[Dict]
    ) -> List[CodeUnit]:
        """Extract CodeUnits from found definitions."""
        units = []

        for i, defn in enumerate(definitions):
            start = defn["line"]
            end = definitions[i + 1]["line"] if i < len(definitions) - 1 else len(lines)
            content = "\n".join(lines[start:end]).rstrip()

            units.append(
                CodeUnit(
                    name=defn["name"],
                    kind=defn["kind"],
                    content=content,
                    start_line=start + 1,
                    end_line=end,
                )
            )

        return units

    def _add_header_unit(
        self, units: List[CodeUnit], lines: List[str], first_def_line: int
    ) -> None:
        """Add header unit if content exists before first definition."""
        if first_def_line > 0:
            header_content = "\n".join(lines[:first_def_line]).rstrip()
            if header_content.strip():
                units.insert(
                    0,
                    CodeUnit(
                        name="header",
                        kind="imports",
                        content=header_content,
                        start_line=1,
                        end_line=first_def_line,
                    ),
                )

    def _generate_chunk_id(self, document_id: str, index: int, content: str) -> str:
        """Generate unique chunk ID."""
        import hashlib

        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{document_id}_code_{index:04d}_{content_hash}"
