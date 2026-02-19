"""
Static Dependency Linker.

Automatically connect code files via import/export relationships.
Creates DEPENDS_ON edges in the Knowledge Manifest.

NASA JPL Power of Ten Rules:
- Rule #2: Bounded dependency counts
- Rule #4: Functions < 60 lines
- Rule #7: Check all return values
- Rule #9: Complete type hints
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional


from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# JPL RULE #2: FIXED UPPER BOUNDS
# =============================================================================

MAX_DEPENDENCIES_PER_FILE = 500
MAX_FILES_IN_GRAPH = 10000
MAX_RESOLUTION_DEPTH = 20

# Edge type constant
DEPENDS_ON = "DEPENDS_ON"
IMPORTS = "IMPORTS"
EXPORTS = "EXPORTS"


# =============================================================================
# DATA MODELS
# =============================================================================


@dataclass
class DependencyEdge:
    """
    An edge representing a dependency between code artifacts.

    AC: Creates DEPENDS_ON edges in the Knowledge Manifest.
    Rule #9: Complete type hints.
    """

    source_artifact_id: str
    target_artifact_id: str
    edge_type: str = DEPENDS_ON
    import_name: str = ""
    source_file: str = ""
    target_file: str = ""
    line_number: int = 0
    is_relative: bool = False
    confidence: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "source_artifact_id": self.source_artifact_id,
            "target_artifact_id": self.target_artifact_id,
            "edge_type": self.edge_type,
            "import_name": self.import_name,
            "source_file": self.source_file,
            "target_file": self.target_file,
            "line_number": self.line_number,
            "is_relative": self.is_relative,
            "confidence": self.confidence,
        }


@dataclass
class CodeFileEntry:
    """
    Entry for a code file in the dependency graph.

    Rule #9: Complete type hints.
    """

    artifact_id: str
    file_path: str
    module_path: str  # Normalized module path (e.g., "mypackage.mymodule")
    language: str
    imports: List[str] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)


@dataclass
class DependencyGraph:
    """
    Graph of code dependencies.

    AC: Shows links between related files.
    Rule #2: Bounded file count.
    Rule #9: Complete type hints.
    """

    files: Dict[str, CodeFileEntry] = field(default_factory=dict)
    edges: List[DependencyEdge] = field(default_factory=list)
    module_to_artifact: Dict[str, str] = field(default_factory=dict)

    @property
    def file_count(self) -> int:
        """Number of files in the graph."""
        return len(self.files)

    @property
    def edge_count(self) -> int:
        """Number of dependency edges."""
        return len(self.edges)

    def get_dependencies(self, artifact_id: str) -> List[DependencyEdge]:
        """Get all dependencies for an artifact."""
        return [e for e in self.edges if e.source_artifact_id == artifact_id]

    def get_dependents(self, artifact_id: str) -> List[DependencyEdge]:
        """Get all artifacts that depend on this one."""
        return [e for e in self.edges if e.target_artifact_id == artifact_id]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "file_count": self.file_count,
            "edge_count": self.edge_count,
            "files": {k: v.__dict__ for k, v in self.files.items()},
            "edges": [e.to_dict() for e in self.edges],
        }


# =============================================================================
# DEPENDENCY LINKER
# =============================================================================


class StaticDependencyLinker:
    """
    Links code files via their import/export relationships.

    Automatically connect code files via import/export.
    Rule #4: Functions < 60 lines.
    """

    def __init__(self, base_path: Optional[Path] = None) -> None:
        """
        Initialize linker.

        Args:
            base_path: Base path for resolving relative imports.
        """
        self._base_path = base_path or Path.cwd()
        self._graph = DependencyGraph()

    @property
    def graph(self) -> DependencyGraph:
        """Get the current dependency graph."""
        return self._graph

    def register_file(
        self,
        artifact_id: str,
        file_path: str,
        language: str,
        imports: List[str],
        exports: List[str],
    ) -> bool:
        """
        Register a code file in the dependency graph.

        Rule #2: Bounded file count.
        Rule #7: Returns False if limit reached.

        Args:
            artifact_id: Unique artifact identifier.
            file_path: Path to the source file.
            language: Programming language.
            imports: List of import module names.
            exports: List of exported symbol names.

        Returns:
            True if registered, False if limit reached.
        """
        if len(self._graph.files) >= MAX_FILES_IN_GRAPH:
            logger.warning(f"File limit reached ({MAX_FILES_IN_GRAPH})")
            return False

        module_path = self._file_to_module(file_path, language)

        entry = CodeFileEntry(
            artifact_id=artifact_id,
            file_path=file_path,
            module_path=module_path,
            language=language,
            imports=imports[:MAX_DEPENDENCIES_PER_FILE],
            exports=exports,
        )

        self._graph.files[artifact_id] = entry
        self._graph.module_to_artifact[module_path] = artifact_id

        return True

    def register_artifact(self, artifact: "IFCodeArtifact") -> bool:
        """
        Register an IFCodeArtifact in the dependency graph.

        Rule #7: Returns False on failure.

        Args:
            artifact: Code artifact to register.

        Returns:
            True if registered successfully.
        """
        imports = [imp.module for imp in artifact.imports]
        exports = [exp.name for exp in artifact.exports]

        return self.register_file(
            artifact_id=artifact.artifact_id,
            file_path=artifact.file_path or "",
            language=artifact.language,
            imports=imports,
            exports=exports,
        )

    def resolve_dependencies(self) -> List[DependencyEdge]:
        """
        Resolve all import relationships to create dependency edges.

        AC: Resolves relative imports to absolute artifact IDs.
        AC: Creates DEPENDS_ON edges in the Knowledge Manifest.
        Rule #4: < 60 lines.

        Returns:
            List of resolved dependency edges.
        """
        edges: List[DependencyEdge] = []

        for artifact_id, entry in self._graph.files.items():
            for import_name in entry.imports:
                # Resolve import to target artifact
                target_id = self._resolve_import(
                    import_name,
                    entry.file_path,
                    entry.language,
                )

                if target_id and target_id != artifact_id:
                    target_entry = self._graph.files.get(target_id)
                    target_file = target_entry.file_path if target_entry else ""

                    edge = DependencyEdge(
                        source_artifact_id=artifact_id,
                        target_artifact_id=target_id,
                        edge_type=DEPENDS_ON,
                        import_name=import_name,
                        source_file=entry.file_path,
                        target_file=target_file,
                        is_relative=import_name.startswith("."),
                    )
                    edges.append(edge)

                    if len(edges) >= MAX_DEPENDENCIES_PER_FILE * len(self._graph.files):
                        break

        self._graph.edges = edges
        return edges

    def _resolve_import(
        self,
        import_name: str,
        source_file: str,
        language: str,
    ) -> Optional[str]:
        """
        Resolve an import to a target artifact ID.

        AC: Resolves relative imports to absolute artifact IDs.
        Rule #4: < 60 lines.
        Rule #7: Returns None if unresolved.
        """
        # Handle relative imports
        if import_name.startswith("."):
            resolved = self._resolve_relative_import(import_name, source_file, language)
            if resolved:
                return self._graph.module_to_artifact.get(resolved)

        # Direct module lookup
        if import_name in self._graph.module_to_artifact:
            return self._graph.module_to_artifact[import_name]

        # Try partial match (e.g., "mypackage.mymodule" matches "mypackage")
        for module_path, artifact_id in self._graph.module_to_artifact.items():
            if import_name.startswith(module_path + "."):
                return artifact_id
            if module_path.startswith(import_name + "."):
                return artifact_id

        return None

    def _resolve_relative_import(
        self,
        import_name: str,
        source_file: str,
        language: str,
    ) -> Optional[str]:
        """
        Resolve a relative import to an absolute module path.

        Rule #4: < 60 lines.
        """
        if not source_file:
            return None

        source_path = PurePosixPath(source_file.replace("\\", "/"))
        parent = source_path.parent

        # Count leading dots
        dots = 0
        for c in import_name:
            if c == ".":
                dots += 1
            else:
                break

        # Navigate up directories
        for _ in range(dots - 1):
            parent = parent.parent
            if str(parent) == ".":
                break

        # Get the remaining module name
        remaining = import_name[dots:]

        if remaining:
            resolved = str(parent / remaining.replace(".", "/"))
        else:
            resolved = str(parent)

        # Convert path back to module notation
        return resolved.replace("/", ".").replace("\\", ".").strip(".")

    def _file_to_module(self, file_path: str, language: str) -> str:
        """
        Convert file path to module path.

        Rule #4: < 60 lines.
        """
        if not file_path:
            return ""

        path = PurePosixPath(file_path.replace("\\", "/"))

        # Remove extension
        if path.suffix in (".py", ".js", ".ts", ".tsx", ".jsx"):
            path = path.with_suffix("")

        # Remove __init__ for Python packages
        if path.name == "__init__":
            path = path.parent

        # Convert to dot notation
        module = str(path).replace("/", ".").replace("\\", ".")

        # Remove leading dots
        return module.lstrip(".")

    def get_links_for_files(
        self,
        file1_id: str,
        file2_id: str,
    ) -> List[DependencyEdge]:
        """
        Get dependency links between two files.

        AC: Ingesting 2 related files shows a link in the graph.
        Rule #7: Returns empty list if no links.

        Args:
            file1_id: First artifact ID.
            file2_id: Second artifact ID.

        Returns:
            List of edges connecting the two files.
        """
        return [
            e
            for e in self._graph.edges
            if (e.source_artifact_id == file1_id and e.target_artifact_id == file2_id)
            or (e.source_artifact_id == file2_id and e.target_artifact_id == file1_id)
        ]

    def clear(self) -> None:
        """Clear the dependency graph."""
        self._graph = DependencyGraph()


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================


def link_code_artifacts(
    artifacts: List["IFCodeArtifact"],
    base_path: Optional[Path] = None,
) -> DependencyGraph:
    """
    Create a dependency graph from a list of code artifacts.

    Automatically connect code files via import/export.

    Args:
        artifacts: List of code artifacts to link.
        base_path: Base path for resolving imports.

    Returns:
        DependencyGraph with resolved edges.
    """
    linker = StaticDependencyLinker(base_path=base_path)

    for artifact in artifacts:
        linker.register_artifact(artifact)

    linker.resolve_dependencies()
    return linker.graph


def edges_to_manifest_format(
    edges: List[DependencyEdge],
) -> List[Dict[str, Any]]:
    """
    Convert dependency edges to Knowledge Manifest format.

    AC: Creates DEPENDS_ON edges in the Knowledge Manifest.

    Args:
        edges: List of dependency edges.

    Returns:
        List of edge dictionaries for manifest.
    """
    return [
        {
            "source_entity_id": e.source_artifact_id,
            "target_entity_id": e.target_artifact_id,
            "predicate": e.edge_type,
            "properties": {
                "import_name": e.import_name,
                "source_file": e.source_file,
                "target_file": e.target_file,
                "line_number": e.line_number,
                "is_relative": e.is_relative,
            },
            "confidence": e.confidence,
        }
        for e in edges
    ]
