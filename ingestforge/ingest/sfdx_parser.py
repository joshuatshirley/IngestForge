"""
SFDX Project Parser for Salesforce codebase understanding.

Parses sfdx-project.json to extract package structure, dependencies,
and map source files to their parent packages.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Set


@dataclass
class SFDXPackage:
    """Represents a Salesforce DX package."""

    name: str
    path: str
    version: str = ""
    version_name: str = ""
    default: bool = False
    dependencies: List[str] = field(default_factory=list)
    package_type: str = "source"  # source, data, unlocked
    files: List[Path] = field(default_factory=list)  # Populated by scanning

    @property
    def is_data_package(self) -> bool:
        """Check if this is a data package."""
        return self.package_type == "data"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "path": self.path,
            "version": self.version,
            "version_name": self.version_name,
            "default": self.default,
            "dependencies": self.dependencies,
            "package_type": self.package_type,
            "file_count": len(self.files),
        }


class SFDXProjectParser:
    """
    Parse sfdx-project.json and manage package structure.

    Provides methods to:
    - Parse package directories and versions
    - Build dependency graph between packages
    - Map source files to their parent package
    - Scan for Apex, LWC, and other metadata

    Example:
        parser = SFDXProjectParser()
        parser.parse(Path("./aie-project"))

        packages = parser.get_packages()
        for pkg in packages:
            print(f"{pkg.name}: {len(pkg.files)} files")

        pkg_name = parser.get_package_for_file(Path("./src/aie-base-code/classes/Foo.cls"))
        print(pkg_name)  # "aie-base-code"
    """

    def __init__(self) -> None:
        self.packages: List[SFDXPackage] = []
        self.project_root: Optional[Path] = None
        self.source_api_version: str = ""
        self.namespace: str = ""
        self._path_to_package: Dict[str, str] = {}  # Normalized path -> package name

    def parse(self, project_root: Path) -> List[SFDXPackage]:
        """
        Parse sfdx-project.json and populate package list.

        Args:
            project_root: Path to the SFDX project root (containing sfdx-project.json)

        Returns:
            List of SFDXPackage objects

        Raises:
            FileNotFoundError: If sfdx-project.json is not found
            ValueError: If the file is not valid JSON
        """
        self.project_root = project_root
        sfdx_file = project_root / "sfdx-project.json"

        if not sfdx_file.exists():
            raise FileNotFoundError(f"sfdx-project.json not found at {project_root}")

        with open(sfdx_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Extract project-level settings
        self.source_api_version = data.get("sourceApiVersion", "")
        self.namespace = data.get("namespace", "")

        # Parse package directories
        self.packages = []
        self._path_to_package = {}

        for pkg_dir in data.get("packageDirectories", []):
            package = self._parse_package_directory(pkg_dir)
            self.packages.append(package)

            # Build path-to-package mapping
            normalized_path = self._normalize_path(pkg_dir.get("path", ""))
            self._path_to_package[normalized_path] = package.name

        return self.packages

    def _parse_package_directory(self, pkg_dir: Dict[str, Any]) -> SFDXPackage:
        """Parse a single package directory entry."""
        path = pkg_dir.get("path", "")
        name = pkg_dir.get("package", "")

        # If no package name, derive from path
        if not name:
            # e.g., "./src/aie-base-code" -> "aie-base-code"
            name = Path(path).name

        # Determine package type
        pkg_type = pkg_dir.get("type", "source")
        if pkg_type == "data" or name.startswith("data-"):
            pkg_type = "data"

        # Parse dependencies
        dependencies = []
        for dep in pkg_dir.get("dependencies", []):
            if isinstance(dep, dict):
                dep_name = dep.get("package", "")
            else:
                dep_name = str(dep)
            if dep_name:
                dependencies.append(dep_name)

        return SFDXPackage(
            name=name,
            path=path,
            version=pkg_dir.get("versionNumber", ""),
            version_name=pkg_dir.get("versionName", ""),
            default=pkg_dir.get("default", False),
            dependencies=dependencies,
            package_type=pkg_type,
        )

    def get_packages(self) -> List[SFDXPackage]:
        """Get all parsed packages."""
        return self.packages

    def get_package(self, name: str) -> Optional[SFDXPackage]:
        """Get a package by name."""
        for pkg in self.packages:
            if pkg.name == name:
                return pkg
        return None

    def get_package_for_file(self, file_path: Path) -> Optional[str]:
        """
        Determine which package a file belongs to.

        Args:
            file_path: Path to the source file

        Returns:
            Package name, or None if not found
        """
        # Convert to absolute path relative to project root
        try:
            if self.project_root:
                if file_path.is_absolute():
                    rel_path = file_path.relative_to(self.project_root)
                else:
                    rel_path = file_path
            else:
                rel_path = file_path
        except ValueError:
            # File is not under project root
            rel_path = file_path

        # Normalize and search for matching package path
        path_str = self._normalize_path(str(rel_path))

        # Find the longest matching package path
        best_match = None
        best_length = 0

        for pkg_path, pkg_name in self._path_to_package.items():
            if path_str.startswith(pkg_path) and len(pkg_path) > best_length:
                best_match = pkg_name
                best_length = len(pkg_path)

        return best_match

    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Build a dependency graph between packages.

        Returns:
            Dict mapping package name to list of dependency package names
        """
        graph = {}
        for pkg in self.packages:
            graph[pkg.name] = pkg.dependencies.copy()
        return graph

    def get_reverse_dependencies(self) -> Dict[str, List[str]]:
        """
        Build reverse dependency graph (what depends on each package).

        Returns:
            Dict mapping package name to list of packages that depend on it
        """
        reverse: dict[str, list[str]] = {pkg.name: [] for pkg in self.packages}

        for pkg in self.packages:
            for dep in pkg.dependencies:
                if dep in reverse:
                    reverse[dep].append(pkg.name)

        return reverse

    def get_package_order(self) -> List[str]:
        """
        Get topological order of packages (dependencies first).

        Returns:
            List of package names in dependency order
        """
        graph = self.get_dependency_graph()
        visited = set()
        order = []

        def visit(name: str, stack: Set[str]) -> Any:
            if name in stack:
                # Circular dependency
                return
            if name in visited:
                return

            stack.add(name)
            for dep in graph.get(name, []):
                if dep in graph:  # Only visit packages we know about
                    visit(dep, stack)
            stack.remove(name)
            visited.add(name)
            order.append(name)

        for pkg in self.packages:
            visit(pkg.name, set())

        return order

    def scan_package_files(
        self,
        package_name: Optional[str] = None,
        extensions: Optional[List[str]] = None,
    ) -> Dict[str, List[Path]]:
        """
        Scan packages for source files.

        Args:
            package_name: Optional package to scan (all if None)
            extensions: File extensions to include (e.g., ['.cls', '.trigger'])

        Returns:
            Dict mapping package name to list of file paths
        """
        if not self.project_root:
            return {}

        if extensions is None:
            extensions = [".cls", ".trigger", ".js", ".html", ".cmp", ".xml"]

        result: dict[str, Any] = {}

        packages = self.packages
        if package_name:
            pkg = self.get_package(package_name)
            packages = [pkg] if pkg else []

        for pkg in packages:
            pkg_path = self.project_root / pkg.path
            if not pkg_path.exists():
                result[pkg.name] = []
                continue

            files: list[Path] = []
            for ext in extensions:
                files.extend(pkg_path.rglob(f"*{ext}"))

            result[pkg.name] = sorted(files)
            pkg.files = files  # Also populate the package object

        return result

    def get_apex_classes(
        self, package_name: Optional[str] = None
    ) -> Dict[str, List[Path]]:
        """Get Apex class files (.cls) for packages."""
        return self.scan_package_files(package_name, extensions=[".cls"])

    def get_triggers(self, package_name: Optional[str] = None) -> Dict[str, List[Path]]:
        """Get trigger files (.trigger) for packages."""
        return self.scan_package_files(package_name, extensions=[".trigger"])

    def _is_valid_lwc_component(self, component_dir: Path) -> bool:
        """
        Check if directory is a valid LWC component.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            component_dir: Path to potential component directory

        Returns:
            True if valid LWC component directory
        """
        if not component_dir.is_dir():
            return False
        if component_dir.name.startswith("."):
            return False

        return True

    def _collect_lwc_components(self, lwc_parent: Path) -> List[Path]:
        """
        Collect LWC component directories from parent.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            lwc_parent: Parent directory containing LWC components

        Returns:
            List of component directory paths
        """
        components = []
        for component_dir in lwc_parent.iterdir():
            if self._is_valid_lwc_component(component_dir):
                components.append(component_dir)

        return components

    def _find_lwc_components_in_package(self, pkg_path: Path) -> List[Path]:
        """
        Find all LWC components in a package.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            pkg_path: Path to package directory

        Returns:
            List of LWC component directories
        """
        lwc_dirs = []

        # Look for lwc directories
        for lwc_parent in pkg_path.rglob("lwc"):
            if not lwc_parent.is_dir():
                continue

            # Collect components from this lwc directory
            components = self._collect_lwc_components(lwc_parent)
            lwc_dirs.extend(components)

        return lwc_dirs

    def get_lwc_components(
        self, package_name: Optional[str] = None
    ) -> Dict[str, List[Path]]:
        """
        Get LWC component directories for packages.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            package_name: Optional specific package name

        Returns:
            Dictionary mapping package names to component paths
        """
        if not self.project_root:
            return {}

        result: dict[str, Any] = {}

        packages = self.packages
        if package_name:
            pkg = self.get_package(package_name)
            packages = [pkg] if pkg else []
        for pkg in packages:
            pkg_path = self.project_root / pkg.path
            lwc_dirs = self._find_lwc_components_in_package(pkg_path)
            result[pkg.name] = sorted(lwc_dirs)

        return result

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the SFDX project."""
        source_packages = [p for p in self.packages if p.package_type == "source"]
        data_packages = [p for p in self.packages if p.package_type == "data"]

        return {
            "project_root": str(self.project_root) if self.project_root else None,
            "source_api_version": self.source_api_version,
            "namespace": self.namespace,
            "total_packages": len(self.packages),
            "source_packages": len(source_packages),
            "data_packages": len(data_packages),
            "packages": [p.to_dict() for p in self.packages],
        }

    def _normalize_path(self, path: str) -> str:
        """Normalize a path for comparison."""
        # Convert to forward slashes, remove leading ./
        normalized = path.replace("\\", "/")
        if normalized.startswith("./"):
            normalized = normalized[2:]
        if normalized.startswith("/"):
            normalized = normalized[1:]
        return normalized.lower()

    def to_mermaid(self, include_data: bool = False) -> str:
        """
        Generate a Mermaid diagram of package dependencies.

        Args:
            include_data: Include data packages in the diagram

        Returns:
            Mermaid diagram source code
        """
        lines = ["graph TD"]

        # Filter packages
        packages = self.packages
        if not include_data:
            packages = [p for p in packages if p.package_type != "data"]

        pkg_names = {p.name for p in packages}

        # Add nodes with styling
        for pkg in packages:
            # Determine node style based on package type
            if pkg.package_type == "data":
                lines.append(f'    {self._mermaid_id(pkg.name)}["{pkg.name}"]:::data')
            elif pkg.default:
                lines.append(
                    f'    {self._mermaid_id(pkg.name)}["{pkg.name}"]:::default'
                )
            else:
                lines.append(f'    {self._mermaid_id(pkg.name)}["{pkg.name}"]')

        # Add edges for dependencies
        for pkg in packages:
            for dep in pkg.dependencies:
                if dep in pkg_names:
                    lines.append(
                        f"    {self._mermaid_id(dep)} --> {self._mermaid_id(pkg.name)}"
                    )

        # Add styling
        lines.extend(
            [
                "",
                "    classDef default fill:#e1f5fe",
                "    classDef data fill:#fff3e0",
            ]
        )

        return "\n".join(lines)

    def _mermaid_id(self, name: str) -> str:
        """Convert package name to valid Mermaid ID."""
        return name.replace("-", "_").replace(".", "_")
