"""Map command - Generate code maps and dependency graphs.

Generates visual representations and documentation of code structure.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Dict, Any
import typer
from rich.panel import Panel
from rich.tree import Tree

from ingestforge.cli.code.base import CodeAnalysisCommand


class MapCommand(CodeAnalysisCommand):
    """Generate code maps and dependency graphs."""

    def execute(
        self,
        target: Path,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        format_type: str = "tree",
    ) -> int:
        """Generate code map and structure visualization.

        Args:
            target: Code directory to map
            project: Project directory
            output: Output file for map
            format_type: Map format (tree/markdown/mermaid)

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate inputs (Commandment #7: Check parameters)
            self.validate_code_path(target)
            self.validate_format_type(format_type)

            if not target.is_dir():
                raise typer.BadParameter("Target must be a directory for mapping")

            # Initialize context
            ctx = self.initialize_context(project, require_storage=False)

            # Generate map
            map_data = self._generate_map(target, format_type)

            # Display results
            if format_type == "tree":
                self._display_tree(map_data, target)
            else:
                self._display_formatted_map(map_data, target, format_type)

            # Save to file if requested
            if output:
                self._save_map(output, map_data, format_type)

            return 0

        except Exception as e:
            return self.handle_error(e, "Code mapping failed")

    def validate_format_type(self, format_type: str) -> None:
        """Validate format type.

        Args:
            format_type: Format string to validate

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        valid_formats = ["tree", "markdown", "mermaid"]

        if format_type.lower() not in valid_formats:
            raise typer.BadParameter(
                f"Invalid format '{format_type}'. "
                f"Must be one of: {', '.join(valid_formats)}"
            )

    def _generate_map(self, target: Path, format_type: str) -> Dict[str, Any]:
        """Generate code map.

        Args:
            target: Target directory
            format_type: Output format

        Returns:
            Map data dictionary
        """
        # Build directory structure
        structure = self._build_structure(target)

        # Get file statistics
        stats = self._get_statistics(structure)

        return {
            "target": str(target),
            "format": format_type,
            "structure": structure,
            "statistics": stats,
        }

    def _build_structure(self, directory: Path) -> Dict[str, Any]:
        """Build directory structure.

        Args:
            directory: Directory to analyze

        Returns:
            Structure dictionary
        """
        code_files = self.find_code_files(directory)
        grouped = self.group_files_by_extension(code_files)

        structure = {
            "name": directory.name,
            "path": str(directory),
            "files": [],
            "subdirs": {},
        }

        for file_path in code_files:
            try:
                rel_path = file_path.relative_to(directory)
                parts = rel_path.parts

                if len(parts) == 1:
                    self._process_root_file(file_path, structure)
                else:
                    self._process_subdir_file(file_path, structure, parts[0])

            except ValueError:
                # File not relative to directory, skip
                continue

        return structure

    def _process_root_file(self, file_path: Path, structure: Dict[str, Any]) -> None:
        """Process a file in the root directory.

        Args:
            file_path: Path to file
            structure: Structure dictionary to update
        """
        structure["files"].append(
            {
                "name": file_path.name,
                "extension": file_path.suffix,
                "path": str(file_path),
            }
        )

    def _process_subdir_file(
        self, file_path: Path, structure: Dict[str, Any], subdir: str
    ) -> None:
        """Process a file in a subdirectory.

        Args:
            file_path: Path to file
            structure: Structure dictionary to update
            subdir: Immediate subdirectory name
        """
        subdir_dict = self._get_or_create_subdir(structure, subdir)

        subdir_dict["files"].append(
            {
                "name": file_path.name,
                "extension": file_path.suffix,
                "path": str(file_path),
            }
        )

    def _get_or_create_subdir(
        self, structure: Dict[str, Any], subdir_name: str
    ) -> Dict[str, Any]:
        """Get or create subdirectory entry in structure.

        Args:
            structure: Structure dictionary
            subdir_name: Subdirectory name

        Returns:
            Subdirectory dictionary
        """
        if subdir_name not in structure["subdirs"]:
            structure["subdirs"][subdir_name] = {
                "name": subdir_name,
                "files": [],
            }

        return structure["subdirs"][subdir_name]

    def _get_statistics(self, structure: Dict[str, Any]) -> Dict[str, Any]:
        """Get statistics from structure.

        Args:
            structure: Structure dictionary

        Returns:
            Statistics dictionary
        """
        total_files = len(structure["files"])
        total_dirs = len(structure["subdirs"])

        for subdir_data in structure["subdirs"].values():
            total_files += len(subdir_data["files"])

        # Count by extension
        extensions: Dict[str, int] = {}

        for file_info in structure["files"]:
            ext = file_info["extension"]
            extensions[ext] = extensions.get(ext, 0) + 1

        for subdir_data in structure["subdirs"].values():
            for file_info in subdir_data["files"]:
                ext = file_info["extension"]
                extensions[ext] = extensions.get(ext, 0) + 1

        return {
            "total_files": total_files,
            "total_directories": total_dirs,
            "extensions": extensions,
        }

    def _display_tree(self, map_data: Dict[str, Any], target: Path) -> None:
        """Display structure as tree.

        Args:
            map_data: Map data
            target: Target directory
        """
        self.console.print()

        tree = Tree(
            f"[bold cyan]{target.name}[/bold cyan]",
            guide_style="bright_blue",
        )

        structure = map_data["structure"]

        # Add root files
        for file_info in structure["files"]:
            tree.add(f"[green]{file_info['name']}[/green]")

        # Add subdirectories
        for subdir_name, subdir_data in sorted(structure["subdirs"].items()):
            branch = tree.add(f"[cyan]{subdir_name}/[/cyan]")

            for file_info in subdir_data["files"]:
                branch.add(f"[green]{file_info['name']}[/green]")

        self.console.print(tree)
        self.console.print()

        # Display statistics
        self._display_statistics_summary(map_data["statistics"])

    def _display_formatted_map(
        self, map_data: Dict[str, Any], target: Path, format_type: str
    ) -> None:
        """Display formatted map.

        Args:
            map_data: Map data
            target: Target directory
            format_type: Format type
        """
        self.console.print()

        if format_type == "markdown":
            content = self._format_as_markdown(map_data)
        elif format_type == "mermaid":
            content = self._format_as_mermaid(map_data)
        else:
            content = "Unsupported format"

        panel = Panel(
            content,
            title=f"[bold cyan]Code Map: {target.name}[/bold cyan]",
            border_style="cyan",
        )

        self.console.print(panel)

    def _display_statistics_summary(self, stats: Dict[str, Any]) -> None:
        """Display statistics summary.

        Args:
            stats: Statistics dictionary
        """
        self.print_info(f"Total files: {stats['total_files']}")
        self.print_info(f"Total directories: {stats['total_directories']}")

        if stats["extensions"]:
            ext_str = ", ".join(
                f"{ext}({count})" for ext, count in sorted(stats["extensions"].items())
            )
            self.print_info(f"File types: {ext_str}")

    def _format_as_markdown(self, map_data: Dict[str, Any]) -> str:
        """Format map as markdown.

        Args:
            map_data: Map data

        Returns:
            Markdown string
        """
        lines = [f"# {map_data['structure']['name']}\n\n"]

        structure = map_data["structure"]

        # Root files
        if structure["files"]:
            lines.append("## Root Files\n\n")
            for file_info in structure["files"]:
                lines.append(f"- `{file_info['name']}`\n")
            lines.append("\n")

        # Subdirectories
        if structure["subdirs"]:
            lines.append("## Directories\n\n")
            for subdir_name, subdir_data in sorted(structure["subdirs"].items()):
                lines.append(f"### {subdir_name}/\n\n")
                for file_info in subdir_data["files"]:
                    lines.append(f"- `{file_info['name']}`\n")
                lines.append("\n")

        return "".join(lines)

    def _format_as_mermaid(self, map_data: Dict[str, Any]) -> str:
        """Format map as mermaid diagram.

        Args:
            map_data: Map data

        Returns:
            Mermaid diagram string
        """
        lines = ["```mermaid\n", "graph TD\n"]

        structure = map_data["structure"]
        root_id = "root"

        lines.append(f"    {root_id}[{structure['name']}]\n")

        # Root files
        for idx, file_info in enumerate(structure["files"]):
            file_id = f"file{idx}"
            lines.append(f"    {file_id}[{file_info['name']}]\n")
            lines.append(f"    {root_id} --> {file_id}\n")

        # Subdirectories
        for subdir_idx, (subdir_name, subdir_data) in enumerate(
            sorted(structure["subdirs"].items())
        ):
            subdir_id = f"dir{subdir_idx}"
            lines.append(f"    {subdir_id}[{subdir_name}/]\n")
            lines.append(f"    {root_id} --> {subdir_id}\n")

            for file_idx, file_info in enumerate(subdir_data["files"]):
                file_id = f"{subdir_id}_file{file_idx}"
                lines.append(f"    {file_id}[{file_info['name']}]\n")
                lines.append(f"    {subdir_id} --> {file_id}\n")

        lines.append("```\n")

        return "".join(lines)

    def _save_map(
        self, output: Path, map_data: Dict[str, Any], format_type: str
    ) -> None:
        """Save map to file.

        Args:
            output: Output file path
            map_data: Map data
            format_type: Format type
        """
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            if format_type == "markdown":
                content = self._format_as_markdown(map_data)
                header = f"# Code Map\n\n" f"**Generated:** {timestamp}\n\n" f"---\n\n"
                content = header + content

            elif format_type == "mermaid":
                content = self._format_as_mermaid(map_data)
                header = f"# Code Map\n\n" f"**Generated:** {timestamp}\n\n"
                content = header + content

            else:
                # Plain text tree format
                content = self._format_as_text_tree(map_data)

            output.write_text(content, encoding="utf-8")
            self.print_success(f"Map saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save map: {e}")

    def _format_as_text_tree(self, map_data: Dict[str, Any]) -> str:
        """Format as plain text tree.

        Args:
            map_data: Map data

        Returns:
            Text tree string
        """
        lines = [f"{map_data['structure']['name']}/\n"]

        structure = map_data["structure"]

        # Root files
        for file_info in structure["files"]:
            lines.append(f"├── {file_info['name']}\n")

        # Subdirectories
        subdirs = list(structure["subdirs"].items())
        for idx, (subdir_name, subdir_data) in enumerate(subdirs):
            is_last = idx == len(subdirs) - 1
            prefix = "└──" if is_last else "├──"

            lines.append(f"{prefix} {subdir_name}/\n")

            for file_info in subdir_data["files"]:
                sub_prefix = "    " if is_last else "│   "
                lines.append(f"{sub_prefix}├── {file_info['name']}\n")

        return "".join(lines)


# Typer command wrapper
def command(
    target: Path = typer.Argument(..., help="Code directory to map"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for map"
    ),
    format_type: str = typer.Option(
        "tree", "--format", "-f", help="Output format (tree/markdown/mermaid)"
    ),
) -> None:
    """Generate code map and structure visualization.

    Creates visual representations of code structure including
    directory layouts, file organization, and relationships.

    Output formats:
    - tree: Interactive tree view (default)
    - markdown: Markdown documentation
    - mermaid: Mermaid diagram for GitHub/docs

    Examples:
        # Display interactive tree
        ingestforge code map src/

        # Generate markdown documentation
        ingestforge code map src/ --format markdown -o structure.md

        # Generate mermaid diagram
        ingestforge code map app/ --format mermaid -o diagram.md

        # Specific project
        ingestforge code map code/ -p /path/to/project
    """
    cmd = MapCommand()
    exit_code = cmd.execute(target, project, output, format_type)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
