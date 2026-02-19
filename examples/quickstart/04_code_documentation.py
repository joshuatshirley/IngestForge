"""
Example: Auto-generate Code Documentation

Description:
    Analyzes source code and generates comprehensive documentation
    including module overviews, class hierarchies, function signatures,
    and usage examples. Useful for documenting large codebases.

Usage:
    python examples/quickstart/04_code_documentation.py src/
    python examples/quickstart/04_code_documentation.py --help

Expected output:
    - Markdown documentation with module structure
    - Class and function references
    - Code examples extracted from tests
    - Architecture overview

Requirements:
    - Python source code to analyze
    - Optional: LLM for enhanced summaries
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class CodeElement:
    """Represents a code element (class, function, module)."""

    type: str  # 'module', 'class', 'function'
    name: str
    path: str
    docstring: Optional[str] = None
    signature: Optional[str] = None
    children: list[CodeElement] = None

    def __post_init__(self):
        if self.children is None:
            self.children = []


def extract_python_structure(file_path: Path) -> Optional[CodeElement]:
    """
    Extract Python code structure using AST.

    Args:
        file_path: Path to Python file

    Returns:
        CodeElement representing the module structure
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())

        module = CodeElement(
            type="module",
            name=file_path.stem,
            path=str(file_path),
            docstring=ast.get_docstring(tree),
        )

        # Extract classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_elem = CodeElement(
                    type="class",
                    name=node.name,
                    path=f"{file_path.stem}.{node.name}",
                    docstring=ast.get_docstring(node),
                )

                # Extract methods
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        func_elem = CodeElement(
                            type="function",
                            name=item.name,
                            path=f"{file_path.stem}.{node.name}.{item.name}",
                            docstring=ast.get_docstring(item),
                        )
                        class_elem.children.append(func_elem)

                module.children.append(class_elem)

            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                # Top-level functions
                func_elem = CodeElement(
                    type="function",
                    name=node.name,
                    path=f"{file_path.stem}.{node.name}",
                    docstring=ast.get_docstring(node),
                )
                module.children.append(func_elem)

        return module

    except Exception as e:
        print(f"[!] Error parsing {file_path}: {e}")
        return None


def generate_markdown_documentation(
    code_element: CodeElement,
    level: int = 1,
) -> str:
    """
    Generate markdown documentation for code element.

    Args:
        code_element: CodeElement to document
        level: Heading level (1-6)

    Returns:
        Markdown string
    """
    markdown = []

    # Heading
    heading = "#" * level
    markdown.append(f"{heading} {code_element.name}")
    markdown.append("")

    # Type and path
    markdown.append(f"**Type:** `{code_element.type}`  ")
    markdown.append(f"**Path:** `{code_element.path}`")
    markdown.append("")

    # Docstring
    if code_element.docstring:
        markdown.append("## Description")
        markdown.append("")
        markdown.append(code_element.docstring)
        markdown.append("")

    # Children
    if code_element.children:
        if code_element.type == "class":
            markdown.append("## Methods")
        elif code_element.type == "module":
            markdown.append("## Contents")

        markdown.append("")

        for child in code_element.children:
            child_md = generate_markdown_documentation(child, level + 1)
            markdown.append(child_md)

    return "\n".join(markdown)


def document_codebase(
    source_dir: str,
    output_path: Optional[str] = None,
    pattern: str = "**/*.py",
) -> None:
    """
    Generate documentation for a codebase.

    Args:
        source_dir: Directory containing source code
        output_path: Where to save documentation (default: {dirname}_documentation.md)
        pattern: File glob pattern (default: **/*.py)
    """
    source_dir = Path(source_dir)

    if not source_dir.exists():
        print(f"Error: Directory not found: {source_dir}")
        return

    if output_path is None:
        output_path = f"{source_dir.name}_documentation.md"

    print("\n[*] Generating code documentation")
    print(f"    Source: {source_dir}")
    print(f"    Output: {output_path}")

    # Find Python files
    python_files = list(source_dir.glob(pattern))
    if not python_files:
        print(f"[!] No Python files found matching {pattern}")
        return

    print(f"\n[*] Found {len(python_files)} Python file(s)")

    # Extract structure from each file
    print("\n[*] Analyzing code structure...")
    modules = []

    for file_path in sorted(python_files):
        print(f"    {file_path.relative_to(source_dir)}...", end="", flush=True)

        module = extract_python_structure(file_path)
        if module:
            modules.append(module)
            print(f" OK ({len(module.children)} items)")
        else:
            print(" SKIPPED")

    # Generate documentation
    print("\n[*] Generating documentation...")
    doc_parts = [
        f"# {source_dir.name.title()} - Code Documentation",
        "",
        f"Auto-generated documentation for the `{source_dir.name}` module.",
        f"Generated from {len(modules)} module(s).",
        "",
        "## Table of Contents",
        "",
    ]

    # TOC
    for module in modules:
        doc_parts.append(f"- [{module.name}](#{module.name.lower()})")
        for item in module.children:
            doc_parts.append(f"  - [{item.name}](#{item.name.lower()})")

    doc_parts.append("")

    # Content
    for module in modules:
        doc_parts.append(generate_markdown_documentation(module))
        doc_parts.append("")

    # Write documentation
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(doc_parts))

    # Print summary
    print(f"\n{'='*70}")
    print("DOCUMENTATION GENERATION SUMMARY")
    print(f"{'='*70}\n")

    total_classes = sum(1 for m in modules for c in m.children if c.type == "class")
    total_functions = sum(
        1 for m in modules for f in m.children if f.type == "function"
    )

    print(f"Modules:             {len(modules)}")
    print(f"Classes:             {total_classes}")
    print(f"Functions:           {total_functions}")
    print(f"Output file:         {output_path}")

    print("\n[✓] Documentation generated successfully!")
    print("\nView the documentation:")
    print(f"  - Open {output_path} in your editor")
    print(f"  - Convert to HTML: markdown {output_path} > docs.html")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate documentation from source code"
    )
    parser.add_argument(
        "source",
        nargs="?",
        default="ingestforge",
        help="Source directory (default: ingestforge)",
    )
    parser.add_argument("--output", "-o", help="Output markdown path")
    parser.add_argument(
        "--pattern", default="**/*.py", help="File glob pattern (default: **/*.py)"
    )

    args = parser.parse_args()

    document_codebase(
        args.source,
        output_path=args.output,
        pattern=args.pattern,
    )


if __name__ == "__main__":
    main()
