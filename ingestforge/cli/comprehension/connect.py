"""Connect command - Discover relationships between concepts.

Maps how concepts relate to each other and their dependencies.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.panel import Panel
from rich.tree import Tree
from rich.markdown import Markdown

from ingestforge.cli.comprehension.base import ComprehensionCommand


class ConnectCommand(ComprehensionCommand):
    """Discover and visualize concept relationships."""

    def execute(
        self,
        concepts: List[str],
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        show_graph: bool = False,
    ) -> int:
        """
        Find relationships between concepts.

        Rule #4: Function under 60 lines
        """
        try:
            # Validate inputs (Commandment #7)
            self.validate_concepts_list(concepts)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Get LLM client
            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            # Search for context about each concept
            all_chunks = self._gather_concept_contexts(ctx["storage"], concepts)

            if not any(all_chunks.values()):
                self._handle_no_context(concepts)
                return 0

            # Generate relationship map
            relationship_data = self._generate_relationships(
                llm_client, concepts, all_chunks
            )

            if not relationship_data:
                self.print_error("Failed to generate relationship map")
                return 1

            # Display relationships
            self._display_relationships(relationship_data, concepts, show_graph)

            # Save to file if requested
            if output:
                self.save_json_output(
                    output, relationship_data, f"Relationships saved to: {output}"
                )

            return 0

        except Exception as e:
            return self.handle_error(e, "Relationship mapping failed")

    def validate_concepts_list(self, concepts: List[str]) -> None:
        """Validate concept list.

        Args:
            concepts: List of concepts to validate

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        if len(concepts) < 2:
            raise typer.BadParameter("Must provide at least 2 concepts to connect")

        if len(concepts) > 10:
            raise typer.BadParameter(
                "Cannot map relationships for more than 10 concepts at once"
            )

        for concept in concepts:
            self.validate_concept(concept)

    def _gather_concept_contexts(
        self, storage: Any, concepts: List[str]
    ) -> Dict[str, list]:
        """Gather context for each concept.

        Args:
            storage: ChunkRepository instance
            concepts: List of concepts

        Returns:
            Dict mapping concept to chunks
        """
        all_chunks = {}

        for concept in concepts:
            chunks = self.search_concept_context(storage, concept, k=15)
            all_chunks[concept] = chunks

        return all_chunks

    def _handle_no_context(self, concepts: List[str]) -> None:
        """Handle case where no context found.

        Args:
            concepts: Concepts that were searched
        """
        self.print_warning(f"No context found for concepts: {', '.join(concepts)}")
        self.print_info(
            "Try:\n"
            "  1. Ingesting documents about these topics\n"
            "  2. Using broader search terms\n"
            "  3. Checking spelling"
        )

    def _generate_relationships(
        self,
        llm_client: Any,
        concepts: List[str],
        all_chunks: Dict[str, list],
    ) -> Optional[Dict[str, Any]]:
        """Generate relationship map using LLM.

        Args:
            llm_client: LLM provider instance
            concepts: List of concepts
            all_chunks: Context chunks for each concept

        Returns:
            Relationship data dict or None if failed
        """
        # Build combined context
        context = self._build_combined_context(concepts, all_chunks)

        # Build prompt
        prompt = self._build_relationships_prompt(concepts, context)

        # Generate relationships
        response = self.generate_with_llm(llm_client, prompt, "relationship map")

        # Parse JSON response
        relationship_data = self.parse_json_response(response)

        if not relationship_data:
            # Fallback: simple structure
            relationship_data = {
                "concepts": concepts,
                "relationships": [],
                "categories": {},
            }

        return relationship_data

    def _build_combined_context(
        self, concepts: List[str], all_chunks: Dict[str, list]
    ) -> str:
        """Build combined context from all concepts.

        Args:
            concepts: List of concepts
            all_chunks: Chunks for each concept

        Returns:
            Combined context string
        """
        context_parts = []

        for concept in concepts:
            chunks = all_chunks.get(concept, [])
            if chunks:
                concept_context = self.format_context_for_prompt(
                    chunks, max_length=1000
                )
                context_parts.append(
                    f"--- Context for '{concept}' ---\n{concept_context}\n"
                )

        return "\n".join(context_parts)

    def _build_relationships_prompt(self, concepts: List[str], context: str) -> str:
        """Build prompt for relationship generation.

        Args:
            concepts: List of concepts
            context: Combined context

        Returns:
            Formatted prompt
        """
        concepts_str = ", ".join(f'"{c}"' for c in concepts)

        prompt_parts = [
            f"Map relationships between these concepts: {concepts_str}\n",
            "\nContext from knowledge base:\n",
            context,
            "\n\nGenerate relationship map in this JSON format:\n",
            "{\n",
            f'  "concepts": [{concepts_str}],\n',
            '  "relationships": [\n',
            "    {\n",
            '      "from": "concept1",\n',
            '      "to": "concept2",\n',
            '      "type": "depends_on|part_of|similar_to|uses|enables|etc",\n',
            '      "description": "how they relate",\n',
            '      "strength": "weak|moderate|strong"\n',
            "    },\n",
            "    ...\n",
            "  ],\n",
            '  "categories": {\n',
            '    "foundational": ["concept1", ...],\n',
            '    "intermediate": [...],\n',
            '    "advanced": [...]\n',
            "  },\n",
            '  "dependencies": {\n',
            '    "concept1": ["prerequisite1", "prerequisite2"],\n',
            "    ...\n",
            "  },\n",
            '  "learning_path": ["concept1", "concept2", ...],\n',
            '  "overview": "brief description of how concepts relate"\n',
            "}\n",
            "\nRelationship types:\n",
            "- depends_on: B depends on A\n",
            "- part_of: B is part of A\n",
            "- similar_to: A and B are similar\n",
            "- uses: A uses B\n",
            "- enables: A enables B\n",
            "- prerequisite: A is prerequisite for B\n",
            "\nRequirements:\n",
            "- Base relationships on provided context\n",
            "- Identify all meaningful connections\n",
            "- Categorize concepts by complexity\n",
            "- Map dependencies between concepts\n",
            "- Suggest logical learning path\n",
            "\nReturn ONLY valid JSON, no additional text.",
        ]

        return "".join(prompt_parts)

    def _display_relationships(
        self,
        relationship_data: Dict[str, Any],
        concepts: List[str],
        show_graph: bool,
    ) -> None:
        """Display relationships.

        Args:
            relationship_data: Relationship data dict
            concepts: List of concepts
            show_graph: Whether to show graph visualization
        """
        self.console.print()

        # Title
        title = "[bold cyan]Concept Relationships[/bold cyan]"
        self.console.print(Panel(title, border_style="cyan"))

        self.console.print()

        # Overview
        overview = relationship_data.get("overview", "")
        if overview:
            self.console.print("[bold]Overview:[/bold]")
            self.console.print(Markdown(overview))
            self.console.print()

        # Relationships
        relationships = relationship_data.get("relationships", [])
        if relationships:
            self._display_relationships_list(relationships)

        # Categories
        categories = relationship_data.get("categories", {})
        if categories:
            self._display_categories(categories)

        # Dependencies
        dependencies = relationship_data.get("dependencies", {})
        if dependencies:
            self._display_dependencies(dependencies)

        # Learning path
        learning_path = relationship_data.get("learning_path", [])
        if learning_path:
            self._display_learning_path(learning_path)

        # Graph visualization
        if show_graph:
            self._display_graph(relationship_data)

    def _display_relationships_list(self, relationships: List[Dict[str, Any]]) -> None:
        """Display relationships as list.

        Args:
            relationships: List of relationship dicts
        """
        self.console.print("[bold green]Relationships:[/bold green]")

        for rel in relationships:
            from_concept = rel.get("from", "")
            to_concept = rel.get("to", "")
            rel_type = rel.get("type", "relates_to")
            description = rel.get("description", "")
            strength = rel.get("strength", "moderate")

            # Color code by strength
            strength_color = {
                "weak": "dim",
                "moderate": "yellow",
                "strong": "bright_yellow",
            }.get(strength, "yellow")

            self.console.print(
                f"  [{strength_color}]{from_concept}[/{strength_color}] "
                f"[cyan]{rel_type}[/cyan] "
                f"[{strength_color}]{to_concept}[/{strength_color}]"
            )

            if description:
                self.console.print(f"    {description}")

        self.console.print()

    def _display_categories(self, categories: Dict[str, List[str]]) -> None:
        """Display concept categories.

        Args:
            categories: Dict mapping category to concepts
        """
        self.console.print("[bold blue]Categories:[/bold blue]")

        category_order = ["foundational", "intermediate", "advanced"]

        for category in category_order:
            if category in categories:
                concepts_list = categories[category]
                self.console.print(
                    f"  [yellow]{category.title()}:[/yellow] {', '.join(concepts_list)}"
                )

        self.console.print()

    def _display_dependencies(self, dependencies: Dict[str, List[str]]) -> None:
        """Display concept dependencies.

        Args:
            dependencies: Dict mapping concept to prerequisites
        """
        self.console.print("[bold magenta]Dependencies:[/bold magenta]")

        for concept, prereqs in dependencies.items():
            if prereqs:
                self.console.print(
                    f"  [yellow]{concept}[/yellow] requires: {', '.join(prereqs)}"
                )

        self.console.print()

    def _display_learning_path(self, learning_path: List[str]) -> None:
        """Display suggested learning path.

        Args:
            learning_path: Ordered list of concepts
        """
        self.console.print("[bold cyan]Suggested Learning Path:[/bold cyan]")

        for idx, concept in enumerate(learning_path, 1):
            arrow = " → " if idx < len(learning_path) else ""
            self.console.print(f"  {idx}. [yellow]{concept}[/yellow]{arrow}", end="")

        self.console.print("\n")

    def _display_graph(self, relationship_data: Dict[str, Any]) -> None:
        """Display graph visualization using tree.

        Args:
            relationship_data: Relationship data dict
        """
        self.console.print("[bold cyan]Relationship Graph:[/bold cyan]")

        # Build tree from relationships
        concepts = relationship_data.get("concepts", [])
        relationships = relationship_data.get("relationships", [])

        # Create root tree
        tree = Tree("[bold]Concept Network[/bold]")

        # Group by root concepts (those with no dependencies)
        dependencies = relationship_data.get("dependencies", {})
        roots = [c for c in concepts if not dependencies.get(c, [])]

        # Add branches
        for root in roots:
            branch = tree.add(f"[yellow]{root}[/yellow]")
            self._add_related_concepts(branch, root, relationships)

        self.console.print(tree)
        self.console.print()

    def _add_related_concepts(
        self, branch: Any, concept: str, relationships: List[Dict]
    ) -> None:
        """Recursively add related concepts to tree.

        Args:
            branch: Tree branch
            concept: Current concept
            relationships: All relationships
        """
        # Find concepts this one relates to
        related = [r for r in relationships if r.get("from") == concept]

        for rel in related[:3]:  # Limit depth
            to_concept = rel.get("to", "")
            rel_type = rel.get("type", "")

            sub_branch = branch.add(
                f"[cyan]{rel_type}[/cyan] → [yellow]{to_concept}[/yellow]"
            )


# Typer command wrapper
def command(
    concepts: List[str] = typer.Argument(
        ..., help="Concepts to connect (2-10 concepts)"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file (JSON format)"
    ),
    show_graph: bool = typer.Option(
        False, "--graph", "-g", help="Show graph visualization"
    ),
) -> None:
    """Discover relationships between concepts.

    Maps how concepts relate, their dependencies, and suggests
    a logical learning path based on your knowledge base.

    Examples:
        # Map relationships
        ingestforge comprehension connect "Variables" "Functions" "Classes"

        # Show graph visualization
        ingestforge comprehension connect "HTTP" "REST" "GraphQL" --graph

        # Save to file
        ingestforge comprehension connect "Git" "GitHub" "CI/CD" -o connections.json

        # Specific project
        ingestforge comprehension connect "TCP" "UDP" "HTTP" -p /path/to/project
    """
    cmd = ConnectCommand()
    exit_code = cmd.execute(concepts, project, output, show_graph)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
