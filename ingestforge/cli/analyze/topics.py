"""Topics command - Analyze and extract topics from content.

Analyzes content to identify main topics, themes, and subject areas.

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from ingestforge.cli.analyze.base import AnalyzeCommand


class TopicsCommand(AnalyzeCommand):
    """Analyze and extract topics from content."""

    def execute(
        self,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        num_topics: int = 10,
    ) -> int:
        """Analyze topics in knowledge base.

        Args:
            project: Project directory
            output: Output file for analysis
            num_topics: Number of top topics to identify

        Returns:
            0 on success, 1 on error
        """
        try:
            # Validate inputs (Commandment #7: Check parameters)
            self.validate_num_topics(num_topics)

            # Initialize context
            ctx = self.initialize_context(project, require_storage=True)

            # Get LLM client (optional for enhanced analysis)
            llm_client = self.get_llm_client(ctx)

            # Retrieve chunks
            chunks = self.get_all_chunks_from_storage(ctx["storage"])

            if not chunks:
                self._handle_no_chunks()
                return 0

            # Analyze topics
            topic_data = self._analyze_topics(chunks, num_topics, llm_client)

            # Display results
            self._display_topics(topic_data)

            # Save to file if requested
            if output:
                self._save_topic_analysis(output, topic_data)

            return 0

        except Exception as e:
            return self.handle_error(e, "Topic analysis failed")

    def validate_num_topics(self, num_topics: int) -> None:
        """Validate number of topics.

        Args:
            num_topics: Number to validate

        Raises:
            typer.BadParameter: If invalid
        """
        import typer

        if num_topics < 1:
            raise typer.BadParameter("Number of topics must be at least 1")

        if num_topics > 50:
            raise typer.BadParameter("Number of topics cannot exceed 50")

    def _handle_no_chunks(self) -> None:
        """Handle case where no chunks found."""
        self.print_warning("Knowledge base is empty")
        self.print_info("Try ingesting some documents first")

    def _analyze_topics(
        self, chunks: list, num_topics: int, llm_client: Optional[Any]
    ) -> Dict[str, Any]:
        """Analyze topics in chunks.

        Args:
            chunks: List of chunks
            num_topics: Number of topics to identify
            llm_client: Optional LLM client

        Returns:
            Topic analysis data
        """
        # Basic keyword extraction
        basic_topics = self._extract_basic_topics(chunks, num_topics)

        # Enhanced analysis with LLM if available
        if llm_client:
            enhanced_analysis = self._get_llm_topic_analysis(
                chunks, basic_topics, llm_client
            )
        else:
            enhanced_analysis = None

        return {
            "num_chunks": len(chunks),
            "num_topics": num_topics,
            "basic_topics": basic_topics,
            "enhanced_analysis": enhanced_analysis,
        }

    def _extract_basic_topics(
        self, chunks: list, num_topics: int
    ) -> List[Dict[str, Any]]:
        """Extract basic topics using keyword analysis.

        Args:
            chunks: List of chunks
            num_topics: Number of topics

        Returns:
            List of topic dictionaries
        """
        # Combine all text
        all_text = " ".join(self.extract_chunk_text(chunk) for chunk in chunks)

        # Extract keywords
        keywords = self.extract_keywords_simple(all_text, num_topics * 3)

        # Group into topics (simplified clustering)
        topics = []

        for i in range(min(num_topics, len(keywords))):
            # Get keyword and related chunks
            keyword = keywords[i]
            related_chunks = self._find_chunks_with_keyword(chunks, keyword)

            topics.append(
                {
                    "keyword": keyword,
                    "chunk_count": len(related_chunks),
                    "relevance": (len(related_chunks) / len(chunks)) * 100,
                }
            )

        return sorted(topics, key=lambda x: x["relevance"], reverse=True)

    def _find_chunks_with_keyword(self, chunks: list, keyword: str) -> list[Any]:
        """Find chunks containing keyword.

        Args:
            chunks: List of chunks
            keyword: Keyword to search for

        Returns:
            List of matching chunks
        """
        matching = []

        for chunk in chunks:
            text = self.extract_chunk_text(chunk).lower()
            if keyword in text:
                matching.append(chunk)

        return matching

    def _get_llm_topic_analysis(
        self, chunks: list, basic_topics: list, llm_client: Any
    ) -> str:
        """Get enhanced topic analysis from LLM using hierarchical approach.

        Analyzes each document separately, then synthesizes across all documents
        to provide comprehensive coverage of the entire corpus.

        Args:
            chunks: List of chunks
            basic_topics: Basic topic data
            llm_client: LLM client

        Returns:
            Enhanced analysis text
        """
        # Group chunks by source document
        by_source = self.group_chunks_by_source(chunks)
        num_docs = len(by_source)

        self.print_info(f"Analyzing {num_docs} documents...")

        # Step 1: Summarize each document
        doc_summaries = self._summarize_documents(by_source, llm_client)

        if not doc_summaries:
            # Fallback to simple sampling if summarization fails
            return self._fallback_simple_analysis(chunks, basic_topics, llm_client)

        # Step 2: Synthesize across all document summaries
        return self._synthesize_corpus_analysis(
            doc_summaries, basic_topics, llm_client, len(chunks)
        )

    def _summarize_documents(
        self, by_source: Dict[str, list], llm_client: Any
    ) -> List[Dict[str, Any]]:
        """Summarize main topics from each document.

        Args:
            by_source: Chunks grouped by source document
            llm_client: LLM client

        Returns:
            List of document summaries
        """
        summaries = []

        for source, doc_chunks in by_source.items():
            try:
                # Sample representative chunks from this document
                sample = self._sample_from_document(doc_chunks, max_samples=15)
                context = self.format_context_for_prompt(sample, max_length=2500)

                # Get document summary
                prompt = (
                    f"Document: {source}\n"
                    f"Chunks: {len(doc_chunks)}\n\n"
                    f"Content excerpts:\n{context}\n\n"
                    "List the 3-5 main topics covered in this document. "
                    "Be specific and concise. Format as bullet points."
                )

                summary = self._generate_content(llm_client, prompt)

                summaries.append(
                    {
                        "source": source,
                        "chunk_count": len(doc_chunks),
                        "summary": summary.strip(),
                    }
                )

            except Exception as e:
                self.print_warning(f"Failed to summarize {source}: {e}")
                continue

        return summaries

    def _sample_from_document(self, chunks: list, max_samples: int = 15) -> list:
        """Sample representative chunks from a single document.

        Args:
            chunks: Document chunks
            max_samples: Maximum samples

        Returns:
            Representative sample
        """
        if len(chunks) <= max_samples:
            return chunks

        # Sample from beginning, middle, and end
        step = len(chunks) // max_samples
        return [chunks[min(i * step, len(chunks) - 1)] for i in range(max_samples)]

    def _synthesize_corpus_analysis(
        self,
        doc_summaries: List[Dict[str, Any]],
        basic_topics: List[Dict[str, Any]],
        llm_client: Any,
        total_chunks: int,
    ) -> str:
        """Synthesize analysis across all document summaries.

        Args:
            doc_summaries: Summaries from each document
            basic_topics: Keyword-based topics
            llm_client: LLM client
            total_chunks: Total number of chunks in corpus

        Returns:
            Comprehensive corpus analysis
        """
        # Build summary context
        summary_text = "\n\n".join(
            f"**{s['source']}** ({s['chunk_count']} chunks):\n{s['summary']}"
            for s in doc_summaries
        )

        keyword_list = ", ".join(t["keyword"] for t in basic_topics[:15])

        prompt = (
            f"Corpus Analysis: {len(doc_summaries)} documents, {total_chunks} chunks\n\n"
            f"Document summaries:\n{summary_text}\n\n"
            f"Top keywords by frequency: {keyword_list}\n\n"
            "Provide a comprehensive analysis:\n"
            "1. **Main Topics and Themes** - What are the major themes across all documents?\n"
            "2. **How Topics Relate** - How do different documents connect thematically?\n"
            "3. **Key Insights** - What are the most important takeaways?\n"
            "4. **Suggested Categories** - How would you organize this content?\n\n"
            "Format as clear markdown."
        )

        try:
            return self.generate_with_llm(llm_client, prompt, "corpus synthesis")
        except Exception as e:
            self.print_warning(f"Synthesis failed: {e}")
            return ""

    def _fallback_simple_analysis(
        self, chunks: list, basic_topics: list, llm_client: Any
    ) -> str:
        """Fallback to simple sampling if hierarchical analysis fails.

        Args:
            chunks: All chunks
            basic_topics: Keyword-based topics
            llm_client: LLM client

        Returns:
            Simple analysis text
        """
        sample_chunks = self._sample_chunks_representatively(chunks, max_samples=30)
        context = self.format_context_for_prompt(sample_chunks, max_length=4000)
        prompt = self._build_topic_analysis_prompt(context, basic_topics)

        try:
            return self.generate_with_llm(llm_client, prompt, "topic analysis")
        except Exception as e:
            self.print_warning(f"Fallback analysis failed: {e}")
            return ""

    def _sample_chunks_representatively(
        self, chunks: list, max_samples: int = 30
    ) -> list:
        """Sample chunks representatively across all documents.

        Args:
            chunks: All chunks
            max_samples: Maximum number of samples

        Returns:
            Representative sample of chunks
        """
        if len(chunks) <= max_samples:
            return chunks

        # Group by source document
        by_source = self.group_chunks_by_source(chunks)

        # Calculate samples per source
        num_sources = len(by_source)
        samples_per_source = max(1, max_samples // num_sources)

        sample_chunks = []
        for source, source_chunks in by_source.items():
            sample_chunks.extend(
                self._sample_from_document(source_chunks, samples_per_source)
            )

        return sample_chunks[:max_samples]

    def _build_topic_analysis_prompt(
        self, context: str, basic_topics: list[Any]
    ) -> str:
        """Build prompt for topic analysis.

        Args:
            context: Context from chunks
            basic_topics: Basic topic data

        Returns:
            Formatted prompt
        """
        topic_list = ", ".join(t["keyword"] for t in basic_topics[:10])

        prompt_parts = [
            "Analyze the main topics and themes in this content.\n\n",
            "Sample content:\n",
            context,
            "\n\nKeywords identified:\n",
            topic_list,
            "\n\nProvide:\n",
            "1. Main topics and themes\n",
            "2. How topics relate to each other\n",
            "3. Key insights about content focus\n",
            "4. Suggested categorization\n\n",
            "Format as clear markdown.",
        ]

        return "".join(prompt_parts)

    def _display_topics(self, topic_data: Dict[str, Any]) -> None:
        """Display topic analysis.

        Args:
            topic_data: Topic analysis data
        """
        self.console.print()

        # Summary
        self.print_info(
            f"Analyzed {topic_data['num_chunks']} chunks "
            f"for top {topic_data['num_topics']} topics"
        )

        self.console.print()

        # Basic topics table
        table = Table(title="Top Topics by Keyword Frequency")
        table.add_column("Rank", style="cyan", width=6)
        table.add_column("Topic/Keyword", style="green", width=20)
        table.add_column("Chunks", style="yellow", width=10)
        table.add_column("Coverage", style="magenta", width=12)

        for idx, topic in enumerate(topic_data["basic_topics"], 1):
            table.add_row(
                str(idx),
                topic["keyword"],
                str(topic["chunk_count"]),
                f"{topic['relevance']:.1f}%",
            )

        self.console.print(table)

        # Enhanced analysis if available
        if topic_data["enhanced_analysis"]:
            self.console.print()

            panel = Panel(
                Markdown(topic_data["enhanced_analysis"]),
                title="[bold cyan]Enhanced Topic Analysis[/bold cyan]",
                border_style="cyan",
            )

            self.console.print(panel)

    def _save_topic_analysis(self, output: Path, topic_data: Dict[str, Any]) -> None:
        """Save topic analysis to file.

        Args:
            output: Output file path
            topic_data: Topic data
        """
        try:
            from datetime import datetime

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            lines = [
                "# Topic Analysis\n\n",
                f"**Generated:** {timestamp}\n",
                f"**Chunks Analyzed:** {topic_data['num_chunks']}\n",
                f"**Topics Identified:** {topic_data['num_topics']}\n\n",
                "---\n\n",
                "## Top Topics\n\n",
            ]

            # Add basic topics
            for idx, topic in enumerate(topic_data["basic_topics"], 1):
                lines.append(
                    f"{idx}. **{topic['keyword']}** - "
                    f"{topic['chunk_count']} chunks "
                    f"({topic['relevance']:.1f}% coverage)\n"
                )

            lines.append("\n")

            # Add enhanced analysis if available
            if topic_data["enhanced_analysis"]:
                lines.append("## Enhanced Analysis\n\n")
                lines.append(topic_data["enhanced_analysis"])
                lines.append("\n")

            output.write_text("".join(lines), encoding="utf-8")
            self.print_success(f"Topic analysis saved to: {output}")

        except Exception as e:
            self.print_warning(f"Failed to save analysis: {e}")


# Typer command wrapper
def command(
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for analysis"
    ),
    num_topics: int = typer.Option(
        10, "--topics", "-n", help="Number of top topics to identify"
    ),
) -> None:
    """Analyze and extract topics from knowledge base.

    Identifies main topics, themes, and subject areas in your
    ingested content using keyword analysis and optional LLM
    enhancement.

    Examples:
        # Analyze top 10 topics
        ingestforge analyze topics

        # Analyze top 20 topics
        ingestforge analyze topics --topics 20

        # Save analysis
        ingestforge analyze topics --output topics.md

        # Specific project
        ingestforge analyze topics -p /path/to/project
    """
    cmd = TopicsCommand()
    exit_code = cmd.execute(project, output, num_topics)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
