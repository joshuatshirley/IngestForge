"""Ingest command - Process documents into the knowledge base.

Processes documents through the full pipeline: extraction → chunking →
enrichment → indexing.

Web URL Connector - Epic (CLI URL Ingestion)
Timestamp: 2026-02-18 20:30 UTC

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
and #1 (Simple Control Flow).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import typer

from ingestforge.cli.core import (
    IngestForgeCommand,
    IngestResult,
)
from ingestforge.enrichment.embeddings import VRAMInfo, MemoryInfo


def is_youtube_url(input_str: str) -> bool:
    """Check if input is a YouTube URL.

    Args:
        input_str: Input string to check

    Returns:
        True if input is a YouTube URL, False otherwise

    Rule #7: Parameter validation
    Rule #1: Early return
    """
    if not input_str:
        return False

    input_str = input_str.strip().lower()

    # Check for HTTP URLs containing YouTube domains
    if input_str.startswith("http"):
        return "youtube.com" in input_str or "youtu.be" in input_str

    return False


def is_web_url(input_str: str) -> bool:
    """Check if input is a web URL (non-YouTube).

    Epic (CLI URL Detection)
    Timestamp: 2026-02-18 20:30 UTC

    Args:
        input_str: Input string to check

    Returns:
        True if input is a web URL (excluding YouTube), False otherwise

    Rule #7: Parameter validation
    Rule #1: Early return
    """
    if not input_str:
        return False

    input_str = input_str.strip().lower()

    # Must start with http:// or https://
    if not (input_str.startswith("http://") or input_str.startswith("https://")):
        return False

    # Exclude YouTube URLs (handled separately)
    if "youtube.com" in input_str or "youtu.be" in input_str:
        return False

    return True


class IngestCommand(IngestForgeCommand):
    """Process documents into knowledge base."""

    # Minimum RAM (GB) required for parallel processing
    MIN_RAM_FOR_PARALLEL: float = 4.0
    # Minimum VRAM (GB) if GPU is available
    MIN_VRAM_FOR_PARALLEL: float = 2.0

    def execute(
        self,
        path: Path,
        project: Optional[Path] = None,
        recursive: bool = False,
        skip_errors: bool = False,
        limit: Optional[int] = None,
        dry_run: bool = False,
        quiet: bool = False,
        youtube_url: Optional[str] = None,
        workers: Optional[int] = None,
        url: Optional[str] = None,
        url_list: Optional[Path] = None,
        headers: Optional[List[str]] = None,
    ) -> int:
        """Process documents through ingestion pipeline.

        Epic (CLI URL Ingestion)
        Timestamp: 2026-02-18 20:30 UTC

        Args:
            path: Document file, directory, or YouTube URL
            project: Project directory
            recursive: Recursively process directories
            skip_errors: Continue on errors instead of stopping
            limit: Maximum number of files to process (useful for testing)
            dry_run: Show what would be processed without actually processing
            quiet: Suppress progress bar output (UX-003)
            youtube_url: YouTube URL passed directly (avoids Windows Path conversion)
            workers: Number of parallel workers ()
            url: Single web URL to ingest ()
            url_list: File containing URLs to batch ingest ()
            headers: Custom HTTP headers in "Name: Value" format ()

        Returns:
            0 on success, 1 on error
        """
        try:
            # Parse custom headers (Epic )
            custom_headers = self._parse_headers(headers) if headers else {}

            # Handle URL list batch processing (Epic )
            if url_list:
                return self._process_url_list(
                    url_list, project, dry_run, quiet, custom_headers, skip_errors
                )

            # Handle single URL ingestion (Epic )
            if url:
                return self._process_web_url(
                    url, project, dry_run, quiet, custom_headers
                )

            # Handle YouTube URL passed directly (avoids Windows Path backslash issue)
            if youtube_url:
                return self._process_youtube_url(youtube_url, project, dry_run, quiet)

            # Check if input is a web URL (Epic )
            path_str = str(path)
            if is_web_url(path_str):
                return self._process_web_url(
                    path_str, project, dry_run, quiet, custom_headers
                )

            # Check if input is a YouTube URL (Commandment #7: Check parameters)
            if is_youtube_url(path_str):
                return self._process_youtube_url(path_str, project, dry_run, quiet)

            # Validate file path
            path = self.validate_file_path(path, must_exist=True, must_be_file=False)

            # Initialize with pipeline (skip for dry run)
            ctx = None
            if not dry_run:
                ctx = self.initialize_context(project, require_pipeline=True)

            # Get list of files to process
            files = self._gather_files(path, recursive)

            if not files:
                self._handle_no_files(path)
                return 0

            # Apply limit if specified
            if limit is not None and limit > 0:
                files = files[:limit]
                self.print_info(f"Limited to first {limit} files")

            # Dry run mode - show what would be processed
            if dry_run:
                self._display_dry_run(files)
                return 0

            # Check memory availability for parallel processing
            use_parallel = self._check_memory_for_parallel()

            # Process files (workers flag support)
            result = self._process_files(
                files,
                ctx["pipeline"],
                skip_errors,
                parallel=use_parallel,
                quiet=quiet,
                max_workers=workers,
            )

            # Display results
            self._display_results(result)

            return 0 if result.success else 1

        except Exception as e:
            return self.handle_error(e, "Ingestion failed")

    def _check_memory_for_parallel(self) -> bool:
        """Check if enough memory is available for parallel processing.

        Returns:
            True if parallel processing is safe, False to use sequential
        """
        # Check VRAM if GPU is available
        vram_info = VRAMInfo.detect()
        if vram_info.available:
            if vram_info.free_gb < self.MIN_VRAM_FOR_PARALLEL:
                self.print_warning(
                    f"Low VRAM ({vram_info.free_gb:.1f} GB free). "
                    "Using sequential processing to avoid OOM."
                )
                return False

        # Check system RAM
        mem_info = MemoryInfo.detect()
        if mem_info.available_gb < self.MIN_RAM_FOR_PARALLEL:
            self.print_warning(
                f"Low RAM ({mem_info.available_gb:.1f} GB free). "
                "Using sequential processing to avoid OOM."
            )
            return False

        return True

    def _process_youtube_url(
        self,
        url: str,
        project: Optional[Path] = None,
        dry_run: bool = False,
        quiet: bool = False,
    ) -> int:
        """Process a YouTube URL.

        Args:
            url: YouTube URL to process
            project: Project directory
            dry_run: Show what would be processed without processing
            quiet: Suppress progress output

        Returns:
            0 on success, 1 on error

        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        """
        from ingestforge.ingest.youtube import (
            is_youtube_available,
            extract_video_id,
        )

        # Check if youtube-transcript-api is available
        if not is_youtube_available():
            self.print_error(
                "youtube-transcript-api not installed.\n"
                "Install with: pip install youtube-transcript-api"
            )
            return 1

        # Extract video ID for validation
        video_id = extract_video_id(url)
        if not video_id:
            self.print_error(f"Invalid YouTube URL: {url}")
            return 1

        # Dry run mode
        if dry_run:
            self.console.print()
            self.console.print("[bold cyan]Dry Run Mode[/bold cyan]")
            self.console.print(f"Would process YouTube video: {video_id}")
            self.console.print(f"URL: https://www.youtube.com/watch?v={video_id}")
            self.console.print()
            return 0

        # Initialize context
        ctx = self.initialize_context(project, require_pipeline=True)

        # Process YouTube URL
        return self._ingest_youtube(url, video_id, ctx["pipeline"], quiet)

    def _ingest_youtube(
        self,
        url: str,
        video_id: str,
        pipeline: Any,
        quiet: bool = False,
    ) -> int:
        """Ingest YouTube video transcript.

        Args:
            url: YouTube URL
            video_id: Extracted video ID
            pipeline: Pipeline instance
            quiet: Suppress progress output

        Returns:
            0 on success, 1 on error

        Rule #4: Function <60 lines
        """
        from ingestforge.ingest.youtube import YouTubeProcessor

        if not quiet:
            self.print_info(f"Processing YouTube video: {video_id}")

        # Create processor and extract transcript
        processor = YouTubeProcessor()
        result = processor.process(url)

        # Check for errors
        if not result.success:
            self.print_error(f"Failed to process video: {result.error}")
            return 1

        # Convert to ChunkRecords
        document_id = f"youtube_{video_id}"
        chunk_records = processor.to_chunk_records(result, document_id)

        if not chunk_records:
            self.print_warning("No transcript chunks created")
            return 1

        # Store chunks in pipeline
        self._store_youtube_chunks(chunk_records, result, pipeline, quiet)

        return 0

    def _store_youtube_chunks(
        self,
        chunk_records: List,
        result: Any,
        pipeline: Any,
        quiet: bool,
    ) -> None:
        """Store YouTube chunks and display results.

        Args:
            chunk_records: ChunkRecord objects to store
            result: YouTubeProcessingResult
            pipeline: Pipeline instance
            quiet: Suppress progress output

        Rule #4: Function <60 lines
        """
        # Display progress
        if not quiet:
            self.print_info(
                f"Created {len(chunk_records)} chunks "
                f"from {result.duration_seconds:.0f}s video"
            )

        # Store chunks
        for record in chunk_records:
            pipeline.storage.add_chunk(record)

        # Display success
        if not quiet:
            self.console.print()
            self.console.print("[bold green]Success![/bold green]")
            self.console.print(f"Video: {result.metadata.video_url}")
            self.console.print(f"Chunks: {len(chunk_records)}")
            self.console.print(f"Words: {result.word_count}")

    def _parse_headers(self, header_list: List[str]) -> Dict[str, str]:
        """Parse custom HTTP headers from "Name: Value" format.

        Epic (Custom Headers)
        Timestamp: 2026-02-18 20:30 UTC

        Args:
            header_list: List of header strings in "Name: Value" format

        Returns:
            Dictionary of header name -> value

        Rule #4: Function <60 lines
        Rule #7: Validates input format
        """
        headers: Dict[str, str] = {}

        for header_str in header_list:
            # Split on first colon only
            if ":" not in header_str:
                self.print_warning(
                    f"Invalid header format (missing colon): {header_str}\n"
                    "Expected format: 'Name: Value'"
                )
                continue

            name, _, value = header_str.partition(":")
            name = name.strip()
            value = value.strip()

            if not name or not value:
                self.print_warning(
                    f"Invalid header (empty name or value): {header_str}"
                )
                continue

            headers[name] = value

        return headers

    def _process_web_url(
        self,
        url: str,
        project: Optional[Path] = None,
        dry_run: bool = False,
        quiet: bool = False,
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """Process a single web URL.

        Epic (CLI URL Ingestion)
        Timestamp: 2026-02-18 20:30 UTC

        Args:
            url: Web URL to ingest
            project: Project directory
            dry_run: Show what would be processed without processing
            quiet: Suppress progress output
            custom_headers: Custom HTTP headers

        Returns:
            0 on success, 1 on error

        Rule #4: Function <60 lines
        Rule #7: Parameter validation with url.py
        """
        from ingestforge.core.security.url import validate_url

        # Validate URL for SSRF (Epic )
        is_valid, error_msg = validate_url(url)
        if not is_valid:
            self.print_error(f"Invalid URL: {error_msg}")
            return 1

        # Dry run mode
        if dry_run:
            self.console.print()
            self.console.print("[bold cyan]Dry Run Mode[/bold cyan]")
            self.console.print(f"Would process URL: {url}")
            if custom_headers:
                self.console.print(f"Custom headers: {len(custom_headers)}")
            self.console.print()
            return 0

        # Initialize context
        ctx = self.initialize_context(project, require_pipeline=True)

        # Process URL
        return self._ingest_web_url(url, ctx["pipeline"], quiet, custom_headers)

    def _ingest_web_url(
        self,
        url: str,
        pipeline: Any,
        quiet: bool = False,
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> int:
        """Ingest web URL content.

        Epic (CLI URL Ingestion)
        Timestamp: 2026-02-18 20:30 UTC

        Args:
            url: Web URL to fetch
            pipeline: Pipeline instance
            quiet: Suppress progress output
            custom_headers: Custom HTTP headers

        Returns:
            0 on success, 1 on error

        Rule #4: Function <60 lines
        """
        from ingestforge.ingest.connectors.web_scraper import WebScraperConnector
        from ingestforge.core.pipeline.artifacts import IFFailureArtifact

        if not quiet:
            self.print_info(f"Processing URL: {url}")

        # Create connector and initialize with custom headers
        connector = WebScraperConnector()

        # Build config with custom headers
        config: Dict[str, Any] = {}
        if custom_headers:
            config["headers"] = custom_headers

        # Connect to initialize session
        if not connector.connect(config):
            self.print_error("Failed to initialize web connector")
            return 1

        try:
            # Fetch URL content as artifact
            artifact = connector.fetch_to_artifact(url)

            # Check if fetch failed
            if isinstance(artifact, IFFailureArtifact):
                self.print_error(f"Failed to fetch URL: {artifact.error_message}")
                return 1

            # Process artifact through pipeline
            pipeline_result = pipeline.process_artifact(artifact)

            if not quiet:
                self.console.print()
                self.console.print("[bold green]Success![/bold green]")
                self.console.print(f"URL: {url}")
                self.console.print(f"Chunks: {pipeline_result.chunks_created}")

            return 0

        finally:
            # Clean up connector
            connector.disconnect()

    def _process_url_list(
        self,
        url_file: Path,
        project: Optional[Path] = None,
        dry_run: bool = False,
        quiet: bool = False,
        custom_headers: Optional[Dict[str, str]] = None,
        skip_errors: bool = False,
    ) -> int:
        """Process batch URLs from file.

        Epic (Batch URL Processing)
        Timestamp: 2026-02-18 20:30 UTC

        Args:
            url_file: File containing URLs (one per line)
            project: Project directory
            dry_run: Show what would be processed
            quiet: Suppress progress output
            custom_headers: Custom HTTP headers
            skip_errors: Continue on errors

        Returns:
            0 on success, 1 on error

        Rule #4: Function <60 lines
        Rule #7: Validates file existence
        """

        # Validate file path
        url_file = self.validate_file_path(url_file, must_exist=True, must_be_file=True)

        # Read URLs from file (Rule #2: Bounded read)
        try:
            urls = url_file.read_text(encoding="utf-8").strip().splitlines()
        except Exception as e:
            self.print_error(f"Failed to read URL file: {e}")
            return 1

        # Filter empty lines and comments
        urls = [u.strip() for u in urls if u.strip() and not u.strip().startswith("#")]

        if not urls:
            self.print_warning("No URLs found in file")
            return 0

        # Dry run mode
        if dry_run:
            self.console.print()
            self.console.print("[bold cyan]Dry Run Mode[/bold cyan]")
            self.console.print(f"Would process {len(urls)} URLs from {url_file.name}")
            # Show first 10 URLs (Rule #2: Bounded)
            max_display = min(10, len(urls))
            for url in urls[:max_display]:
                self.console.print(f"  - {url}")
            if len(urls) > max_display:
                self.console.print(f"  ... and {len(urls) - max_display} more")
            self.console.print()
            return 0

        # Initialize context
        ctx = self.initialize_context(project, require_pipeline=True)

        # Process URLs with progress tracking
        return self._batch_ingest_urls(
            urls, ctx["pipeline"], quiet, custom_headers, skip_errors
        )

    def _batch_ingest_urls(
        self,
        urls: List[str],
        pipeline: Any,
        quiet: bool,
        custom_headers: Optional[Dict[str, str]],
        skip_errors: bool,
    ) -> int:
        """Batch ingest multiple URLs.

        Epic (Batch Processing)
        Timestamp: 2026-02-18 20:30 UTC

        Args:
            urls: List of URLs to process
            pipeline: Pipeline instance
            quiet: Suppress progress
            custom_headers: Custom HTTP headers
            skip_errors: Continue on errors

        Returns:
            0 on success, 1 on error

        Rule #4: Function <60 lines
        Rule #2: Bounded iteration
        """
        from ingestforge.core.security.url import validate_url

        success_count = 0
        error_count = 0
        errors: List[str] = []

        # Process each URL (Rule #2: Bounded by list size)
        for idx, url in enumerate(urls, 1):
            if not quiet:
                self.print_info(f"Processing URL {idx}/{len(urls)}: {url}")

            # Validate URL
            is_valid, error_msg = validate_url(url)
            if not is_valid:
                error_msg_full = f"{url}: {error_msg}"
                errors.append(error_msg_full)
                error_count += 1
                if not skip_errors:
                    self.print_error(error_msg_full)
                    return 1
                continue

            # Process URL
            try:
                result = self._ingest_web_url(
                    url, pipeline, quiet=True, custom_headers=custom_headers
                )
                if result == 0:
                    success_count += 1
                else:
                    error_count += 1
                    if not skip_errors:
                        return 1
            except Exception as e:
                error_msg_full = f"{url}: {str(e)}"
                errors.append(error_msg_full)
                error_count += 1
                if not skip_errors:
                    return 1

        # Display summary
        if not quiet:
            self.console.print()
            self.console.print("[bold]Batch Processing Complete[/bold]")
            self.console.print(f"Success: {success_count}/{len(urls)}")
            if error_count > 0:
                self.console.print(f"[yellow]Errors: {error_count}[/yellow]")
                self._display_errors(errors)

        return 0 if error_count == 0 else 1

    def _display_dry_run(self, files: List[Path]) -> None:
        """Display files that would be processed in dry run mode.

        Args:
            files: List of files that would be processed
        """
        self.console.print()
        self.console.print(
            "[bold cyan]Dry Run Mode[/bold cyan] - No files will be processed"
        )
        self.console.print()
        self.console.print(f"[bold]Would process {len(files)} file(s):[/bold]")
        self.console.print()

        # Group by extension for summary
        by_ext: dict[str, List[Path]] = {}
        for f in files:
            ext = f.suffix.lower() or "(no extension)"
            by_ext.setdefault(ext, []).append(f)

        # Show extension summary
        for ext, ext_files in sorted(by_ext.items()):
            self.console.print(f"  {ext}: {len(ext_files)} file(s)")

        self.console.print()

        # Show first 20 files (Commandment #2: Fixed upper bound)
        max_display = min(20, len(files))
        for f in files[:max_display]:
            self.console.print(f"    - {f.name}")

        if len(files) > max_display:
            remaining = len(files) - max_display
            self.console.print(f"    ... and {remaining} more")

        self.console.print()

    def _gather_files(self, path: Path, recursive: bool) -> List[Path]:
        """Gather list of files to process.

        Args:
            path: File or directory path
            recursive: Whether to recurse into subdirectories

        Returns:
            List of file paths to process
        """
        if path.is_file():
            return [path]

        # Directory - find all processable files
        return self._find_documents(path, recursive)

    def _find_documents(self, directory: Path, recursive: bool) -> List[Path]:
        """Find all processable documents in directory.

        Args:
            directory: Directory to search
            recursive: Whether to recurse into subdirectories

        Returns:
            List of document paths
        """
        # Supported extensions (Commandment #6: Smallest scope)
        extensions = {
            ".txt",
            ".md",
            ".markdown",
            ".mdown",
            ".pdf",
            ".docx",
            ".epub",
            ".html",
            ".htm",
            # LaTeX
            ".tex",
            ".latex",
            ".ltx",
            # Jupyter
            ".ipynb",
            # Audio (TICKET-201)
            ".mp3",
            ".wav",
            ".m4a",
            ".flac",
            ".ogg",
            ".webm",
        }

        files = []

        # Use rglob for recursive, glob for non-recursive
        pattern = "**/*" if recursive else "*"

        # Find all files with supported extensions (Commandment #2: Bounded)
        for file_path in directory.glob(pattern):
            if file_path.is_file() and file_path.suffix.lower() in extensions:
                files.append(file_path)

        return sorted(files)

    def _handle_no_files(self, path: Path) -> None:
        """Handle case where no processable files found.

        Args:
            path: Path that was searched
        """
        self.print_warning(f"No processable documents found in: {path}")
        self.print_info(
            "Supported formats:\n"
            "  - Text: .txt, .md, .markdown, .mdown\n"
            "  - Documents: .pdf, .docx, .epub\n"
            "  - Web: .html, .htm\n"
            "  - LaTeX: .tex, .latex, .ltx\n"
            "  - Notebooks: .ipynb"
        )

    def _process_files(
        self,
        files: List[Path],
        pipeline: Any,
        skip_errors: bool,
        parallel: bool = True,
        quiet: bool = False,
        max_workers: Optional[int] = None,
    ) -> IngestResult:
        """Process list of files through pipeline.

        Args:
            files: List of file paths
            pipeline: Pipeline instance
            skip_errors: Whether to continue on errors
            parallel: Enable parallel processing (default: True)
            quiet: Suppress progress bar output (UX-003)
            max_workers: Number of parallel workers ()

        Returns:
            IngestResult with statistics
        """
        from ingestforge.cli.commands.ingest_progress import run_ingestion_with_progress

        # Use clean progress display with parallel processing ()
        progress_result = run_ingestion_with_progress(
            files,
            pipeline,
            skip_errors,
            parallel=parallel,
            max_workers=max_workers,
            quiet=quiet,
        )

        # Convert to IngestResult
        result = IngestResult()
        result.documents_ingested = progress_result["files_processed"]
        result.chunks_created = progress_result["chunks_created"]
        result.items_processed = progress_result["files_processed"]
        result.items_failed = len(progress_result["errors"])
        result.errors = progress_result["errors"]
        result.success = progress_result["success"]

        return result

    def _process_single_file(
        self, file_path: Path, pipeline: Any, result: IngestResult, skip_errors: bool
    ) -> bool:
        """Process a single file through pipeline.

        Args:
            file_path: File to process
            pipeline: Pipeline instance
            result: Result accumulator
            skip_errors: Whether to continue on errors

        Returns:
            True if successful, False if failed
        """
        try:
            # Process file (Commandment #7: Check inputs validated by pipeline)
            pipeline_result = pipeline.process_file(file_path)

            # Update result statistics
            result.documents_ingested += 1
            result.chunks_created += pipeline_result.chunks_created
            result.items_processed += 1

            return True

        except Exception as e:
            # Record error
            result.errors.append(f"{file_path.name}: {str(e)}")
            result.items_failed += 1

            if not skip_errors:
                raise  # Re-raise to stop processing

            return False

    def _display_results(self, result: IngestResult) -> None:
        """Display ingestion results.

        Args:
            result: IngestResult with statistics
        """
        # Summary already shown by progress display
        # Only show additional info if needed
        if result.documents_ingested > 0 and result.success:
            avg_chunks = result.avg_chunks_per_document
            self.print_info(f"Average: {avg_chunks:.1f} chunks per document")

    def _display_errors(self, errors: List[str]) -> None:
        """Display error list.

        Args:
            errors: List of error messages
        """
        self.console.print()
        self.print_warning(f"Encountered {len(errors)} errors:")

        # Show first 10 errors (Commandment #2: Fixed upper bound)
        max_display = min(10, len(errors))
        for error in errors[:max_display]:
            self.console.print(f"  • {error}")

        if len(errors) > max_display:
            remaining = len(errors) - max_display
            self.console.print(f"  ... and {remaining} more")


# Typer command wrapper
def command(
    path_or_url: str = typer.Argument(..., help="Document file, directory, or URL"),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    recursive: bool = typer.Option(
        False, "--recursive", "-r", help="Recursively process directories"
    ),
    skip_errors: bool = typer.Option(
        False, "--skip-errors", help="Continue processing on errors"
    ),
    limit: Optional[int] = typer.Option(
        None,
        "--limit",
        "-n",
        help="Process only the first N files (useful for testing)",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would be processed without actually processing",
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress progress bar output"
    ),
    workers: Optional[int] = typer.Option(
        None,
        "--workers",
        "-w",
        help="Number of parallel workers (default: cpu_count - 1)",
    ),
    url: Optional[str] = typer.Option(
        None, "--url", help="Single web URL to ingest ()"
    ),
    url_list: Optional[Path] = typer.Option(
        None, "--url-list", help="File containing URLs to batch ingest ()"
    ),
    headers: Optional[List[str]] = typer.Option(
        None,
        "--header",
        help="Custom HTTP header in 'Name: Value' format (can be repeated)",
    ),
) -> None:
    """Process documents, URLs, or YouTube videos into the knowledge base.

    Epic (CLI URL Ingestion)
    Timestamp: 2026-02-18 20:30 UTC

    Processes documents through the full pipeline:
    1. Extract text and metadata
    2. Split into semantic chunks
    3. Generate embeddings
    4. Index in vector database

    Supported formats:
    - Text: .txt, .md, .markdown, .mdown
    - Documents: .pdf, .docx, .epub
    - Web: .html, .htm
    - LaTeX: .tex, .latex, .ltx
    - Notebooks: .ipynb
    - YouTube: URLs (https://youtube.com/watch?v=... or https://youtu.be/...)
    - Web URLs: Any HTTP/HTTPS URL ()

    Examples:
        # Process single file
        ingestforge ingest document.pdf

        # Process YouTube video
        ingestforge ingest https://youtube.com/watch?v=dQw4w9WgXcQ

        # Process web URL ()
        ingestforge ingest --url https://example.com/article

        # Process web URL with custom headers ()
        ingestforge ingest --url https://example.com/api \
            --header "Authorization: Bearer token" \
            --header "User-Agent: IngestForge/1.0"

        # Batch process URLs from file ()
        ingestforge ingest --url-list urls.txt

        # Directory ingestion
        ingestforge ingest documents/

        # Process directory recursively
        ingestforge ingest documents/ --recursive

        # Continue on errors
        ingestforge ingest documents/ --recursive --skip-errors

        # Process only first 5 files (for testing)
        ingestforge ingest documents/ --limit 5

        # Show what would be processed
        ingestforge ingest documents/ --dry-run

        # Process for specific project
        ingestforge ingest docs/ --project /path/to/project

        # Suppress progress bar (for CI/scripts)
        ingestforge ingest documents/ --quiet

        # Use 4 parallel workers ()
        ingestforge ingest documents/ --workers 4
    """
    cmd = IngestCommand()

    # Handle URL-specific modes ()
    if url or url_list:
        exit_code = cmd.execute(
            Path("."),  # Dummy path, ignored for URL mode
            project=project,
            recursive=recursive,
            skip_errors=skip_errors,
            limit=limit,
            dry_run=dry_run,
            quiet=quiet,
            workers=workers,
            url=url,
            url_list=url_list,
            headers=headers,
        )
    # Handle YouTube URLs as strings to avoid Windows Path backslash conversion
    # (Rule #7: Parameter validation before conversion)
    elif is_youtube_url(path_or_url):
        exit_code = cmd.execute(
            Path("."),  # Dummy path, will be ignored for YouTube
            project,
            recursive,
            skip_errors,
            limit,
            dry_run,
            quiet,
            youtube_url=path_or_url,  # Pass URL separately
            workers=workers,  # Pass workers
        )
    # Handle web URLs ()
    elif is_web_url(path_or_url):
        exit_code = cmd.execute(
            Path("."),  # Dummy path, will be ignored for URL
            project=project,
            recursive=recursive,
            skip_errors=skip_errors,
            limit=limit,
            dry_run=dry_run,
            quiet=quiet,
            workers=workers,
            url=path_or_url,
            headers=headers,
        )
    else:
        exit_code = cmd.execute(
            Path(path_or_url),
            project,
            recursive,
            skip_errors,
            limit,
            dry_run,
            quiet,
            workers=workers,  # Pass workers
        )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
