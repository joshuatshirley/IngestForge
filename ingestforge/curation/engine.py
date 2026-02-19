"""
Curation engine implementing the state machine for web curation workflow.

Handles the Search → Preview → Decision → Ingest/Skip cycle with proper
state transitions and error handling.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional
from datetime import datetime, timezone

from ingestforge.curation.models import (
    CurationState,
    CurationDecision,
    CurationItem,
    CurationSession,
    CurationSessionManager,
)


@dataclass
class PreviewResult:
    """Result of previewing a URL."""

    success: bool
    item: CurationItem
    error: Optional[str] = None


@dataclass
class IngestResult:
    """Result of ingesting a URL."""

    success: bool
    item: CurationItem
    chunks_created: int = 0
    document_id: Optional[str] = None
    error: Optional[str] = None


class CurationEngine:
    """
    State machine engine for web curation workflow.

    Manages the lifecycle of a curation session:
    1. Start session with search query
    2. Preview each URL in the queue
    3. Wait for user decision (ingest/skip)
    4. Process decision and advance queue
    5. Complete when queue is exhausted or user quits

    Example:
        manager = CurationSessionManager(project_path)
        engine = CurationEngine(manager, project_path)

        # Start new session
        session = engine.start_session("python tutorials", max_results=10)

        # Curation loop
        while not engine.is_complete:
            preview = engine.preview_current()
            if preview.success:
                print(f"Title: {preview.item.title}")
                print(f"Preview: {preview.item.preview_text}")
                # User decides...
                if user_wants_ingest:
                    result = engine.ingest_current()
                else:
                    engine.skip_current()
            else:
                engine.skip_current()  # Skip failed fetches

        print(engine.get_summary())
    """

    def __init__(
        self,
        session_manager: CurationSessionManager,
        project_path: Optional[Path] = None,
        progress_callback: Optional[Callable[[str, str], None]] = None,
        verbose: bool = False,
        fetch_timeout: int = 60,
        process_timeout: int = 300,
    ):
        """
        Initialize the curation engine.

        Args:
            session_manager: Manager for session persistence
            project_path: Project directory for pipeline operations
            progress_callback: Optional callback(stage, message) for progress updates
            verbose: Enable verbose logging for debugging
            fetch_timeout: Timeout in seconds for URL fetching (default 60s)
            process_timeout: Timeout in seconds for pipeline processing (default 300s)
        """
        self.session_manager = session_manager
        self.project_path = project_path or Path.cwd()
        self.progress_callback = progress_callback
        self.verbose = verbose
        self.fetch_timeout = fetch_timeout
        self.process_timeout = process_timeout
        self._session: Optional[CurationSession] = None

    @property
    def session(self) -> Optional[CurationSession]:
        """Current curation session."""
        return self._session

    @property
    def current_item(self) -> Optional[CurationItem]:
        """Current item in the queue."""
        if self._session:
            return self._session.current_item
        return None

    @property
    def is_complete(self) -> bool:
        """Check if the session is complete."""
        if not self._session:
            return True
        return self._session.state == CurationState.COMPLETE

    def _notify(self, stage: str, message: str) -> None:
        """Send progress notification."""
        if self.progress_callback:
            self.progress_callback(stage, message)

    def _set_state(self, state: CurationState) -> None:
        """Update session state and persist."""
        if self._session:
            self._session.state = state
            self.session_manager.save(self._session)

    def start_session(
        self,
        query: str,
        max_results: int = 20,
        academic: bool = False,
    ) -> CurationSession:
        """
        Start a new curation session with a web search.

        Rule #4: Reduced from 69 lines to <60 lines via helper extraction

        Args:
            query: Search query string
            max_results: Maximum number of results to fetch
            academic: If True, prioritize educational/academic sources

        Returns:
            New CurationSession with search results

        Raises:
            ValueError: If query is empty
            RuntimeError: If search fails
        """
        if not query or not query.strip():
            raise ValueError("Search query cannot be empty")
        self._create_new_session(query, academic)
        try:
            self._perform_search(query, max_results, academic)
            return self._session
        except ImportError as e:
            self._set_state(CurationState.COMPLETE)
            raise ImportError(
                "duckduckgo-search is required for web search. "
                "Install with: pip install duckduckgo-search"
            ) from e
        except Exception as e:
            self._set_state(CurationState.COMPLETE)
            raise RuntimeError(f"Search failed: {e}") from e

    def _create_new_session(self, query: str, academic: bool) -> None:
        """
        Create and initialize new curation session.

        Rule #4: Extracted to reduce function size
        """
        session_id = self.session_manager.generate_session_id()
        self._session = CurationSession(
            id=session_id,
            query=query.strip(),
            state=CurationState.SEARCHING,
            academic_mode=academic,
        )
        self.session_manager.save(self._session)
        self._notify("search", f"Searching for: {query}")

    def _perform_search(self, query: str, max_results: int, academic: bool) -> None:
        """
        Execute search and populate session with results.

        Rule #4: Extracted to reduce function size
        """
        from ingestforge.ingest.web_search import WebSearcher

        searcher = WebSearcher()

        if academic:
            search_session = searcher.search_academic(query, max_results=max_results)
        else:
            search_session = searcher.search(query, max_results=max_results)

        # Convert results to curation items
        for result in search_session.results:
            item = CurationItem.from_search_result(result)
            self._session.items.append(item)

        self._notify("search", f"Found {len(self._session.items)} results")

        # Transition to previewing if we have results
        if self._session.items:
            self._set_state(CurationState.PREVIEWING)
        else:
            self._set_state(CurationState.COMPLETE)

    def resume_session(self, session_id: str) -> Optional[CurationSession]:
        """
        Resume an existing session.

        Args:
            session_id: ID of session to resume

        Returns:
            CurationSession if found and resumable, None otherwise
        """
        session = self.session_manager.load(session_id)
        if session and session.state != CurationState.COMPLETE:
            self._session = session
            # Restore to previewing state if it was waiting
            if session.state in (
                CurationState.WAITING_FOR_DECISION,
                CurationState.SEARCHING,
            ):
                self._set_state(CurationState.PREVIEWING)
            return session
        return None

    def _log(self, msg: str) -> None:
        """Log message if verbose mode is enabled."""
        import sys

        if self.verbose:
            print(f"[curation] {msg}", file=sys.stderr, flush=True)

    def preview_current(self) -> PreviewResult:
        """
        Fetch and preview the current item in the queue.

        Returns:
            PreviewResult with item data and preview text

        Raises:
            RuntimeError: If no session is active
        """
        if not self._session:
            raise RuntimeError("No active session")

        item = self._session.current_item
        if not item:
            self._set_state(CurationState.COMPLETE)
            return PreviewResult(
                success=False,
                item=CurationItem(
                    id="",
                    url="",
                    title="",
                    snippet="",
                ),
                error="No more items in queue",
            )

        self._set_state(CurationState.PREVIEWING)
        self._notify("preview", f"Fetching: {item.url}")
        self._log(f"preview_current: Fetching {item.url}")

        try:
            result = self._fetch_item_content(item)
            self._update_item_preview(item, result)

            self._set_state(CurationState.WAITING_FOR_DECISION)
            self._notify("preview", f"Ready: {item.title}")

            return PreviewResult(success=True, item=item)

        except Exception as e:
            return self._handle_preview_error(item, e)

    def _fetch_item_content(self, item: Any):
        """Fetch and process item URL content."""
        self._log("preview_current: Importing HTMLProcessor...")
        from ingestforge.ingest.html_processor import HTMLProcessor
        from ingestforge.cli.core.add import (
            run_with_timeout,
            TimeoutError as AddTimeoutError,
        )

        self._log("preview_current: Creating HTMLProcessor...")
        processor = HTMLProcessor()

        self._log(
            f"preview_current: Calling process_url (timeout: {self.fetch_timeout}s)..."
        )

        def _fetch():
            return processor.process_url(item.url)

        try:
            result = run_with_timeout(_fetch, self.fetch_timeout)
        except AddTimeoutError:
            raise Exception(f"URL fetch timed out after {self.fetch_timeout}s")

        self._log(
            f"preview_current: Got {len(result.text) if result.text else 0} chars"
        )
        return result

    def _update_item_preview(self, item: Any, result: Any) -> None:
        """Update item with preview data from fetch result."""
        preview_text = result.text[:500] if result.text else ""
        item.preview_text = preview_text.strip()
        item.word_count = len(result.text.split()) if result.text else 0
        item.fetch_error = None

    def _handle_preview_error(self, item: Any, error: Exception) -> PreviewResult:
        """Handle preview fetch error."""
        item.fetch_error = str(error)
        item.preview_text = None
        item.word_count = None

        self._set_state(CurationState.SKIPPING_ERROR)
        self._notify("error", f"Fetch failed: {error}")

        return PreviewResult(success=False, item=item, error=str(error))

    def ingest_current(self) -> IngestResult:
        """
        Ingest the current item through the pipeline.

        Returns:
            IngestResult with processing outcome

        Raises:
            RuntimeError: If no session is active or item not ready
        """
        if not self._session:
            raise RuntimeError("No active session")

        item = self._session.current_item
        if not item:
            return IngestResult(
                success=False,
                item=CurationItem(id="", url="", title="", snippet=""),
                error="No current item",
            )

        self._set_state(CurationState.INGESTING)
        self._notify("ingest", f"Ingesting: {item.url}")

        try:
            result = self._ingest_via_pipeline(item)
            self._update_item_after_ingest(item, result)
            self._advance_queue()

            self._notify("ingest", f"Ingested {item.chunks_created} chunks")

            return IngestResult(
                success=True,
                item=item,
                chunks_created=item.chunks_created,
                document_id=item.document_id,
            )

        except Exception as e:
            return self._handle_ingest_error(item, e)

    def _ingest_via_pipeline(self, item: Any) -> dict[str, Any]:
        """Run item through ingestion pipeline."""
        from ingestforge.cli.core.add import add_url

        self._log(
            f"ingest_current: Calling add_url (fetch_timeout={self.fetch_timeout}s, process_timeout={self.process_timeout}s)"
        )
        return add_url(
            item.url,
            project_path=self.project_path,
            verbose=self.verbose,
            fetch_timeout=self.fetch_timeout,
            process_timeout=self.process_timeout,
        )

    def _update_item_after_ingest(self, item: Any, result: dict) -> None:
        """Update item with ingestion results."""
        item.decision = CurationDecision.INGEST
        item.chunks_created = result.get("chunks_created", 0)
        item.document_id = result.get("document_id")
        item.processed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        self._session.total_ingested += 1

    def _handle_ingest_error(self, item: Any, error: Exception) -> IngestResult:
        """Handle ingestion error."""
        item.fetch_error = str(error)
        self._set_state(CurationState.SKIPPING_ERROR)
        self._notify("error", f"Ingest failed: {error}")

        return IngestResult(success=False, item=item, error=str(error))

    def skip_current(self, reason: str = "user_skip") -> bool:
        """
        Skip the current item and advance the queue.

        Args:
            reason: Reason for skipping (for logging)

        Returns:
            True if successfully skipped, False if no item to skip
        """
        if not self._session:
            return False

        item = self._session.current_item
        if not item:
            self._set_state(CurationState.COMPLETE)
            return False

        # Mark item as skipped
        if item.fetch_error:
            item.decision = CurationDecision.SKIP_ERROR
            self._session.total_errors += 1
        else:
            item.decision = CurationDecision.SKIP
            self._session.total_skipped += 1

        item.processed_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        self._set_state(CurationState.SKIPPING)
        self._notify("skip", f"Skipped: {item.title}")

        # Advance to next item
        self._advance_queue()

        return True

    def _advance_queue(self) -> None:
        """Advance to the next item in the queue."""
        if not self._session:
            return

        self._session.current_index += 1

        if self._session.is_complete:
            self._set_state(CurationState.COMPLETE)
            self._notify("complete", "Curation session complete")
        else:
            self._set_state(CurationState.PREVIEWING)

        self.session_manager.save(self._session)

    def quit_session(self) -> None:
        """End the current session early."""
        if self._session:
            self._set_state(CurationState.COMPLETE)
            self._notify("complete", "Session ended by user")

    def get_summary(self) -> dict[str, Any]:
        """
        Get a summary of the current session.

        Returns:
            Dict with session statistics
        """
        if not self._session:
            return {"error": "No active session"}

        return {
            "session_id": self._session.id,
            "query": self._session.query,
            "state": self._session.state.value,
            "total_items": len(self._session.items),
            "current_index": self._session.current_index,
            "remaining": self._session.remaining_count,
            "total_ingested": self._session.total_ingested,
            "total_skipped": self._session.total_skipped,
            "total_errors": self._session.total_errors,
            "academic_mode": self._session.academic_mode,
            "created_at": self._session.created_at,
            "updated_at": self._session.updated_at,
        }

    def get_queue_status(self) -> list[Any]:
        """
        Get status of all items in the queue.

        Returns:
            List of dicts with item status
        """
        if not self._session:
            return []

        return [
            {
                "index": i,
                "id": item.id,
                "title": item.title[:50] + "..."
                if len(item.title) > 50
                else item.title,
                "domain": item.domain,
                "decision": item.decision.value,
                "is_current": i == self._session.current_index,
                "chunks_created": item.chunks_created,
            }
            for i, item in enumerate(self._session.items)
        ]
