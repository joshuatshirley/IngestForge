"""
Utility Methods Mixin for Pipeline.

Provides utility methods for:
- File hashing and duplicate detection
- Document ID generation
- Library path extraction
- State initialization and finalization
- File movement and cleanup
- Pipeline reset

This module is part of the Pipeline refactoring (Sprint 3, Rule #4)
to reduce pipeline.py from 1680 lines to <400 lines.
"""

import hashlib
import time
from pathlib import Path

from ingestforge.core.logging import PipelineLogger
from ingestforge.core.security import PathTraversalError
from ingestforge.core.state import DocumentState, ProcessingState, ProcessingStatus


class _Logger:
    """Lazy logger holder.

    Rule #6: Encapsulates logger state in smallest scope.
    Avoids slow startup from rich library import.
    """

    _instance = None

    @classmethod
    def get(cls):
        """Get logger (lazy-loaded)."""
        if cls._instance is None:
            from ingestforge.core.logging import get_logger

            cls._instance = get_logger(__name__)
        return cls._instance


class PipelineUtilsMixin:
    """
    Mixin providing utility methods for Pipeline.

    Rule #4: Extracted from pipeline.py to reduce file size
    """

    def _report_progress(self, stage: str, progress: float, message: str = "") -> None:
        """Report progress through callback."""
        if self._progress_callback:
            self._progress_callback(stage, progress, message)
        _Logger.get().info(f"[{stage}] {progress:.0%} - {message}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA-256 hash of file."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()[:16]

    def _get_hash_store(self):
        """Get or create the content hash store for dedup checks.

        Returns a HashStore backed by .data/content_hashes.json, or None
        if the content_hash_verifier module is unavailable.
        """
        if not hasattr(self, "_hash_store"):
            try:
                from ingestforge.ingest.content_hash_verifier import HashStore

                store_path = self.config.data_path / "content_hashes.json"
                self._hash_store = HashStore(store_path)
            except Exception as e:
                _Logger.get().debug(f"Content hash store unavailable: {e}")
                self._hash_store = None
        return self._hash_store

    def _is_duplicate_content(self, file_path: Path, document_id: str) -> bool:
        """Check if file content has already been processed.

        Uses the hash store to detect identical files and verifies
        that the previous processing completed successfully via state manager.
        Returns False (allow reprocessing) if hash store is unavailable.
        """
        try:
            store = self._get_hash_store()
            if store is None:
                return False

            record = store.get(str(file_path))
            if record is None:
                return False

            # Also check that the document was successfully processed
            doc = self.state_manager.state.get_document(document_id)
            if doc and doc.status == ProcessingStatus.COMPLETED:
                return True

            return False
        except Exception:
            _Logger.get().exception(
                f"Error checking duplicate content for {file_path}: "
                "Returning False to allow reprocessing"
            )
            return False

    def _store_content_hash(self, file_path: Path) -> None:
        """Store content hash after successful processing."""
        try:
            from ingestforge.ingest.content_hash_verifier import hash_content

            store = self._get_hash_store()
            if store is None:
                return
            hashes = hash_content(file_path)
            store.store(str(file_path), hashes)
        except Exception as e:
            _Logger.get().debug(f"Could not store content hash: {e}")

    def _generate_document_id(self, file_path: Path) -> str:
        """Generate unique document ID from file."""
        file_hash = self._compute_file_hash(file_path)
        name_part = file_path.stem[:30].replace(" ", "_")
        return f"{name_part}_{file_hash}"

    def _extract_library_from_path(self, file_path: Path) -> str:
        """
        Extract library name from folder structure.

        Documents in .ingest/pending/<library-name>/ are tagged with that library.
        Documents directly in .ingest/pending/ get library="default".

        Args:
            file_path: Path to the document being processed.

        Returns:
            Library name string.
        """
        try:
            relative = file_path.relative_to(self.config.pending_path)
            parts = relative.parts
            if len(parts) > 1:
                # First subfolder is the library name
                return parts[0]
        except ValueError as e:
            # File not under pending_path
            _Logger.get().debug(
                f"File {file_path} not under pending_path, using default library: {e}"
            )
        return "default"

    def _initialize_document_processing(
        self, file_path: Path, document_id: str, doc_state: DocumentState
    ) -> PipelineLogger:
        """
        Initialize document processing state and logging.

        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            file_path: Document to process
            document_id: Unique document ID
            doc_state: State tracker

        Returns:
            Initialized PipelineLogger

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            # SEC-002: Sanitize path disclosure
            logger.error(f"Document not found: {file_path}")
            raise FileNotFoundError("Document not found: [REDACTED]")

        plog = PipelineLogger(document_id)
        doc_state.start_processing()
        doc_state.file_size_bytes = file_path.stat().st_size
        doc_state.file_hash = self._compute_file_hash(file_path)

        return plog

    def _finalize_document_processing(
        self,
        document_id: str,
        file_path: Path,
        all_chunks: list,
        indexed_count: int,
        doc_state: DocumentState,
        plog: PipelineLogger,
        start_time: float,
    ):
        """
        Finalize document processing and return result.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            document_id: Unique document ID
            file_path: Document path
            all_chunks: All created chunks
            indexed_count: Number of indexed chunks
            doc_state: State tracker
            plog: Pipeline logger
            start_time: Start timestamp

        Returns:
            PipelineResult with statistics
        """
        from ingestforge.core.pipeline import PipelineResult

        doc_state.complete()
        processing_time = time.time() - start_time
        plog.finish(success=True, chunks=len(all_chunks))

        return PipelineResult(
            document_id=document_id,
            source_file=str(file_path),
            success=True,
            chunks_created=len(all_chunks),
            chunks_indexed=indexed_count,
            processing_time_sec=processing_time,
        )

    def _move_to_completed(self, file_path: Path, document_id: str):
        """Move processed file to completed directory with path validation."""
        completed_dir = self.config.completed_path
        dest_path = completed_dir / f"{document_id}_{file_path.name}"

        # Check if file is in pending path or a subfolder of pending path (library folder)
        def is_in_pending_tree(path: Path) -> bool:
            try:
                path.relative_to(self.config.pending_path)
                return True
            except ValueError:
                _Logger.get().debug(
                    f"Path {path} is not in pending tree {self.config.pending_path}"
                )
                return False

        # Only move files from allowed directories (pending tree or processing)
        if is_in_pending_tree(file_path):
            try:
                self._safe_ops.safe_move(file_path, dest_path)
            except PathTraversalError as e:
                _Logger.get().warning(f"Path traversal blocked: {e}")
        elif file_path.parent == self.config.processing_path:
            try:
                self._safe_ops.safe_move(file_path, dest_path)
            except PathTraversalError as e:
                _Logger.get().warning(f"Path traversal blocked: {e}")

    def reset(self, confirm: bool = False) -> None:
        """
        Reset all processed data.

        Rule #1: Reduced nesting with helper method
        Rule #4: Function <60 lines

        Args:
            confirm: Must be True to actually reset.
        """
        if not confirm:
            raise ValueError("Must pass confirm=True to reset pipeline")

        # Clear storage
        self.storage.clear()

        # Reset state
        self.state_manager.state = ProcessingState(
            project_name=self.config.project.name
        )
        self.state_manager.save()
        self._clear_data_directories()

        _Logger.get().info("Pipeline reset complete")

    def _clear_data_directories(self) -> None:
        """
        Clear all data subdirectories.

        Rule #1: Extracted to reduce nesting
        Rule #4: Function <60 lines
        """
        for subdir in ["chunks", "embeddings", "index"]:
            self._clear_directory(self.config.data_path / subdir)

    def _clear_directory(self, data_dir: Path) -> None:
        """
        Clear files in a directory.

        Rule #1: Reduced nesting with guard clause
        Rule #4: Function <60 lines
        """
        if not data_dir.exists():
            return

        for f in data_dir.iterdir():
            if f.is_file():
                f.unlink()
