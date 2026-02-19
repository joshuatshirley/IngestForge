"""
Directory watcher for automatic document ingestion.

Monitors the .ingest/pending directory for new documents.
"""

import threading
import time
from pathlib import Path
from typing import Any, Callable, Optional, Set

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class DirectoryWatcher:
    """
    Watch directory for new documents and trigger processing.

    Uses watchdog library for cross-platform file system events,
    with fallback to polling if watchdog is unavailable.
    """

    def __init__(
        self,
        config: Config,
        on_new_file: Callable[[Path], None],
    ):
        """
        Initialize watcher.

        Args:
            config: IngestForge configuration
            on_new_file: Callback when new file is detected
        """
        self.config = config
        self.on_new_file = on_new_file
        self.watch_dir = config.pending_path
        self.supported_formats = set(config.ingest.supported_formats)

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._processed_files: Set[Path] = set()
        self._observer = None

    def start(self) -> None:
        """Start watching for new files."""
        if self._running:
            return

        self._running = True
        self.watch_dir.mkdir(parents=True, exist_ok=True)

        # Try watchdog first
        if self._try_watchdog():
            logger.info(f"Watching {self.watch_dir} (using watchdog)")
        else:
            # Fallback to polling
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()
            logger.info(f"Watching {self.watch_dir} (using polling)")

    def stop(self) -> None:
        """Stop watching."""
        self._running = False

        if self._observer:
            self._observer.stop()
            self._observer.join()
            self._observer = None

        if self._thread:
            self._thread.join(timeout=2)
            self._thread = None

        logger.info("Watcher stopped")

    def _try_watchdog(self) -> bool:
        """Try to use watchdog for file system events."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler

            class Handler(FileSystemEventHandler):
                def __init__(handler_self: Any, watcher: Any) -> None:
                    handler_self.watcher = watcher

                def on_created(handler_self: Any, event: Any) -> None:
                    if event.is_directory:
                        return
                    path = Path(event.src_path)
                    handler_self.watcher._handle_new_file(path)

                def on_moved(handler_self: Any, event: Any) -> None:
                    if event.is_directory:
                        return
                    path = Path(event.dest_path)
                    handler_self.watcher._handle_new_file(path)

            self._observer = Observer()
            handler = Handler(self)
            self._observer.schedule(handler, str(self.watch_dir), recursive=False)
            self._observer.start()
            return True

        except ImportError:
            logger.warning("watchdog not installed, using polling fallback")
            return False

    def _poll_loop(self) -> None:
        """Polling fallback when watchdog unavailable."""
        interval = self.config.ingest.watch_interval_sec

        while self._running:
            try:
                self._scan_directory()
            except Exception as e:
                logger.error(f"Error scanning directory: {e}")

            time.sleep(interval)

    def _scan_directory(self) -> None:
        """Scan directory for new files."""
        for file_path in self.watch_dir.iterdir():
            if file_path.is_file():
                self._handle_new_file(file_path)

    def _handle_new_file(self, file_path: Path) -> None:
        """Handle a potentially new file."""
        # Skip if already processed
        if file_path in self._processed_files:
            return

        # Skip unsupported formats
        if file_path.suffix.lower() not in self.supported_formats:
            return

        # Skip temporary/partial files
        if file_path.name.startswith("."):
            return
        if file_path.name.endswith(".tmp"):
            return
        if file_path.name.endswith(".part"):
            return

        # Wait briefly to ensure file is fully written
        try:
            initial_size = file_path.stat().st_size
            time.sleep(0.5)
            if file_path.exists():
                final_size = file_path.stat().st_size
                if final_size != initial_size:
                    # File still being written
                    return
        except Exception:
            return

        self._processed_files.add(file_path)
        logger.info(f"New file detected: {file_path.name}")

        try:
            self.on_new_file(file_path)
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")

    def _is_processable_file(self, file_path: Path) -> bool:
        """
        Check if file is processable.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            file_path: Path to check

        Returns:
            True if file should be processed
        """
        if not file_path.is_file():
            return False
        if file_path.suffix.lower() not in self.supported_formats:
            return False
        if file_path in self._processed_files:
            return False

        return True

    def _process_file_with_error_handling(self, file_path: Path) -> None:
        """
        Process file with error handling.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            file_path: File to process
        """
        self._processed_files.add(file_path)
        logger.info(f"Found existing file: {file_path.name}")

        try:
            self.on_new_file(file_path)
        except Exception as e:
            logger.error(f"Error processing {file_path.name}: {e}")

    def process_existing(self) -> None:
        """
        Process any existing files in the watch directory.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints
        """
        logger.info(f"Scanning {self.watch_dir} for existing files")
        for file_path in self.watch_dir.iterdir():
            if self._is_processable_file(file_path):
                self._process_file_with_error_handling(file_path)


def watch_and_process(
    config: Config,
    processor_callback: Callable[[Path], None],
    process_existing: bool = True,
) -> None:
    """
    Convenience function to watch and process files.

    Args:
        config: IngestForge configuration
        processor_callback: Called for each new file
        process_existing: Process existing files on startup

    Returns:
        DirectoryWatcher instance (call .stop() to stop watching)
    """
    watcher = DirectoryWatcher(config, processor_callback)

    if process_existing:
        watcher.process_existing()

    watcher.start()
    return watcher
