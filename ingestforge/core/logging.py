"""
Structured Logging for IngestForge.

This module provides a logging infrastructure that supports context binding,
specialized loggers for different processing stages, and consistent formatting
across the entire application.

Architecture Context
--------------------
Logging is a Core layer service used by every module in the system. All modules
should import get_logger() from here rather than using Python's logging directly:

    # Good - uses IngestForge's structured logging
    from ingestforge.core.logging import get_logger
    logger = get_logger(__name__)

    # Avoid - bypasses our structure
    import logging
    logger = logging.getLogger(__name__)

Logger Types
------------
**StructuredLogger**
    Base logger with context binding support. Allows attaching key-value pairs
    that appear in all subsequent log messages:

        logger = get_logger(__name__)
        logger.bind(document_id="doc_123")
        logger.info("Processing started")  # includes document_id

**PipelineLogger**
    Specialized for document processing pipelines. Tracks stages (split, extract,
    chunk, enrich, index) with timing and progress:

        plog = PipelineLogger(document_id)
        plog.start_stage("chunk")
        plog.log_progress("Created 42 chunks")
        plog.finish(success=True, chunks=42)

Module-Level Factory
--------------------
The get_logger() function provides cached logger instances:

    logger = get_logger("ingestforge.chunking.semantic")

Loggers are cached by name, so multiple calls return the same instance.
This is the recommended entry point for all logging needs.

Design Decisions
----------------
1. **Context binding**: Avoids repetitive passing of IDs to every log call.
2. **Stage-aware logging**: Pipeline stages are first-class concepts.
3. **Timing built-in**: Duration tracking without manual instrumentation.
4. **Lazy initialization**: Loggers configured on first use, not import.
"""

import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


@dataclass
class LogConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    file_path: Optional[Path] = None
    console: bool = True


class StructuredLogger:
    """
    Structured logger with context support.

    Provides consistent logging across the application with
    support for structured fields and context tracking.
    """

    def __init__(self, name: str, config: Optional[LogConfig] = None) -> None:
        self.logger = logging.getLogger(name)
        self.config = config or LogConfig()
        self._context: dict[str, Any] = {}
        self._setup_logger()

    def _setup_logger(self) -> None:
        """Configure the logger."""
        level = getattr(logging, self.config.level.upper(), logging.INFO)
        self.logger.setLevel(level)

        # Remove existing handlers
        self.logger.handlers.clear()

        formatter = logging.Formatter(
            self.config.format,
            datefmt=self.config.date_format,
        )

        # Console handler - use Rich if available for better terminal coordination
        if self.config.console:
            try:
                from rich.logging import RichHandler

                console_handler = RichHandler(
                    show_time=True,
                    show_path=False,
                    markup=True,
                    rich_tracebacks=True,
                )
                console_handler.setLevel(level)
                # RichHandler has its own formatting
            except ImportError:
                console_handler = logging.StreamHandler(sys.stdout)
                console_handler.setLevel(level)
                console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)

        # File handler
        if self.config.file_path:
            self.config.file_path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(self.config.file_path)
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)

    def _format_message(self, message: str, **kwargs: Any) -> str:
        """Format message with context and extra fields."""
        fields = {**self._context, **kwargs}
        if fields:
            field_str = " | ".join(f"{k}={v}" for k, v in fields.items())
            return f"{message} | {field_str}"
        return message

    def debug(self, message: str, **kwargs: Any) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(message, **kwargs))

    def info(self, message: str, **kwargs: Any) -> None:
        """Log info message."""
        self.logger.info(self._format_message(message, **kwargs))

    def warning(self, message: str, **kwargs: Any) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(message, **kwargs))

    def error(self, message: str, **kwargs: Any) -> None:
        """Log error message."""
        self.logger.error(self._format_message(message, **kwargs))

    def exception(self, message: str, **kwargs: Any) -> None:
        """Log exception with traceback."""
        self.logger.exception(self._format_message(message, **kwargs))


# Module-level logger factory
_loggers: dict[str, StructuredLogger] = {}


def get_logger(name: str, config: Optional[LogConfig] = None) -> StructuredLogger:
    """
    Get or create a structured logger.

    Args:
        name: Logger name (typically __name__).
        config: Optional logging configuration.

    Returns:
        Configured StructuredLogger instance.
    """
    if name not in _loggers:
        _loggers[name] = StructuredLogger(name, config)
    return _loggers[name]


def configure_logging(
    level: str = "INFO",
    log_file: Optional[Path] = None,
    console: bool = True,
) -> None:
    """
    Configure global logging settings.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_file: Optional file path for log output.
        console: Whether to log to console.
    """
    config = LogConfig(
        level=level,
        file_path=log_file,
        console=console,
    )

    # Update root logger
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    _ConfigHolder.set_config(config)


class _ConfigHolder:
    """Holds default logging configuration.

    Rule #6: Encapsulates singleton state in smallest scope.
    """

    _config: LogConfig = LogConfig()

    @classmethod
    def get_config(cls) -> LogConfig:
        """Get the default config."""
        return cls._config

    @classmethod
    def set_config(cls, config: LogConfig) -> None:
        """Set the default config."""
        cls._config = config


class PipelineLogger:
    """
    Specialized logger for pipeline operations.

    Tracks processing stages and provides timing information.
    """

    def __init__(self, document_id: str) -> None:
        self.document_id = document_id
        self.logger = get_logger("ingestforge.pipeline")
        self._stage_start: Optional[datetime] = None
        self._current_stage: Optional[str] = None

    def start_stage(self, stage: str) -> None:
        """Mark the start of a processing stage."""
        self._finish_current_stage()
        self._current_stage = stage
        self._stage_start = datetime.now()
        self.logger.info(
            "Starting stage",
            document_id=self.document_id,
            stage=stage,
        )

    def _finish_current_stage(self) -> None:
        """Log completion of current stage if any."""
        if self._current_stage and self._stage_start:
            duration = (datetime.now() - self._stage_start).total_seconds()
            self.logger.info(
                "Completed stage",
                document_id=self.document_id,
                stage=self._current_stage,
                duration_sec=f"{duration:.2f}",
            )

    def finish(
        self, success: bool, chunks: int = 0, error: Optional[str] = None
    ) -> None:
        """Mark pipeline completion."""
        self._finish_current_stage()
        if success:
            self.logger.info(
                "Pipeline completed successfully",
                document_id=self.document_id,
                chunks_created=chunks,
            )
        else:
            self.logger.error(
                "Pipeline failed",
                document_id=self.document_id,
                error=error,
            )

    def log_progress(self, message: str, **kwargs: Any) -> None:
        """Log progress within a stage."""
        self.logger.debug(
            message,
            document_id=self.document_id,
            stage=self._current_stage,
            **kwargs,
        )
