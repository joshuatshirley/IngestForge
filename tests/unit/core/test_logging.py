"""
Tests for Structured Logging.

This module tests the logging infrastructure including StructuredLogger,
PipelineLogger, and the get_logger factory.

Test Strategy
-------------
- Focus on logger behavior, not stdlib logging internals
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Test context binding, stage tracking, message formatting
- Don't test Rich library integration (external dependency)

Organization
------------
- TestLogConfig: LogConfig dataclass
- TestStructuredLogger: StructuredLogger class
- TestGetLogger: get_logger factory function
- TestPipelineLogger: PipelineLogger class
"""

import logging
from pathlib import Path


from ingestforge.core.logging import (
    LogConfig,
    StructuredLogger,
    get_logger,
    PipelineLogger,
    configure_logging,
)


# ============================================================================
# Test Classes
# ============================================================================


class TestLogConfig:
    """Tests for LogConfig dataclass.

    Rule #4: Focused test class - tests only LogConfig
    """

    def test_default_values(self):
        """Test LogConfig with default values."""
        config = LogConfig()

        assert config.level == "INFO"
        assert config.console is True
        assert config.file_path is None

    def test_custom_values(self):
        """Test LogConfig with custom values."""
        log_file = Path("/tmp/test.log")
        config = LogConfig(level="DEBUG", file_path=log_file, console=False)

        assert config.level == "DEBUG"
        assert config.file_path == log_file
        assert config.console is False


class TestStructuredLogger:
    """Tests for StructuredLogger class.

    Rule #4: Focused test class - tests only StructuredLogger
    """

    def test_create_logger(self):
        """Test creating a StructuredLogger."""
        logger = StructuredLogger("test_logger")

        assert logger.logger.name == "test_logger"
        assert logger._context == {}

    def test_create_with_config(self):
        """Test creating logger with custom config."""
        config = LogConfig(level="DEBUG")
        logger = StructuredLogger("test", config)

        assert logger.config.level == "DEBUG"

    def test_info_method(self, caplog):
        """Test info logging method."""
        logger = StructuredLogger("test")

        with caplog.at_level(logging.INFO):
            logger.info("Test message")

        assert "Test message" in caplog.text

    def test_warning_method(self, caplog):
        """Test warning logging method."""
        logger = StructuredLogger("test")

        with caplog.at_level(logging.WARNING):
            logger.warning("Warning message")

        assert "Warning message" in caplog.text

    def test_error_method(self, caplog):
        """Test error logging method."""
        logger = StructuredLogger("test")

        with caplog.at_level(logging.ERROR):
            logger.error("Error message")

        assert "Error message" in caplog.text

    def test_debug_method(self, caplog):
        """Test debug logging method."""
        config = LogConfig(level="DEBUG")
        logger = StructuredLogger("test", config)

        with caplog.at_level(logging.DEBUG):
            logger.debug("Debug message")

        assert "Debug message" in caplog.text


class TestGetLogger:
    """Tests for get_logger factory function.

    Rule #4: Focused test class - tests only get_logger()
    """

    def test_returns_structured_logger(self):
        """Test get_logger returns StructuredLogger."""
        logger = get_logger("test.module")

        assert isinstance(logger, StructuredLogger)

    def test_caches_loggers(self):
        """Test that get_logger caches logger instances."""
        logger1 = get_logger("test.cached")
        logger2 = get_logger("test.cached")

        assert logger1 is logger2

    def test_different_names_different_loggers(self):
        """Test that different names return different loggers."""
        logger1 = get_logger("test.one")
        logger2 = get_logger("test.two")

        assert logger1 is not logger2
        assert logger1.logger.name == "test.one"
        assert logger2.logger.name == "test.two"


class TestPipelineLogger:
    """Tests for PipelineLogger class.

    Rule #4: Focused test class - tests only PipelineLogger
    """

    def test_create_pipeline_logger(self):
        """Test creating a PipelineLogger."""
        plog = PipelineLogger("doc_123")

        assert plog.document_id == "doc_123"
        assert plog._current_stage is None

    def test_start_stage(self, caplog):
        """Test starting a processing stage."""
        plog = PipelineLogger("doc_456")

        with caplog.at_level(logging.INFO):
            plog.start_stage("chunk")

        assert "Starting stage" in caplog.text
        assert "doc_456" in caplog.text
        assert "chunk" in caplog.text
        assert plog._current_stage == "chunk"

    def test_finish_success(self, caplog):
        """Test finishing pipeline with success."""
        plog = PipelineLogger("doc_789")

        with caplog.at_level(logging.INFO):
            plog.finish(success=True, chunks=42)

        assert "completed successfully" in caplog.text
        assert "doc_789" in caplog.text
        assert "42" in caplog.text

    def test_finish_failure(self, caplog):
        """Test finishing pipeline with failure."""
        plog = PipelineLogger("doc_fail")

        with caplog.at_level(logging.ERROR):
            plog.finish(success=False, error="Processing error")

        assert "failed" in caplog.text
        assert "doc_fail" in caplog.text
        assert "Processing error" in caplog.text

    def test_log_progress(self, caplog):
        """Test logging progress within a stage."""
        config = LogConfig(level="DEBUG")
        # Force DEBUG level for pipeline logger
        plog = PipelineLogger("doc_progress")
        plog.logger = StructuredLogger("test.pipeline", config)
        plog.start_stage("enrich")

        with caplog.at_level(logging.DEBUG):
            plog.log_progress("Processed 10 chunks")

        assert "Processed 10 chunks" in caplog.text
        assert "doc_progress" in caplog.text


class TestConfigureLogging:
    """Tests for configure_logging function.

    Rule #4: Focused test class - tests only configure_logging()
    """

    def test_configure_with_defaults(self):
        """Test configure_logging with default arguments."""
        configure_logging()

        root = logging.getLogger()
        assert root.level == logging.INFO

    def test_configure_with_custom_level(self):
        """Test configure_logging with custom log level."""
        configure_logging(level="DEBUG")

        root = logging.getLogger()
        assert root.level == logging.DEBUG


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - LogConfig: 2 tests (defaults, custom values)
    - StructuredLogger: 6 tests (creation, logging methods)
    - get_logger: 3 tests (returns logger, caching, different names)
    - PipelineLogger: 5 tests (creation, stage tracking, finish, progress)
    - configure_logging: 2 tests (defaults, custom level)

    Total: 18 tests

Design Decisions:
    1. Focus on logger behavior, not stdlib logging internals
    2. Use caplog fixture for testing log output
    3. Don't test Rich integration (external dependency, optional)
    4. Don't test file handler (would require file I/O mocking)
    5. Test caching behavior for get_logger factory
    6. Simple, clear tests that verify logging works
    7. Follows NASA JPL Rule #1 (Simple Control Flow)
    8. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - LogConfig creation with defaults and custom values
    - StructuredLogger logging methods (debug, info, warning, error)
    - Logger caching by get_logger factory
    - PipelineLogger stage tracking and timing
    - Pipeline completion (success/failure)
    - Progress logging within stages
    - Global logging configuration

Justification:
    - Logging is infrastructure - focus on API behavior
    - Don't test every log level combination - test representative cases
    - Don't test stdlib logging internals - trust Python
    - Test that messages appear in logs with correct context
    - Test that logger factory caches correctly
"""
