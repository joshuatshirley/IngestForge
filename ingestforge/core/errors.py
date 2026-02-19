"""
Safe Error Message Wrapper for IngestForge.

This module provides error message sanitization to prevent information disclosure
vulnerabilities (SEC-002). JPL Rule #4 and #9 compliant.
"""

import logging
import re
from enum import Enum
from typing import Optional, TypedDict


class ErrorCode(str, Enum):
    """Error codes for user-friendly messages (E001-E999)."""

    # Installation (E0xx)
    E001_PYTHON_VERSION = "E001"
    E002_NODE_NOT_FOUND = "E002"
    E003_DEPENDENCY_FAILED = "E003"
    E004_INSTALL_TIMEOUT = "E004"

    # Configuration (E1xx)
    E101_CONFIG_NOT_FOUND = "E101"
    E102_CONFIG_INVALID = "E102"
    E103_MODEL_NOT_FOUND = "E103"
    E104_API_KEY_MISSING = "E104"
    E105_STORAGE_PATH_INVALID = "E105"

    # Ingestion (E2xx)
    E201_FILE_NOT_FOUND = "E201"
    E202_UNSUPPORTED_FORMAT = "E202"
    E203_OCR_FAILED = "E203"
    E204_PARSING_ERROR = "E204"
    E205_FILE_TOO_LARGE = "E205"

    # Query (E3xx)
    E301_COLLECTION_EMPTY = "E301"
    E302_QUERY_TIMEOUT = "E302"
    E303_INVALID_QUERY = "E303"
    E304_EMBEDDINGS_FAILED = "E304"

    # Storage (E4xx)
    E401_DB_CONNECTION = "E401"
    E402_DISK_FULL = "E402"
    E403_PERMISSION_DENIED = "E403"
    E404_MIGRATION_FAILED = "E404"


class ErrorContext(TypedDict, total=False):
    """Context information for error formatting."""

    operation: str
    file_path: Optional[str]
    details: Optional[str]
    command: Optional[str]


# JPL Rule #2: Bounded - max 3 suggestions per error
MAX_SUGGESTIONS = 3

ERROR_SOLUTIONS: dict[ErrorCode, dict[str, object]] = {
    ErrorCode.E001_PYTHON_VERSION: {
        "message": "Python 3.10+ required",
        "fixes": [
            "Install Python 3.10+: https://python.org",
            "Check version: python --version",
            "Update PATH to use correct Python",
        ],
        "docs": "https://docs.ingestforge.io/install#python",
    },
    ErrorCode.E002_NODE_NOT_FOUND: {
        "message": "Node.js not found",
        "fixes": [
            "Install Node.js 18+: https://nodejs.org",
            "Check installation: node --version",
            "Restart terminal after install",
        ],
        "docs": "https://docs.ingestforge.io/install#nodejs",
    },
    ErrorCode.E003_DEPENDENCY_FAILED: {
        "message": "Dependency installation failed",
        "fixes": [
            "Check internet connection",
            "Try: pip install --upgrade pip",
            "Check disk space: df -h (Linux/Mac) or dir (Windows)",
        ],
        "docs": "https://docs.ingestforge.io/troubleshoot#E003",
    },
    ErrorCode.E004_INSTALL_TIMEOUT: {
        "message": "Installation timeout",
        "fixes": [
            "Check network connectivity",
            "Increase timeout: pip install --timeout=300",
            "Use a different PyPI mirror",
        ],
        "docs": "https://docs.ingestforge.io/troubleshoot#E004",
    },
    ErrorCode.E101_CONFIG_NOT_FOUND: {
        "message": "Configuration file not found",
        "fixes": [
            "Run setup wizard: ingestforge setup",
            "Check config path: ~/.ingestforge/config.yml",
            "Create minimal config: ingestforge config init",
        ],
        "docs": "https://docs.ingestforge.io/config#setup",
    },
    ErrorCode.E102_CONFIG_INVALID: {
        "message": "Configuration file invalid",
        "fixes": [
            "Validate YAML syntax: yamllint config.yml",
            "Check required fields in docs",
            "Regenerate: ingestforge setup --force",
        ],
        "docs": "https://docs.ingestforge.io/config#validation",
    },
    ErrorCode.E103_MODEL_NOT_FOUND: {
        "message": "Embedding model not found",
        "fixes": [
            "Download model: ingestforge setup --download-models",
            "Check model cache: ~/.cache/ingestforge/",
            "Verify internet connection",
        ],
        "docs": "https://docs.ingestforge.io/models#download",
    },
    ErrorCode.E104_API_KEY_MISSING: {
        "message": "API key not configured",
        "fixes": [
            "Set key: export ANTHROPIC_API_KEY=sk-...",
            "Or add to config.yml: api_keys section",
            "Get key: https://console.anthropic.com",
        ],
        "docs": "https://docs.ingestforge.io/config#api-keys",
    },
    ErrorCode.E105_STORAGE_PATH_INVALID: {
        "message": "Storage path is invalid or inaccessible",
        "fixes": [
            "Check path exists and is writable",
            "Use absolute paths in config.yml",
            "Verify permissions: ls -la (Linux/Mac)",
        ],
        "docs": "https://docs.ingestforge.io/config#storage",
    },
    ErrorCode.E201_FILE_NOT_FOUND: {
        "message": "File or directory not found",
        "fixes": [
            "Check the path exists: ls <path>",
            "Use absolute paths (e.g., /home/user/docs)",
            "Verify file permissions: ls -la <path>",
        ],
        "docs": "https://docs.ingestforge.io/troubleshoot#E201",
    },
    ErrorCode.E202_UNSUPPORTED_FORMAT: {
        "message": "File format not supported",
        "fixes": [
            "Check supported formats in docs",
            "Convert to PDF or TXT",
            "Request format support: GitHub issues",
        ],
        "docs": "https://docs.ingestforge.io/ingest#formats",
    },
    ErrorCode.E203_OCR_FAILED: {
        "message": "OCR extraction failed",
        "fixes": [
            "Check Tesseract is installed: tesseract --version",
            "Install: apt-get install tesseract-ocr (Linux)",
            "Try alternative: ingestforge ingest --use-vlm",
        ],
        "docs": "https://docs.ingestforge.io/ingest#ocr",
    },
    ErrorCode.E204_PARSING_ERROR: {
        "message": "Document parsing failed",
        "fixes": [
            "Check file is not corrupted",
            "Try alternative processor: --processor=unstructured",
            "Report issue with sample file",
        ],
        "docs": "https://docs.ingestforge.io/troubleshoot#E204",
    },
    ErrorCode.E205_FILE_TOO_LARGE: {
        "message": "File exceeds size limit",
        "fixes": [
            "Split large PDFs: ingestforge transform split",
            "Increase limit in config.yml: max_file_size_mb",
            "Process in chunks: --chunk-size=1000",
        ],
        "docs": "https://docs.ingestforge.io/ingest#limits",
    },
    ErrorCode.E301_COLLECTION_EMPTY: {
        "message": "No documents in collection",
        "fixes": [
            "Ingest documents first: ingestforge ingest <path>",
            "Check collection: ingestforge storage stats",
            "Verify storage backend is running",
        ],
        "docs": "https://docs.ingestforge.io/query#setup",
    },
    ErrorCode.E302_QUERY_TIMEOUT: {
        "message": "Query execution timeout",
        "fixes": [
            "Increase timeout: --timeout=60",
            "Reduce search depth: --max-results=10",
            "Check database performance",
        ],
        "docs": "https://docs.ingestforge.io/troubleshoot#E302",
    },
    ErrorCode.E303_INVALID_QUERY: {
        "message": "Query syntax invalid",
        "fixes": [
            "Use natural language queries",
            "Check filter syntax in docs",
            "Try: ingestforge query --help",
        ],
        "docs": "https://docs.ingestforge.io/query#syntax",
    },
    ErrorCode.E304_EMBEDDINGS_FAILED: {
        "message": "Embedding generation failed",
        "fixes": [
            "Check model availability",
            "Verify API key is valid",
            "Try local model: --embedding-model=local",
        ],
        "docs": "https://docs.ingestforge.io/troubleshoot#E304",
    },
    ErrorCode.E401_DB_CONNECTION: {
        "message": "Database connection failed",
        "fixes": [
            "Check ChromaDB is running",
            "Verify connection in config.yml",
            "Start storage: ingestforge storage start",
        ],
        "docs": "https://docs.ingestforge.io/storage#troubleshoot",
    },
    ErrorCode.E402_DISK_FULL: {
        "message": "Insufficient disk space",
        "fixes": [
            "Check disk space: df -h (Linux/Mac)",
            "Clean old data: ingestforge cleanup",
            "Move storage to larger drive",
        ],
        "docs": "https://docs.ingestforge.io/troubleshoot#E402",
    },
    ErrorCode.E403_PERMISSION_DENIED: {
        "message": "Permission denied",
        "fixes": [
            "Check file/directory permissions",
            "Run with appropriate privileges",
            "Fix ownership: chown -R user:group path",
        ],
        "docs": "https://docs.ingestforge.io/troubleshoot#E403",
    },
    ErrorCode.E404_MIGRATION_FAILED: {
        "message": "Database migration failed",
        "fixes": [
            "Backup data first: ingestforge backup",
            "Check logs: ~/.ingestforge/logs/",
            "Rollback: ingestforge storage rollback",
        ],
        "docs": "https://docs.ingestforge.io/storage#migration",
    },
}


class SafeErrorMessage:
    """Sanitizes error messages to prevent information disclosure."""

    _SENSITIVE_PATTERNS = [
        # Windows paths - match drive letter with path components
        (r"[A-Z]:[/\][\w\/.-]+", "[PATH]"),
        # Unix absolute paths
        (r"/[\w/.-]+", "[PATH]"),
        # Artifact IDs
        (r'artifact_id[=:]?\s*["\']?([a-zA-Z0-9_-]+)["\']?', "artifact_id=[ID]"),
        # User IDs
        (r'user_id[=:]?\s*["\']?([a-zA-Z0-9_-]+)["\']?', "user_id=[ID]"),
        # Document IDs
        (r'document_id[=:]?\s*["\']?([a-zA-Z0-9_-]+)["\']?', "document_id=[ID]"),
        # Email addresses
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL]"),
        # IP addresses
        (r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "[IP]"),
        # API keys
        (r'api[_-]?key[=:]?\s*["\']?([a-zA-Z0-9_-]+)["\']?', "api_key=[REDACTED]"),
        # Tokens
        (r'token[=:]?\s*["\']?([a-zA-Z0-9_-]+)["\']?', "token=[REDACTED]"),
    ]

    @staticmethod
    def sanitize(
        error: Exception,
        context: str,
        logger: Optional[logging.Logger] = None,
    ) -> str:
        """Sanitize an exception for safe display to end users."""
        log = logger or logging.getLogger(__name__)
        log.error(
            f"[{context}] Error occurred: {type(error).__name__}: {error}",
            exc_info=True,
        )
        return (
            f"An error occurred during {context}. "
            "Please check the logs or contact support."
        )

    @staticmethod
    def sanitize_partial(
        error: Exception,
        context: str,
        logger: Optional[logging.Logger] = None,
    ) -> str:
        """Sanitize with partial details preserved."""
        log = logger or logging.getLogger(__name__)
        log.error(
            f"[{context}] Error occurred: {type(error).__name__}: {error}",
            exc_info=True,
        )
        error_msg = str(error)
        for pattern, replacement in SafeErrorMessage._SENSITIVE_PATTERNS:
            error_msg = re.sub(pattern, replacement, error_msg)
        return f"{context} failed: {type(error).__name__}: {error_msg}"

    @staticmethod
    def sanitize_message(message: str) -> str:
        """Sanitize a string message by removing sensitive patterns."""
        sanitized = message
        for pattern, replacement in SafeErrorMessage._SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, replacement, sanitized)
        return sanitized


def sanitize_error(
    error: Exception,
    context: str,
    logger: Optional[logging.Logger] = None,
) -> str:
    """Convenience function wrapping SafeErrorMessage.sanitize()."""
    return SafeErrorMessage.sanitize(error, context, logger)
