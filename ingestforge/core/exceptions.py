"""
Centralized Exception Hierarchy for IngestForge.

This module defines all custom exceptions used throughout IngestForge.
All exceptions inherit from IngestForgeError for easy catching.

UX-004: Helpful Error Messages Implementation
---------------------------------------------
Each exception includes:
- user_message: Human-readable description of what went wrong
- why_it_happened: Explanation of the root cause
- how_to_fix: Actionable steps to resolve the issue
- error_code: Unique identifier for documentation lookup (e.g., "IF-FILE-001")

Usage
-----
    from ingestforge.core.exceptions import (
        IngestForgeError,
        ProcessingError,
        ChunkingError,
    )

    try:
        process_document(doc)
    except ProcessingError as e:
        logger.error(f"Processing failed: {e}")
    except IngestForgeError as e:
        logger.error(f"IngestForge error: {e}")

Exception Hierarchy
-------------------
    IngestForgeError (base)
    ├── SecurityError
    │   ├── PathTraversalError
    │   └── SSRFError
    ├── ProcessingError
    │   ├── ExtractionError
    │   ├── ChunkingError
    │   └── EnrichmentError
    │       └── EmbeddingError
    ├── LLMError
    │   ├── RateLimitError
    │   ├── ConfigurationError
    │   └── ContextLengthError
    ├── StorageError
    │   └── IndexError
    ├── RetryError
    ├── TimeoutError
    ├── FileNotFoundError
    ├── APITimeoutError
    ├── ConnectionError
    ├── DependencyError
    └── ValidationError
        └── ConfigValidationError

Design Principles
-----------------
1. All exceptions inherit from IngestForgeError
2. Catch specific exceptions when you can handle them
3. Let IngestForgeError propagate for general error handling
4. Include helpful error messages with context
5. UX-004: Every exception provides "why" and "how to fix" guidance
"""

from typing import Any, List, Optional
import builtins
import re


def sanitize_path(path: str) -> str:
    """Sanitize a file path to avoid leaking sensitive info.

    Replaces user directories and sensitive path components with placeholders.
    Follows Rule #7: Check Parameters and sanitize sensitive info.

    Args:
        path: Original file path

    Returns:
        Sanitized path with sensitive components replaced
    """
    if not path:
        return path

    # Patterns to sanitize (user home directories, API keys in paths)
    patterns = [
        # Windows user paths: C:\Users\username -> C:\Users\<user>
        (r"[A-Za-z]:\\Users\\[^\\]+", r"<user-home>"),
        # Unix/Mac home paths: /home/username or /Users/username -> <user-home>
        (r"/(?:home|Users)/[^/]+", r"<user-home>"),
        # API keys or tokens in paths (common patterns)
        (r"[a-zA-Z0-9]{32,}", r"<key>"),
        # Environment variable expansions that leaked
        (r"\$[A-Z_]+", r"<env>"),
    ]

    result = path
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)

    return result


def sanitize_message(message: str) -> str:
    """Sanitize an error message to avoid leaking sensitive info.

    Removes or masks API keys, tokens, passwords, and file paths.
    Follows Rule #7: Check Parameters and sanitize sensitive info.

    Args:
        message: Original error message

    Returns:
        Sanitized message with sensitive info replaced
    """
    if not message:
        return message

    result = message

    # Sanitize common sensitive patterns
    patterns = [
        # API keys (various formats)
        (r"(sk-|pk-|api_key[=:][\s]*)[a-zA-Z0-9_-]{20,}", r"\1<api-key>"),
        (r"(ANTHROPIC_API_KEY|OPENAI_API_KEY|API_KEY)[=:]\s*[^\s]+", r"\1=<hidden>"),
        # Bearer tokens
        (r"Bearer\s+[a-zA-Z0-9_.-]+", r"Bearer <token>"),
        # Basic auth
        (r"://[^:]+:[^@]+@", r"://<user>:<pass>@"),
        # Long hex strings (potential keys/hashes)
        (r"[a-fA-F0-9]{40,}", r"<hash>"),
        # File paths (apply path sanitization)
        (r"[A-Za-z]:\\[^\s\"']+", lambda m: sanitize_path(m.group(0))),
        (r"/(?:home|Users|var|etc)/[^\s\"']+", lambda m: sanitize_path(m.group(0))),
    ]

    for pattern, replacement in patterns:
        if callable(replacement):
            result = re.sub(pattern, replacement, result)
        else:
            result = re.sub(pattern, str(replacement), result)

    return result


def get_root_cause(exc: BaseException) -> BaseException:
    """Extract the root cause from a chain of exceptions.

    Follows nested __cause__ and __context__ attributes to find
    the original error that started the chain.

    Args:
        exc: Exception to analyze

    Returns:
        Root cause exception (may be the same as input)
    """
    seen = set()
    current = exc

    while current is not None:
        # Avoid infinite loops
        if id(current) in seen:
            break
        seen.add(id(current))

        # Prefer explicit cause over implicit context
        if current.__cause__ is not None:
            current = current.__cause__
        elif current.__context__ is not None:
            current = current.__context__
        else:
            break

    return current


class IngestForgeError(Exception):
    """
    Base exception for all IngestForge errors.

    All custom exceptions in IngestForge inherit from this class,
    making it easy to catch any IngestForge-specific error.

    UX-004: Includes helpful error information:
    - error_code: Unique code for documentation lookup (e.g., "IF-ERR-001")
    - why_it_happened: Explanation of the root cause
    - how_to_fix: List of actionable suggestions

    Example
    -------
        try:
            pipeline.process(document)
        except IngestForgeError as e:
            logger.error(f"Pipeline failed: {e}")
            print(f"Fix: {e.how_to_fix}")
    """

    # Default error info - subclasses should override
    error_code: str = "IF-ERR-000"
    why_it_happened: str = "An unexpected error occurred"
    how_to_fix: List[str] = ["Check the error message for details"]

    def __init__(
        self,
        message: str,
        *,
        error_code: Optional[str] = None,
        why_it_happened: Optional[str] = None,
        how_to_fix: Optional[List[str]] = None,
    ) -> None:
        """Initialize IngestForgeError with helpful information.

        Args:
            message: Human-readable error message
            error_code: Unique identifier (e.g., "IF-FILE-001")
            why_it_happened: Explanation of root cause
            how_to_fix: List of actionable fix suggestions
        """
        # Sanitize the message to avoid leaking sensitive info
        sanitized_message = sanitize_message(message)
        super().__init__(sanitized_message)

        # Override class defaults if provided
        if error_code is not None:
            self.error_code = error_code
        if why_it_happened is not None:
            self.why_it_happened = why_it_happened
        if how_to_fix is not None:
            self.how_to_fix = how_to_fix

    @property
    def user_message(self) -> str:
        """Get the user-friendly error message."""
        return str(self)

    def get_root_cause(self) -> BaseException:
        """Get the root cause of this exception chain."""
        return get_root_cause(self)


# ============================================================================
# Security Exceptions
# ============================================================================


class SecurityError(IngestForgeError):
    """
    Base exception for security-related errors.

    Raised when a security violation is detected, such as
    path traversal attempts or SSRF attacks.
    """

    error_code = "IF-SEC-000"
    why_it_happened = "A security check failed"
    how_to_fix = ["Review the input for potentially malicious content"]


class PathTraversalError(SecurityError):
    """
    Raised when a path traversal attack is detected.

    This occurs when a path contains sequences like '../' that
    would allow access to files outside the allowed directory.

    Example
    -------
        # Attempting to access /etc/passwd via traversal
        PathSanitizer().sanitize_path("../../../etc/passwd")
        # Raises: PathTraversalError("Path traversal detected: ../../../etc/passwd")
    """

    error_code = "IF-SEC-001"
    why_it_happened = (
        "The path contains sequences like '../' that could access files "
        "outside the allowed directory"
    )
    how_to_fix = [
        "Use absolute paths instead of relative paths",
        "Remove '..' sequences from the path",
        "Ensure the file is within the project directory",
    ]


class SSRFError(SecurityError):
    """
    Raised when a Server-Side Request Forgery (SSRF) attack is detected.

    This occurs when a URL points to internal/private network addresses
    that should not be accessible from the server.

    Example
    -------
        # Attempting to access AWS metadata endpoint
        URLValidator().validate("http://169.254.169.254/latest/meta-data")
        # Raises: SSRFError("URL resolves to private IP range")
    """

    error_code = "IF-SEC-002"
    why_it_happened = (
        "The URL points to an internal or private network address that "
        "should not be accessible"
    )
    how_to_fix = [
        "Use a public URL instead of an internal address",
        "Check that the URL doesn't resolve to localhost or private IP ranges",
        "Verify the URL is from a trusted source",
    ]


# ============================================================================
# Processing Exceptions
# ============================================================================


class ProcessingError(IngestForgeError):
    """
    Base exception for document processing errors.

    Raised when document processing fails at any stage:
    extraction, parsing, validation, etc.
    """

    error_code = "IF-PROC-000"
    why_it_happened = "Document processing failed at some stage"
    how_to_fix = [
        "Check that the file is not corrupted",
        "Verify the file format is supported",
        "Try processing the file again",
    ]


class ExtractionError(ProcessingError):
    """
    Raised when content extraction from a document fails.

    This can occur when:
    - PDF parsing fails
    - OCR cannot recognize text
    - File format is corrupted
    """

    error_code = "IF-PROC-001"
    why_it_happened = (
        "Could not extract text content from the document. The file may be "
        "corrupted, password-protected, or in an unsupported format"
    )
    how_to_fix = [
        "Check if the file opens correctly in its native application",
        "Ensure the file is not password-protected",
        "For scanned PDFs, install Tesseract OCR for text extraction",
        "Try converting the file to a different format (e.g., PDF to TXT)",
    ]


class ChunkingError(ProcessingError):
    """
    Raised when text chunking fails.

    This can occur when:
    - Text is too short to chunk
    - Chunking strategy encounters invalid input
    - Chunk size constraints cannot be satisfied
    """

    error_code = "IF-PROC-002"
    why_it_happened = (
        "Could not split the document into chunks. The text may be too short "
        "or the chunking parameters may be incompatible"
    )
    how_to_fix = [
        "Ensure the document has sufficient content (at least a few sentences)",
        "Adjust chunk_size in configuration if chunks are too large/small",
        "Check that chunk_overlap is smaller than chunk_size",
    ]


class EnrichmentError(ProcessingError):
    """
    Raised when chunk enrichment fails.

    This can occur when:
    - Entity extraction fails
    - Question generation fails
    - Summary generation fails
    """

    error_code = "IF-PROC-003"
    why_it_happened = (
        "Failed to enrich chunks with additional metadata like entities, "
        "summaries, or questions"
    )
    how_to_fix = [
        "Check if NLP models are installed correctly",
        "Verify there is sufficient memory for enrichment",
        "Try disabling enrichment with --no-enrich flag",
    ]


class EmbeddingError(EnrichmentError):
    """
    Raised when embedding generation fails.

    This can occur when:
    - Embedding model is not available
    - Text exceeds model's token limit
    - GPU memory is exhausted
    """

    error_code = "IF-PROC-004"
    why_it_happened = (
        "Failed to generate vector embeddings for text chunks. "
        "The embedding model may not be loaded or memory may be exhausted"
    )
    how_to_fix = [
        "Check if sentence-transformers is installed: pip install sentence-transformers",
        "Reduce batch size with INGESTFORGE_EMBEDDING_BATCH_SIZE environment variable",
        "Use a smaller embedding model in configuration",
        "Free up memory by closing other applications",
    ]


# ============================================================================
# LLM Exceptions
# ============================================================================


class LLMError(IngestForgeError):
    """
    Base exception for LLM-related errors.

    Raised when LLM operations fail, such as API calls
    to OpenAI, Anthropic, Google, or local models.
    """

    error_code = "IF-LLM-000"
    why_it_happened = "An LLM operation failed"
    how_to_fix = [
        "Check your API key is set correctly",
        "Verify your internet connection",
        "Try a different LLM provider",
    ]


class RateLimitError(LLMError):
    """
    Raised when an LLM API rate limit is exceeded.

    The retry decorator will typically handle this by
    waiting and retrying, but if retries are exhausted,
    this exception propagates.

    Attributes
    ----------
    retry_after : float, optional
        Seconds to wait before retrying (if provided by API)
    """

    error_code = "IF-LLM-001"
    why_it_happened = (
        "The LLM API rate limit was exceeded. You've made too many "
        "requests in a short period"
    )
    how_to_fix = [
        "Wait a few minutes before retrying",
        "Reduce the number of concurrent requests",
        "Consider upgrading your API tier for higher limits",
        "Use a local LLM like llama.cpp for unlimited requests",
    ]

    def __init__(
        self,
        message: str,
        retry_after: Optional[float] = None,
        *,
        error_code: Optional[str] = None,
        why_it_happened: Optional[str] = None,
        how_to_fix: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            why_it_happened=why_it_happened,
            how_to_fix=how_to_fix,
        )
        self.retry_after = retry_after


class ConfigurationError(LLMError):
    """
    Raised when LLM configuration is invalid.

    This can occur when:
    - API key is missing or invalid
    - Model name is not recognized
    - Required parameters are missing
    """

    error_code = "IF-LLM-002"
    why_it_happened = (
        "The LLM configuration is invalid. The API key may be missing, "
        "expired, or the model name may be incorrect"
    )
    how_to_fix = [
        "Set your API key: export OPENAI_API_KEY=your-key",
        "Or for Anthropic: export ANTHROPIC_API_KEY=your-key",
        "Verify the model name is correct (e.g., gpt-4, claude-3-sonnet)",
        "Check your API key is still valid at the provider's dashboard",
        "Run 'ingestforge doctor' to diagnose configuration issues",
    ]


class ContextLengthError(LLMError):
    """
    Raised when input exceeds the model's context length.

    Attributes
    ----------
    max_tokens : int
        Maximum tokens the model supports
    actual_tokens : int
        Actual token count of the input
    """

    error_code = "IF-LLM-003"
    why_it_happened = (
        "The input text exceeds the LLM's maximum context window. "
        "The text is too long for the model to process"
    )
    how_to_fix = [
        "Use fewer context chunks with --k flag (e.g., --k 3)",
        "Switch to a model with larger context (e.g., claude-3-sonnet)",
        "Reduce the chunk_size in configuration for smaller chunks",
    ]

    def __init__(
        self,
        message: str,
        max_tokens: Optional[int] = None,
        actual_tokens: Optional[int] = None,
        *,
        error_code: Optional[str] = None,
        why_it_happened: Optional[str] = None,
        how_to_fix: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            why_it_happened=why_it_happened,
            how_to_fix=how_to_fix,
        )
        self.max_tokens = max_tokens
        self.actual_tokens = actual_tokens


# ============================================================================
# Storage Exceptions
# ============================================================================


class StorageError(IngestForgeError):
    """
    Raised when storage operations fail.

    This can occur when:
    - Database connection fails
    - Write operation fails
    - Index corruption is detected
    """

    error_code = "IF-STOR-000"
    why_it_happened = (
        "A storage operation failed. The database may be corrupted, "
        "locked by another process, or the disk may be full"
    )
    how_to_fix = [
        "Check disk space with 'df -h' (Linux) or check drive properties (Windows)",
        "Ensure no other IngestForge processes are running",
        "Try rebuilding the index with 'ingestforge index rebuild'",
        "Delete the .ingest directory and re-run ingest if corruption persists",
    ]


class IndexError(StorageError):
    """
    Raised when index operations fail.

    This can occur when:
    - BM25 index is corrupted
    - Vector index cannot be loaded
    - Index update fails
    """

    error_code = "IF-STOR-001"
    why_it_happened = (
        "The search index is corrupted or cannot be loaded. "
        "This may happen after interrupted writes or disk errors"
    )
    how_to_fix = [
        "Rebuild the index: ingestforge index rebuild",
        "Delete and recreate: rm -rf .ingest/indexes && ingestforge ingest .",
        "Check for disk errors on the storage volume",
    ]


# ============================================================================
# Infrastructure Exceptions
# ============================================================================


class RetryError(IngestForgeError):
    """
    Raised when all retry attempts are exhausted.

    This exception wraps the original exception that caused
    the retries to fail.

    Attributes
    ----------
    attempts : int
        Number of retry attempts made
    last_exception : Exception
        The exception from the final attempt
    """

    error_code = "IF-INFRA-001"
    why_it_happened = (
        "The operation failed repeatedly and all retry attempts were exhausted. "
        "This typically indicates a persistent problem"
    )
    how_to_fix = [
        "Check your network connection",
        "Verify the external service is available",
        "Wait a few minutes and try again",
        "Check service status pages for outages",
    ]

    def __init__(
        self,
        message: str,
        attempts: Optional[int] = None,
        last_exception: Optional[Exception] = None,
        *,
        error_code: Optional[str] = None,
        why_it_happened: Optional[str] = None,
        how_to_fix: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            why_it_happened=why_it_happened,
            how_to_fix=how_to_fix,
        )
        self.attempts = attempts
        self.last_exception = last_exception


class TimeoutError(IngestForgeError):
    """
    Raised when an operation times out.

    Attributes
    ----------
    timeout : float
        The timeout duration in seconds
    operation : str
        Description of the operation that timed out
    """

    error_code = "IF-INFRA-002"
    why_it_happened = (
        "The operation took too long and was terminated. "
        "The server may be slow or unresponsive"
    )
    how_to_fix = [
        "Check your network connection speed",
        "Try again during off-peak hours",
        "Increase the timeout setting if available",
        "Process smaller files or batches",
    ]

    def __init__(
        self,
        message: str,
        timeout: Optional[float] = None,
        operation: Optional[str] = None,
        *,
        error_code: Optional[str] = None,
        why_it_happened: Optional[str] = None,
        how_to_fix: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            why_it_happened=why_it_happened,
            how_to_fix=how_to_fix,
        )
        self.timeout = timeout
        self.operation = operation


# ============================================================================
# File & Connection Exceptions (UX-004)
# ============================================================================


class FileNotFoundError(IngestForgeError):
    """
    Raised when a required file cannot be found.

    Note: This shadows the built-in FileNotFoundError intentionally
    to provide helpful error messages. Use builtins.FileNotFoundError
    if you need the standard exception.
    """

    error_code = "IF-FILE-001"
    why_it_happened = (
        "The specified file or directory could not be found. "
        "The path may be incorrect or the file may have been moved or deleted"
    )
    how_to_fix = [
        "Check that the file path is correct",
        "Verify the file exists: ls <path> (Linux) or dir <path> (Windows)",
        "Ensure you have read permissions for the file",
        "Check for typos in the filename",
    ]


class APITimeoutError(TimeoutError):
    """
    Raised when an API call times out.

    Specialized timeout for external API requests.
    """

    error_code = "IF-API-001"
    why_it_happened = (
        "The API request timed out waiting for a response. "
        "The service may be overloaded or your connection may be slow"
    )
    how_to_fix = [
        "Check your internet connection",
        "Try again in a few minutes",
        "Switch to a local LLM (llama.cpp) for offline operation",
        "Check the API provider's status page for outages",
    ]


class ConnectionError(IngestForgeError):
    """
    Raised when a network connection fails.

    Note: This shadows the built-in ConnectionError intentionally
    to provide helpful error messages.
    """

    error_code = "IF-CONN-001"
    why_it_happened = (
        "Could not establish a network connection. "
        "You may be offline or the server may be unreachable"
    )
    how_to_fix = [
        "Check your internet connection",
        "Verify the URL or server address is correct",
        "Check if a firewall or proxy is blocking the connection",
        "Try accessing the service in a web browser",
        "Use 'ingestforge doctor' to diagnose network issues",
    ]


class DependencyError(IngestForgeError):
    """
    Raised when a required dependency is missing.

    Used for optional dependencies that are needed for specific features.
    """

    error_code = "IF-DEP-001"
    why_it_happened = (
        "A required Python package is not installed. "
        "Some features require optional dependencies"
    )
    how_to_fix = [
        "Install the missing package with pip install <package>",
        "For PDF support: pip install pdfplumber",
        "For OCR: pip install pytesseract (plus Tesseract binary)",
        "For LLMs: pip install anthropic openai",
        "Run 'ingestforge doctor' to see all missing dependencies",
    ]


# ============================================================================
# Validation Exceptions
# ============================================================================


class ValidationError(IngestForgeError):
    """
    Raised when validation fails.

    This can occur when:
    - Configuration is invalid
    - Input data doesn't meet requirements
    - Schema validation fails
    """

    error_code = "IF-VAL-000"
    why_it_happened = "Validation failed for input data or configuration"
    how_to_fix = [
        "Check the error message for specific validation failures",
        "Review the expected format or value constraints",
    ]


class ConfigValidationError(ValidationError):
    """
    Raised when configuration validation fails.

    Attributes
    ----------
    field : str
        The configuration field that failed validation
    value : any
        The invalid value
    """

    error_code = "IF-VAL-001"
    why_it_happened = (
        "A configuration value is invalid. "
        "The ingestforge.yaml file may have incorrect settings"
    )
    how_to_fix = [
        "Check ingestforge.yaml for syntax errors",
        "Verify the value type matches what's expected",
        "See documentation for valid configuration options",
        "Run 'ingestforge config show' to view current settings",
        "Reset to defaults: ingestforge config reset",
    ]

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Any = None,
        *,
        error_code: Optional[str] = None,
        why_it_happened: Optional[str] = None,
        how_to_fix: Optional[List[str]] = None,
    ) -> None:
        super().__init__(
            message,
            error_code=error_code,
            why_it_happened=why_it_happened,
            how_to_fix=how_to_fix,
        )
        self.field = field
        self.value = value


# ============================================================================
# Error Info Lookup (UX-004)
# ============================================================================


# Mapping from standard exceptions to helpful error info
# Used by ErrorRenderer to provide context for non-IngestForge exceptions
STANDARD_ERROR_INFO: dict[type, dict[str, Any]] = {
    builtins.FileNotFoundError: {
        "error_code": "IF-FILE-001",
        "why_it_happened": "The specified file or directory could not be found",
        "how_to_fix": [
            "Check that the file path is correct",
            "Verify the file exists using your file explorer",
            "Ensure you have read permissions for the file",
        ],
    },
    builtins.PermissionError: {
        "error_code": "IF-FILE-002",
        "why_it_happened": "You don't have permission to access this file or directory",
        "how_to_fix": [
            "Run the command with elevated privileges (sudo on Linux/Mac)",
            "Check file permissions: ls -la <file>",
            "Ensure you own the file or have read/write access",
        ],
    },
    builtins.ConnectionError: {
        "error_code": "IF-CONN-001",
        "why_it_happened": "Could not establish a network connection",
        "how_to_fix": [
            "Check your internet connection",
            "Verify the URL or server address",
            "Check if a firewall is blocking the connection",
        ],
    },
    builtins.TimeoutError: {
        "error_code": "IF-INFRA-002",
        "why_it_happened": "The operation took too long and was terminated",
        "how_to_fix": [
            "Check your network connection",
            "Try again in a few minutes",
            "Process smaller files or batches",
        ],
    },
    ModuleNotFoundError: {
        "error_code": "IF-DEP-001",
        "why_it_happened": "A required Python package is not installed",
        "how_to_fix": [
            "Install the missing package: pip install <package-name>",
            "Run 'ingestforge doctor' to see all missing dependencies",
        ],
    },
    ImportError: {
        "error_code": "IF-DEP-002",
        "why_it_happened": "A required module could not be imported correctly",
        "how_to_fix": [
            "Reinstall the package: pip install --force-reinstall <package>",
            "Check for version conflicts: pip check",
            "Create a fresh virtual environment",
        ],
    },
    KeyError: {
        "error_code": "IF-CFG-001",
        "why_it_happened": "A required configuration key is missing",
        "how_to_fix": [
            "Check your ingestforge.yaml configuration file",
            "Run 'ingestforge init' to create a new configuration",
            "See documentation for required settings",
        ],
    },
    ValueError: {
        "error_code": "IF-VAL-002",
        "why_it_happened": "An invalid value was provided",
        "how_to_fix": [
            "Check the error message for the expected value format",
            "Verify your input matches the required type",
        ],
    },
    TypeError: {
        "error_code": "IF-VAL-003",
        "why_it_happened": "A value of the wrong type was provided",
        "how_to_fix": [
            "Check that you're passing the correct argument types",
            "Review the function documentation for expected types",
        ],
    },
    MemoryError: {
        "error_code": "IF-MEM-001",
        "why_it_happened": "The system ran out of memory",
        "how_to_fix": [
            "Close other applications to free up RAM",
            "Process smaller files or fewer at a time",
            "Reduce batch size with INGESTFORGE_EMBEDDING_BATCH_SIZE=8",
            "Use a smaller embedding model",
        ],
    },
    OSError: {
        "error_code": "IF-SYS-001",
        "why_it_happened": "A system-level error occurred",
        "how_to_fix": [
            "Check disk space and permissions",
            "Ensure the system has required resources",
            "Review system logs for more details",
        ],
    },
}


def get_error_info(exc: BaseException) -> dict[str, Any]:
    """Get helpful error information for any exception.

    Looks up the exception type in STANDARD_ERROR_INFO or extracts
    info from IngestForgeError subclasses.

    Args:
        exc: Exception to get info for

    Returns:
        Dict with error_code, why_it_happened, how_to_fix
    """
    # For our custom exceptions, extract the attributes
    if isinstance(exc, IngestForgeError):
        return {
            "error_code": exc.error_code,
            "why_it_happened": exc.why_it_happened,
            "how_to_fix": exc.how_to_fix,
        }

    # Look up standard exceptions
    exc_type = type(exc)
    if exc_type in STANDARD_ERROR_INFO:
        return STANDARD_ERROR_INFO[exc_type]

    # Check parent classes
    for parent_type, info in STANDARD_ERROR_INFO.items():
        if isinstance(exc, parent_type):
            return info

    # Default fallback
    return {
        "error_code": "IF-ERR-999",
        "why_it_happened": "An unexpected error occurred",
        "how_to_fix": [
            "Check the error message for details",
            "Run 'ingestforge doctor' to diagnose issues",
            "Report the issue if it persists",
        ],
    }
