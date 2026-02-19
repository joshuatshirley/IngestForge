"""
User-Friendly Error Formatter for IngestForge.

This module formats error codes into actionable user messages with troubleshooting
steps. JPL Rule #2, #4, #7, and #9 compliant.
"""

from typing import Optional
from ingestforge.core.errors import ErrorCode, ErrorContext, ERROR_SOLUTIONS

# JPL Rule #2: Bounded loops
MAX_CONTEXT_LINES = 5
MAX_SUGGESTIONS = 3


def format_user_error(
    code: ErrorCode,
    context: ErrorContext,
    show_technical: bool = False,
) -> str:
    """
    Format error for end-user display with troubleshooting steps.

    JPL Rule #4: <50 lines.
    JPL Rule #7: Returns explicit string.
    JPL Rule #9: Full type hints.

    Args:
        code: Error code from ErrorCode enum
        context: Error context with operation, file_path, details
        show_technical: Show technical details (--debug mode)

    Returns:
        Formatted error message with troubleshooting steps
    """
    solution = ERROR_SOLUTIONS.get(code, {})

    lines: list[str] = [
        f"✗ Error [{code.value}]: {solution.get('message', 'Unknown error')}",
        "",
    ]

    # Add context information (JPL bounded)
    if context.get("operation"):
        lines.append(f"  Operation: {context['operation']}")

    if context.get("file_path"):
        lines.append(f"  File: {context['file_path']}")

    if context.get("command"):
        lines.append(f"  Command: {context['command']}")

    if context.get("details") and show_technical:
        lines.append(f"  Details: {context['details']}")

    if any(context.get(k) for k in ["operation", "file_path", "command"]):
        lines.append("")

    # Add troubleshooting steps (JPL bounded to MAX_SUGGESTIONS)
    lines.append("  Troubleshooting:")
    fixes = solution.get("fixes", [])
    if isinstance(fixes, list):
        # JPL Rule #2: Bounded iteration
        for i, fix in enumerate(fixes[:MAX_SUGGESTIONS], 1):
            lines.append(f"    {i}. {fix}")
    else:
        lines.append("    No solutions available")

    # Add documentation link
    docs_link = solution.get("docs")
    if docs_link and isinstance(docs_link, str):
        lines.append(f"\n  See: {docs_link}")

    return "\n".join(lines)


def get_error_code_from_exception(
    error: Exception,
    operation: str,
) -> Optional[ErrorCode]:
    """
    Infer error code from exception type and context.

    JPL Rule #4: <40 lines.
    JPL Rule #7: Returns explicit Optional[ErrorCode].
    JPL Rule #9: Full type hints.

    Args:
        error: The exception that occurred
        operation: The operation being performed

    Returns:
        Best-match ErrorCode or None if no match
    """
    error_type = type(error).__name__
    error_msg = str(error).lower()

    # Installation errors
    if "python" in error_msg and "version" in error_msg:
        return ErrorCode.E001_PYTHON_VERSION
    if "node" in error_msg or "npm" in error_msg:
        return ErrorCode.E002_NODE_NOT_FOUND

    # File errors
    if error_type == "FileNotFoundError":
        return ErrorCode.E201_FILE_NOT_FOUND
    if "permission" in error_msg or error_type == "PermissionError":
        return ErrorCode.E403_PERMISSION_DENIED

    # Config errors
    if "config" in operation.lower():
        if "not found" in error_msg:
            return ErrorCode.E101_CONFIG_NOT_FOUND
        if "invalid" in error_msg or "yaml" in error_msg:
            return ErrorCode.E102_CONFIG_INVALID

    # Storage errors
    if "disk" in error_msg or "space" in error_msg:
        return ErrorCode.E402_DISK_FULL
    if "database" in error_msg or "connection" in error_msg:
        return ErrorCode.E401_DB_CONNECTION

    # Query errors
    if "query" in operation.lower():
        if "timeout" in error_msg:
            return ErrorCode.E302_QUERY_TIMEOUT
        if "empty" in error_msg or "no documents" in error_msg:
            return ErrorCode.E301_COLLECTION_EMPTY

    return None
