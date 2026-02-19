"""
Security Utilities for IngestForge.

This module provides defensive security measures against common web application
vulnerabilities. All file and URL operations in IngestForge should use these
utilities to ensure safe handling of user-provided paths and URLs.

Architecture Context
--------------------
Security utilities sit in the Core layer and are used defensively throughout
the application, particularly in:

    - Pipeline: SafeFileOperations for moving processed documents
    - Ingest: Path validation for incoming documents
    - HTML Processor: URL validation for web content fetching
    - API: Request validation for REST endpoints

Threat Model
------------
IngestForge processes user-provided documents and URLs, creating attack vectors:

    1. **Directory Traversal** (Path Traversal)
       Attack: User uploads file named "../../../etc/passwd"
       Defense: PathSanitizer.sanitize_path() blocks ../ sequences

    2. **Server-Side Request Forgery (SSRF)**
       Attack: User provides URL like "http://169.254.169.254/metadata"
       Defense: URLValidator blocks private IPs, localhost, link-local

Components
----------
**PathSanitizer**
    Validates and sanitizes file paths:
    - sanitize_filename(): Removes dangerous characters
    - sanitize_path(): Resolves and validates within base directory

**URLValidator**
    Validates URLs for SSRF attacks:
    - Blocks private IP ranges (10.x, 172.16-31.x, 192.168.x)
    - Blocks localhost and link-local addresses

**SafeFileOperations**
    File operations with built-in path validation:
    - safe_move(): File transfer with validation
    - validate_path(): Validate path within base directory

Usage Pattern
-------------
Security utilities are designed for transparent use:

    # Instead of raw file operations:
    shutil.move(user_path, dest)  # DANGEROUS

    # Use SafeFileOperations:
    safe_ops = SafeFileOperations(base_dir)
    safe_ops.safe_move(user_path, dest)  # Validated

    # Instead of raw URL fetching:
    requests.get(user_url)  # DANGEROUS

    # Use URLValidator:
    validator = URLValidator()
    validator.validate(user_url)  # Raises SSRFError if unsafe
    requests.get(user_url)

Convenience Functions
---------------------
For one-off validations, use the module-level functions:

    from ingestforge.core.security import sanitize_filename

    safe_name = sanitize_filename("../../../etc/passwd")
    # Returns: "passwd"
"""

# Exceptions (for backward compatibility with imports from security module)
from ingestforge.core.exceptions import PathTraversalError, SSRFError

# Path sanitization
from ingestforge.core.security.path import PathSanitizer, sanitize_filename

# URL validation (SSRF protection)
# URLValidator class not yet implemented, use validate_url function instead
# from ingestforge.core.security.url import URLValidator

# Safe file operations
from ingestforge.core.security.files import SafeFileOperations

# Safe command execution (shell injection protection)
from ingestforge.core.security.command import (
    CommandValidator,
    CommandInjectionError,
    SafeCommand,
    safe_run,
    ALLOWED_EXECUTABLES,
)

# Safe environment variable parsing
from ingestforge.core.security.env import (
    get_env_int,
    get_env_float,
    get_env_path,
    get_env_whitelist,
    get_env_bool,
    STORAGE_BACKENDS,
    OCR_ENGINES,
    LLM_PROVIDERS,
    CHUNKING_STRATEGIES,
    PERFORMANCE_MODES,
)

# Network lock / Air-gap mode (SEC-002)
from ingestforge.core.security.network_lock import (
    NetworkSecurityError,
    NetworkConfig,
    enable_offline_mode,
    disable_offline_mode,
    is_offline_mode,
    add_to_whitelist,
    offline_context,
    get_network_status,
    check_network_allowed,
)

# Access Control List (ACL) for multi-user collaboration (TICKET-303)
from ingestforge.core.security.acl import (
    ACLGuard,
    ACLEntry,
    Permission,
    PermissionDenied,
    Role,
    ROLE_PERMISSIONS,
)

__all__ = [
    # Exceptions
    "PathTraversalError",
    "SSRFError",
    # Path sanitization
    "PathSanitizer",
    "sanitize_filename",
    # URL validation
    # "URLValidator",  # Not yet implemented
    # File operations
    "SafeFileOperations",
    # Command execution
    "CommandValidator",
    "CommandInjectionError",
    "SafeCommand",
    "safe_run",
    "ALLOWED_EXECUTABLES",
    # Environment variables
    "get_env_int",
    "get_env_float",
    "get_env_path",
    "get_env_whitelist",
    "get_env_bool",
    "STORAGE_BACKENDS",
    "OCR_ENGINES",
    "LLM_PROVIDERS",
    "CHUNKING_STRATEGIES",
    "PERFORMANCE_MODES",
    # Network lock (SEC-002)
    "NetworkSecurityError",
    "NetworkConfig",
    "enable_offline_mode",
    "disable_offline_mode",
    "is_offline_mode",
    "add_to_whitelist",
    "offline_context",
    "get_network_status",
    "check_network_allowed",
    # ACL (TICKET-303)
    "ACLGuard",
    "ACLEntry",
    "Permission",
    "PermissionDenied",
    "Role",
    "ROLE_PERMISSIONS",
]
