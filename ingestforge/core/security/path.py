"""
Path validation and sanitization for security.

Provides PathSanitizer class for preventing directory traversal attacks.
"""

import re
from pathlib import Path
from typing import Optional, Union

from ingestforge.core.exceptions import PathTraversalError
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class PathSanitizer:
    """
    Sanitize and validate file paths to prevent directory traversal attacks.

    This class provides methods to:
    - Sanitize filenames by removing dangerous characters
    - Validate that resolved paths stay within allowed directories
    - Detect and block '../' traversal attempts

    Example:
        >>> sanitizer = PathSanitizer(base_dir=Path("/app/data"))
        >>> safe_path = sanitizer.sanitize("../../../etc/passwd")
        >>> # Returns: Path("/app/data/etc/passwd") - traversal blocked

        >>> sanitizer.validate_within_base("../../../etc/passwd")
        >>> # Raises PathTraversalError
    """

    # Characters that are dangerous in filenames across platforms
    DANGEROUS_CHARS = re.compile(r'[<>:"|?*\x00-\x1f]')

    # Pattern to detect directory traversal attempts
    TRAVERSAL_PATTERN = re.compile(r"(^|[/\\])\.\.([/\\]|$)")

    def __init__(self, base_dir: Optional[Path] = None) -> None:
        """
        Initialize the path sanitizer.

        Args:
            base_dir: The base directory that all paths must resolve within.
                     If None, uses current working directory.
        """
        self._base_dir = (base_dir or Path.cwd()).resolve()

    @property
    def base_dir(self) -> Path:
        """Get the base directory."""
        return self._base_dir

    def sanitize_filename(self, filename: str) -> str:
        """
        Sanitize a filename by removing dangerous characters.

        Removes:
        - Path separators (/ and \\)
        - Parent directory references (..)
        - Null bytes and control characters
        - Windows-forbidden characters (<>:"|?*)

        Args:
            filename: The filename to sanitize.

        Returns:
            A sanitized filename safe for use on most filesystems.

        Example:
            >>> sanitizer.sanitize_filename("../../../passwd")
            'passwd'
            >>> sanitizer.sanitize_filename("file<name>.txt")
            'filename.txt'
        """
        if not filename:
            return "unnamed"

        # Remove path separators - filename should not contain paths
        filename = filename.replace("/", "_").replace("\\", "_")

        # Remove or replace dangerous sequences
        # Handle ".." by replacing with underscore
        while ".." in filename:
            filename = filename.replace("..", "_")

        # Remove dangerous characters
        filename = self.DANGEROUS_CHARS.sub("", filename)

        # Remove leading/trailing dots and spaces (Windows issues)
        filename = filename.strip(". ")

        # Ensure we have something left
        if not filename:
            return "unnamed"

        # Limit length to reasonable filesystem maximum
        max_length = 255
        if len(filename) > max_length:
            # Preserve extension if present
            path = Path(filename)
            ext = path.suffix
            if ext:
                name = path.stem[: max_length - len(ext)]
                filename = name + ext
            else:
                filename = filename[:max_length]

        return filename

    def sanitize_path(self, path: Union[str, Path]) -> Path:
        """
        Sanitize a path and resolve it within the base directory.

        This method:
        1. Detects and blocks '../' traversal attempts
        2. Resolves the path to an absolute path
        3. Ensures the result is within base_dir

        Args:
            path: The path to sanitize.

        Returns:
            A Path object guaranteed to be within base_dir.

        Raises:
            PathTraversalError: If the path attempts to escape base_dir.

        Example:
            >>> sanitizer = PathSanitizer(Path("/app/data"))
            >>> sanitizer.sanitize_path("docs/file.pdf")
            Path('/app/data/docs/file.pdf')
            >>> sanitizer.sanitize_path("../etc/passwd")
            # Raises PathTraversalError
        """
        assert path is not None, "Path cannot be None"
        assert self._base_dir is not None, "Base directory must be set"
        assert self._base_dir.is_absolute(), "Base directory must be absolute"

        path_str = str(path)

        # Check for obvious traversal attempts before resolution
        if self.TRAVERSAL_PATTERN.search(path_str):
            raise PathTraversalError(f"Path traversal detected in: {path_str!r}")

        # Convert to Path and resolve
        path_obj = Path(path_str)

        # If path is relative, join with base_dir
        if not path_obj.is_absolute():
            full_path = self._base_dir / path_obj
        else:
            full_path = path_obj

        # Resolve to get canonical path (resolves symlinks and ..)
        try:
            resolved = full_path.resolve()
        except (OSError, ValueError) as e:
            # SEC-002: Sanitize path disclosure
            logger.error(f"Invalid path: {path_str!r}")
            raise PathTraversalError("Invalid path: [REDACTED]") from e
        assert resolved.is_absolute(), "Resolved path must be absolute"

        # Verify the resolved path is within base_dir
        try:
            resolved.relative_to(self._base_dir)
        except ValueError:
            raise PathTraversalError(
                f"Path escapes base directory: {path_str!r} resolves to {resolved}"
            )

        return resolved


def sanitize_filename(filename: str) -> str:
    """
    Convenience function to sanitize a filename.

    Args:
        filename: The filename to sanitize.

    Returns:
        A sanitized filename.
    """
    return PathSanitizer().sanitize_filename(filename)
