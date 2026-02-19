"""
Safe file operations with built-in path validation.

Provides SafeFileOperations for file operations that enforce path sanitization.
"""

import shutil
from pathlib import Path
from typing import Union

from ingestforge.core.security.path import PathSanitizer
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class SafeFileOperations:
    """
    Wrappers for file operations that enforce path sanitization.

    All operations validate that source and destination paths are within
    the allowed base directory before proceeding.

    Example:
        >>> ops = SafeFileOperations(base_dir=Path("/app/data"))
        >>> ops.safe_move(Path("pending/doc.pdf"), Path("completed/doc.pdf"))
        # OK - both paths within /app/data

        >>> ops.safe_move(Path("../../../etc/passwd"), Path("stolen.txt"))
        # Raises PathTraversalError
    """

    def __init__(self, base_dir: Path) -> None:
        """
        Initialize safe file operations.

        Args:
            base_dir: The base directory all operations must stay within.
        """
        self._sanitizer = PathSanitizer(base_dir)

    @property
    def base_dir(self) -> Path:
        """Get the base directory."""
        return self._sanitizer.base_dir

    def validate_path(self, path: Union[str, Path]) -> Path:
        """
        Validate a path is within the base directory.

        Args:
            path: The path to validate.

        Returns:
            The validated, resolved path.

        Raises:
            PathTraversalError: If path escapes base_dir.
        """
        return self._sanitizer.sanitize_path(path)

    def safe_move(
        self, source: Union[str, Path], destination: Union[str, Path]
    ) -> Path:
        """
        Safely move a file, validating both paths.

        Args:
            source: Source file path.
            destination: Destination path.

        Returns:
            The destination path.

        Raises:
            PathTraversalError: If either path escapes base_dir.
            FileNotFoundError: If source doesn't exist.
        """
        assert source is not None, "Source path cannot be None"
        assert destination is not None, "Destination path cannot be None"

        src_path = self.validate_path(source)
        dst_path = self.validate_path(destination)
        assert src_path.is_relative_to(self.base_dir), "Source must be within base_dir"
        assert dst_path.is_relative_to(
            self.base_dir
        ), "Destination must be within base_dir"

        if not src_path.exists():
            # SEC-002: Sanitize path disclosure
            logger.error(f"Source file not found: {src_path}")
            raise FileNotFoundError("Source file not found: [REDACTED]")

        # Ensure destination directory exists
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        shutil.move(str(src_path), str(dst_path))
        return dst_path
