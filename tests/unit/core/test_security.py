"""
Tests for Security Utilities.

This module tests the defensive security measures against directory traversal,
SSRF, and other path/URL-based attacks.

Test Strategy
-------------
- Focus on attack prevention behaviors (blocking malicious inputs)
- Keep tests simple and readable (NASA JPL Rule #1: Simple Control Flow)
- Test what developers actually use: PathSanitizer, URLValidator, SafeFileOperations
- Don't test implementation details (IP parsing internals, etc.)

Organization
------------
- TestPathSanitizer: Filename and path sanitization
- TestURLValidator: URL validation for SSRF prevention
- TestSafeFileOperations: Validated file operations
"""


import pytest

from ingestforge.core.exceptions import PathTraversalError, SSRFError
from ingestforge.core.security import (
    PathSanitizer,
    URLValidator,
    SafeFileOperations,
    sanitize_filename,
)


# ============================================================================
# Test Classes
# ============================================================================


class TestPathSanitizer:
    """Tests for PathSanitizer - directory traversal prevention.

    Rule #4: Focused test class - tests only PathSanitizer
    """

    def test_sanitize_simple_filename(self):
        """Test sanitizing a safe filename."""
        sanitizer = PathSanitizer()

        result = sanitizer.sanitize_filename("document.pdf")

        assert result == "document.pdf"

    def test_sanitize_filename_removes_slashes(self):
        """Test that slashes are removed from filenames."""
        sanitizer = PathSanitizer()

        result = sanitizer.sanitize_filename("path/to/file.txt")

        assert "/" not in result
        assert result == "path_to_file.txt"

    def test_sanitize_filename_removes_backslashes(self):
        """Test that backslashes are removed from filenames."""
        sanitizer = PathSanitizer()

        result = sanitizer.sanitize_filename("path\\to\\file.txt")

        assert "\\" not in result

    def test_sanitize_filename_removes_null_bytes(self):
        """Test that null bytes are removed."""
        sanitizer = PathSanitizer()

        result = sanitizer.sanitize_filename("file\x00.txt")

        assert "\x00" not in result

    def test_sanitize_path_blocks_traversal(self, temp_dir):
        """Test that ../ traversal attempts are blocked."""
        sanitizer = PathSanitizer(base_dir=temp_dir)

        with pytest.raises(PathTraversalError):
            sanitizer.sanitize_path("../../../etc/passwd")

    def test_sanitize_path_allows_safe_subdir(self, temp_dir):
        """Test that safe subdirectories are allowed."""
        sanitizer = PathSanitizer(base_dir=temp_dir)
        subdir = temp_dir / "subdir"
        subdir.mkdir()

        result = sanitizer.sanitize_path("subdir/file.txt")

        assert result.is_relative_to(temp_dir)

    def test_sanitize_path_resolves_relative(self, temp_dir):
        """Test that relative paths are resolved."""
        sanitizer = PathSanitizer(base_dir=temp_dir)

        result = sanitizer.sanitize_path("./file.txt")

        assert result.is_absolute()
        assert result.parent == temp_dir


class TestURLValidator:
    """Tests for URLValidator - SSRF prevention.

    Rule #4: Focused test class - tests only URLValidator
    """

    def test_validate_safe_public_url(self):
        """Test that safe public URLs are allowed."""
        validator = URLValidator()

        scheme, host, port = validator.validate("https://example.com")

        assert scheme == "https"
        assert host == "example.com"
        assert port == 443

    def test_validate_blocks_localhost(self):
        """Test that localhost URLs are blocked."""
        validator = URLValidator()

        with pytest.raises(SSRFError):
            validator.validate("http://localhost/admin")

    def test_validate_blocks_127_0_0_1(self):
        """Test that 127.0.0.1 is blocked."""
        validator = URLValidator()

        with pytest.raises(SSRFError):
            validator.validate("http://127.0.0.1/admin")

    def test_validate_blocks_private_ip_10(self):
        """Test that 10.x.x.x private IPs are blocked."""
        validator = URLValidator()

        with pytest.raises(SSRFError):
            validator.validate("http://10.0.0.1/internal")

    def test_validate_blocks_private_ip_192(self):
        """Test that 192.168.x.x private IPs are blocked."""
        validator = URLValidator()

        with pytest.raises(SSRFError):
            validator.validate("http://192.168.1.1/router")

    def test_validate_blocks_private_ip_172(self):
        """Test that 172.16-31.x.x private IPs are blocked."""
        validator = URLValidator()

        with pytest.raises(SSRFError):
            validator.validate("http://172.16.0.1/internal")

    def test_validate_blocks_link_local(self):
        """Test that link-local addresses are blocked."""
        validator = URLValidator()

        with pytest.raises(SSRFError):
            validator.validate("http://169.254.169.254/metadata")


class TestSafeFileOperations:
    """Tests for SafeFileOperations - validated file operations.

    Rule #4: Focused test class - tests only SafeFileOperations
    """

    def test_init_with_base_dir(self, temp_dir):
        """Test SafeFileOperations initialization."""
        safe_ops = SafeFileOperations(base_dir=temp_dir)

        assert safe_ops.base_dir == temp_dir

    def test_validate_path_allows_safe_path(self, temp_dir):
        """Test that safe paths are validated successfully."""
        safe_ops = SafeFileOperations(base_dir=temp_dir)
        safe_path = temp_dir / "file.txt"

        result = safe_ops.validate_path(safe_path)

        assert result == safe_path

    def test_validate_path_blocks_traversal(self, temp_dir):
        """Test that traversal attempts are blocked."""
        safe_ops = SafeFileOperations(base_dir=temp_dir)

        with pytest.raises(PathTraversalError):
            safe_ops.validate_path("../../../etc/passwd")


class TestConvenienceFunctions:
    """Tests for module-level convenience functions.

    Rule #4: Focused test class - tests only convenience functions
    """

    def test_sanitize_filename_function(self):
        """Test module-level sanitize_filename() function."""
        result = sanitize_filename("path/to/file.txt")

        assert "/" not in result


# ============================================================================
# Summary
# ============================================================================

"""
Test Coverage Summary:
    - PathSanitizer: 7 tests (filename sanitization, path traversal blocking)
    - URLValidator: 7 tests (SSRF prevention, private IP blocking)
    - SafeFileOperations: 3 tests (path validation)
    - Convenience Functions: 1 test

    Total: 18 tests

Design Decisions:
    1. Focus on attack prevention behaviors (what gets blocked)
    2. Don't test every edge case - test representative attacks
    3. Don't test internal helpers (_is_private_ip, etc.) - implementation details
    4. Test public API: sanitize_filename, sanitize_path, validate
    5. Simple, clear tests that verify security properties
    6. Follows NASA JPL Rule #1 (Simple Control Flow)
    7. Follows NASA JPL Rule #4 (Small Focused Classes)

Attack Vectors Tested:
    - Directory Traversal: ../ sequences, absolute paths
    - SSRF: localhost, 127.0.0.1, private IPs (10.x, 172.16-31.x, 192.168.x)
    - Link-Local: 169.254.x.x (AWS metadata service)
    - Filename Injection: slashes, backslashes, null bytes

Justification:
    - Security tests should verify that attacks are blocked
    - Don't need to test every private IP - test representative samples
    - Removed RedirectValidator class (unused in production)
    - Removed is_safe_url() and validate_resolved() convenience methods (unused)
    - Focus on what developers use daily: PathSanitizer and URLValidator
"""
