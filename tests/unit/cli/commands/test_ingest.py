"""Unit tests for ingest command YouTube URL detection.

Tests the is_youtube_url helper function and URL routing.
"""


from ingestforge.cli.commands.ingest import is_youtube_url


class TestIsYouTubeURL:
    """Tests for YouTube URL detection."""

    def test_youtube_standard_url(self) -> None:
        """Test standard YouTube watch URL."""
        assert is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_youtube_short_url(self) -> None:
        """Test YouTube short URL."""
        assert is_youtube_url("https://youtu.be/dQw4w9WgXcQ") is True

    def test_youtube_http(self) -> None:
        """Test HTTP YouTube URL."""
        assert is_youtube_url("http://www.youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_youtube_no_www(self) -> None:
        """Test YouTube URL without www."""
        assert is_youtube_url("https://youtube.com/watch?v=dQw4w9WgXcQ") is True

    def test_youtube_embed(self) -> None:
        """Test YouTube embed URL."""
        assert is_youtube_url("https://www.youtube.com/embed/dQw4w9WgXcQ") is True

    def test_youtube_shorts(self) -> None:
        """Test YouTube Shorts URL."""
        assert is_youtube_url("https://www.youtube.com/shorts/dQw4w9WgXcQ") is True

    def test_youtube_with_params(self) -> None:
        """Test YouTube URL with extra parameters."""
        url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ&list=PLx0sYbCqOb8TBPRdmBHs"
        assert is_youtube_url(url) is True

    def test_youtube_with_timestamp(self) -> None:
        """Test YouTube URL with timestamp."""
        assert (
            is_youtube_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=120s") is True
        )

    def test_not_url(self) -> None:
        """Test non-URL string."""
        assert is_youtube_url("document.pdf") is False

    def test_other_website(self) -> None:
        """Test URL from different website."""
        assert is_youtube_url("https://example.com/video") is False

    def test_empty_string(self) -> None:
        """Test empty string."""
        assert is_youtube_url("") is False

    def test_file_path(self) -> None:
        """Test file path."""
        assert is_youtube_url("/path/to/document.pdf") is False

    def test_whitespace_only(self) -> None:
        """Test whitespace-only string."""
        assert is_youtube_url("   ") is False

    def test_case_insensitive(self) -> None:
        """Test case insensitivity."""
        assert is_youtube_url("HTTPS://WWW.YOUTUBE.COM/watch?v=dQw4w9WgXcQ") is True
