"""Tests for network_lock module (SEC-002).

Tests the air-gap verification functionality:
- Offline mode enable/disable
- Network whitelist
- Socket interception
- Context manager
"""

import os

from ingestforge.core.security.network_lock import (
    NetworkConfig,
    NetworkSecurityError,
    enable_offline_mode,
    disable_offline_mode,
    is_offline_mode,
    add_to_whitelist,
    offline_context,
    get_network_status,
    check_network_allowed,
)


class TestNetworkConfig:
    """Test NetworkConfig dataclass."""

    def test_config_defaults(self) -> None:
        """Config should have sensible defaults."""
        config = NetworkConfig()
        assert config.enabled is False
        assert config.allow_localhost is True
        assert len(config.whitelist) == 0

    def test_is_allowed_when_disabled(self) -> None:
        """All connections allowed when offline mode disabled."""
        config = NetworkConfig(enabled=False)
        assert config.is_allowed("example.com", 443) is True
        assert config.is_allowed("10.0.0.1", 80) is True

    def test_is_allowed_localhost(self) -> None:
        """Localhost should be allowed by default."""
        config = NetworkConfig(enabled=True, allow_localhost=True)
        assert config.is_allowed("localhost", 8080) is True
        assert config.is_allowed("127.0.0.1", 3000) is True
        assert config.is_allowed("::1", 443) is True

    def test_is_allowed_whitelist(self) -> None:
        """Whitelisted hosts should be allowed."""
        config = NetworkConfig(
            enabled=True,
            whitelist={("api.example.com", 443)},
        )
        assert config.is_allowed("api.example.com", 443) is True
        assert config.is_allowed("api.example.com", 80) is False

    def test_blocked_when_not_whitelisted(self) -> None:
        """Non-whitelisted hosts should be blocked."""
        config = NetworkConfig(
            enabled=True,
            allow_localhost=False,
            whitelist=set(),
        )
        assert config.is_allowed("example.com", 443) is False


class TestOfflineMode:
    """Test offline mode functions."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        disable_offline_mode()

    def test_enable_offline_mode(self) -> None:
        """enable_offline_mode should activate restriction."""
        enable_offline_mode()
        assert is_offline_mode() is True

    def test_disable_offline_mode(self) -> None:
        """disable_offline_mode should deactivate restriction."""
        enable_offline_mode()
        disable_offline_mode()
        assert is_offline_mode() is False

    def test_enable_with_whitelist(self) -> None:
        """enable_offline_mode should accept whitelist."""
        whitelist = {("api.example.com", 443)}
        enable_offline_mode(whitelist=whitelist)

        assert is_offline_mode() is True
        assert check_network_allowed("api.example.com", 443) is True

    def test_env_var_check(self) -> None:
        """is_offline_mode should check environment variable."""
        os.environ["INGESTFORGE_OFFLINE_MODE"] = "1"
        try:
            assert is_offline_mode() is True
        finally:
            os.environ.pop("INGESTFORGE_OFFLINE_MODE", None)


class TestAddToWhitelist:
    """Test add_to_whitelist function."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        disable_offline_mode()

    def test_add_to_whitelist(self) -> None:
        """add_to_whitelist should add entries."""
        enable_offline_mode()
        add_to_whitelist("api.example.com", 443)

        assert check_network_allowed("api.example.com", 443) is True


class TestCheckNetworkAllowed:
    """Test check_network_allowed function."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        disable_offline_mode()

    def test_check_allowed_when_disabled(self) -> None:
        """All should be allowed when offline mode disabled."""
        disable_offline_mode()
        assert check_network_allowed("example.com", 443) is True

    def test_check_localhost_allowed(self) -> None:
        """Localhost should be allowed by default."""
        enable_offline_mode(allow_localhost=True)
        assert check_network_allowed("localhost", 8080) is True

    def test_check_blocked(self) -> None:
        """Non-whitelisted should be blocked."""
        enable_offline_mode(allow_localhost=False)
        assert check_network_allowed("example.com", 443) is False


class TestOfflineContext:
    """Test offline_context context manager."""

    def test_context_enables_offline(self) -> None:
        """Context should enable offline mode."""
        assert is_offline_mode() is False

        with offline_context():
            assert is_offline_mode() is True

        assert is_offline_mode() is False

    def test_context_with_whitelist(self) -> None:
        """Context should accept whitelist."""
        with offline_context(whitelist={("api.example.com", 443)}):
            assert check_network_allowed("api.example.com", 443) is True

    def test_context_restores_state(self) -> None:
        """Context should restore previous state."""
        enable_offline_mode()

        with offline_context():
            pass

        # Should still be enabled
        assert is_offline_mode() is True

        disable_offline_mode()


class TestGetNetworkStatus:
    """Test get_network_status function."""

    def teardown_method(self) -> None:
        """Clean up after each test."""
        disable_offline_mode()

    def test_status_when_disabled(self) -> None:
        """Status should show disabled state."""
        disable_offline_mode()
        status = get_network_status()

        assert status["offline_mode"] is False
        assert "whitelist_count" in status

    def test_status_when_enabled(self) -> None:
        """Status should show enabled state."""
        enable_offline_mode(whitelist={("a.com", 443), ("b.com", 80)})
        status = get_network_status()

        assert status["offline_mode"] is True
        assert status["whitelist_count"] == 2


class TestNetworkSecurityError:
    """Test NetworkSecurityError exception."""

    def test_exception_message(self) -> None:
        """Exception should have informative message."""
        error = NetworkSecurityError("Blocked: example.com:443")
        assert "example.com" in str(error)
        assert "443" in str(error)
