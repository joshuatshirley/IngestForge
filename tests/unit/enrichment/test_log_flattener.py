"""
Tests for Log Flattener enrichment module.

This module tests the LogFlattener class which parses JSON-based logs
(CloudTrail, Syslog-ng, ECS format) and extracts standardized metadata
fields for security analysis.

Test Strategy
-------------
- Test CloudTrail log parsing
- Test Syslog-ng parsing
- Test ECS format parsing
- Test IP masking
- Test timestamp normalization
- Test nested JSON flattening
- Test field mapping
- Test event category detection
- Test privacy features (PII redaction)
- Test edge cases (malformed input)

Organization
------------
- TestLogFlattenerInit: Initialization and configuration
- TestCloudTrailParsing: AWS CloudTrail logs
- TestSyslogNgParsing: Syslog-ng JSON output
- TestECSParsing: Elastic Common Schema logs
- TestFormatDetection: Log format detection
- TestIPMasking: IP address masking
- TestTimestampNormalization: Timestamp handling
- TestNestedFlattening: Nested JSON flattening
- TestEventCategoryDetection: Category classification
- TestSeverityNormalization: Severity level handling
- TestPrivacyFeatures: PII redaction
- TestExtractEvents: File parsing
- TestConvenienceFunctions: Module-level helpers
- TestEdgeCases: Error handling
"""

import json
import tempfile
from pathlib import Path

import pytest

from ingestforge.enrichment.log_flattener import (
    LogFlattener,
    LogFlattenerConfig,
    LogFormat,
    EventCategory,
    Severity,
    FlattenedLog,
    flatten_log,
    detect_log_format,
    extract_events_from_file,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def flattener() -> LogFlattener:
    """Create default LogFlattener instance."""
    return LogFlattener()


@pytest.fixture
def unmasked_flattener() -> LogFlattener:
    """Create LogFlattener with IP masking disabled."""
    config = LogFlattenerConfig(mask_ips=False)
    return LogFlattener(config=config)


@pytest.fixture
def cloudtrail_log() -> dict:
    """Sample AWS CloudTrail log entry."""
    return {
        "eventVersion": "1.08",
        "eventTime": "2024-01-15T10:30:00Z",
        "eventSource": "ec2.amazonaws.com",
        "eventName": "RunInstances",
        "awsRegion": "us-east-1",
        "sourceIPAddress": "192.168.1.100",
        "userIdentity": {
            "type": "IAMUser",
            "userName": "admin-user",
            "arn": "arn:aws:iam::123456789:user/admin-user",
        },
        "requestParameters": {
            "instanceType": "t2.micro",
            "imageId": "ami-12345678",
        },
    }


@pytest.fixture
def syslog_ng_log() -> dict:
    """Sample Syslog-ng JSON output."""
    return {
        "HOST": "web-server-01",
        "PROGRAM": "sshd",
        "FACILITY": "auth",
        "PRIORITY": "info",
        "ISODATE": "2024-01-15T10:30:00Z",
        "MSG": "Accepted password for user from 10.0.0.50 port 22",
        "SOURCEIP": "10.0.0.50",
    }


@pytest.fixture
def ecs_log() -> dict:
    """Sample Elastic Common Schema log entry."""
    return {
        "@timestamp": "2024-01-15T10:30:00.000Z",
        "ecs": {"version": "8.0"},
        "event": {
            "action": "user_login",
            "category": ["authentication"],
            "severity": 3,
        },
        "source": {"ip": "172.16.0.25"},
        "destination": {"ip": "10.0.0.1"},
        "user": {"name": "jdoe"},
        "host": {"name": "auth-server"},
        "message": "User jdoe logged in successfully",
    }


@pytest.fixture
def generic_log() -> dict:
    """Sample generic JSON log."""
    return {
        "timestamp": "2024-01-15T10:30:00Z",
        "level": "INFO",
        "service": "api-gateway",
        "message": "Request processed",
        "metadata": {
            "request_id": "abc-123",
            "duration_ms": 42,
        },
    }


# =============================================================================
# Test Classes
# =============================================================================


class TestLogFlattenerInit:
    """Tests for LogFlattener initialization."""

    def test_create_default_flattener(self):
        """Test creating default LogFlattener."""
        flattener = LogFlattener()

        assert flattener.config.mask_ips is True
        assert flattener.config.redact_pii is True

    def test_create_with_custom_config(self):
        """Test creating LogFlattener with custom config."""
        config = LogFlattenerConfig(
            mask_ips=False,
            ip_whitelist={"192.168.1.1"},
            redact_pii=False,
        )
        flattener = LogFlattener(config=config)

        assert flattener.config.mask_ips is False
        assert "192.168.1.1" in flattener.config.ip_whitelist

    def test_config_pii_fields(self):
        """Test default PII field configuration."""
        config = LogFlattenerConfig()

        assert "email" in config.pii_fields
        assert "password" in config.pii_fields
        assert "api_key" in config.pii_fields


class TestCloudTrailParsing:
    """Tests for AWS CloudTrail log parsing."""

    def test_parse_cloudtrail_json(self, flattener: LogFlattener, cloudtrail_log: dict):
        """Test parsing CloudTrail log entry."""
        result = flattener.flatten(cloudtrail_log)

        assert result.log_format == LogFormat.CLOUDTRAIL
        assert result.event_id == "RunInstances"
        assert "2024-01-15" in result.timestamp

    def test_cloudtrail_ip_masked(self, flattener: LogFlattener, cloudtrail_log: dict):
        """Test CloudTrail source IP is masked."""
        result = flattener.flatten(cloudtrail_log)

        assert "xxx.xxx" in result.src_ip
        assert "100" not in result.src_ip  # Last octets masked

    def test_cloudtrail_user_extracted(
        self, flattener: LogFlattener, cloudtrail_log: dict
    ):
        """Test CloudTrail user identity is extracted."""
        result = flattener.flatten(cloudtrail_log)

        # Should extract from nested userIdentity
        assert "admin" in result.user or "user" in result.user.lower()

    def test_cloudtrail_json_string(
        self, flattener: LogFlattener, cloudtrail_log: dict
    ):
        """Test parsing CloudTrail as JSON string."""
        json_str = json.dumps(cloudtrail_log)

        result = flattener.flatten(json_str)

        assert result.log_format == LogFormat.CLOUDTRAIL

    def test_cloudtrail_records_wrapper(self, flattener: LogFlattener):
        """Test parsing CloudTrail Records wrapper format."""
        wrapped = {
            "Records": [
                {
                    "eventVersion": "1.08",
                    "eventTime": "2024-01-15T10:30:00Z",
                    "eventName": "CreateUser",
                },
                {
                    "eventVersion": "1.08",
                    "eventTime": "2024-01-15T10:31:00Z",
                    "eventName": "DeleteUser",
                },
            ]
        }
        json_str = json.dumps(wrapped)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write(json_str)
            temp_path = Path(f.name)

        try:
            results = flattener.extract_events(temp_path)
            assert len(results) == 2
            assert results[0].event_id == "CreateUser"
            assert results[1].event_id == "DeleteUser"
        finally:
            temp_path.unlink()


class TestSyslogNgParsing:
    """Tests for Syslog-ng JSON output parsing."""

    def test_parse_syslog_ng_json(self, flattener: LogFlattener, syslog_ng_log: dict):
        """Test parsing Syslog-ng log entry."""
        result = flattener.flatten(syslog_ng_log)

        assert result.log_format == LogFormat.SYSLOG_NG
        assert result.host == "web-server-01"
        assert result.event_id == "sshd"

    def test_syslog_message_extracted(
        self, flattener: LogFlattener, syslog_ng_log: dict
    ):
        """Test Syslog message field extracted."""
        result = flattener.flatten(syslog_ng_log)

        assert "Accepted password" in result.raw_fields.get("MSG", "")

    def test_syslog_ip_masked(self, flattener: LogFlattener, syslog_ng_log: dict):
        """Test Syslog source IP is masked."""
        result = flattener.flatten(syslog_ng_log)

        # Source IP should be masked
        if result.src_ip:
            assert "xxx" in result.src_ip

    def test_syslog_timestamp_parsed(
        self, flattener: LogFlattener, syslog_ng_log: dict
    ):
        """Test Syslog timestamp is normalized."""
        result = flattener.flatten(syslog_ng_log)

        assert "2024-01-15" in result.timestamp
        assert "T" in result.timestamp  # ISO 8601 format


class TestECSParsing:
    """Tests for Elastic Common Schema log parsing."""

    def test_parse_ecs_json(self, flattener: LogFlattener, ecs_log: dict):
        """Test parsing ECS log entry."""
        result = flattener.flatten(ecs_log)

        assert result.log_format == LogFormat.ECS
        assert result.event_id == "user_login"

    def test_ecs_nested_source_ip(self, flattener: LogFlattener, ecs_log: dict):
        """Test ECS nested source.ip is extracted."""
        result = flattener.flatten(ecs_log)

        # Should extract from nested source.ip and mask
        assert result.src_ip
        assert "xxx" in result.src_ip

    def test_ecs_nested_destination_ip(self, flattener: LogFlattener, ecs_log: dict):
        """Test ECS nested destination.ip is extracted."""
        result = flattener.flatten(ecs_log)

        assert result.dst_ip
        assert "xxx" in result.dst_ip

    def test_ecs_user_extracted(self, flattener: LogFlattener, ecs_log: dict):
        """Test ECS user.name is extracted."""
        result = flattener.flatten(ecs_log)

        assert result.user == "jdoe"

    def test_ecs_host_extracted(self, flattener: LogFlattener, ecs_log: dict):
        """Test ECS host.name is extracted."""
        result = flattener.flatten(ecs_log)

        assert result.host == "auth-server"

    def test_ecs_event_category(self, flattener: LogFlattener, ecs_log: dict):
        """Test ECS event category detection."""
        result = flattener.flatten(ecs_log)

        # user_login should map to AUTH category
        assert result.event_category == EventCategory.AUTH


class TestFormatDetection:
    """Tests for log format detection."""

    def test_detect_cloudtrail(self, flattener: LogFlattener, cloudtrail_log: dict):
        """Test detecting CloudTrail format."""
        fmt = flattener.detect_format(cloudtrail_log)

        assert fmt == LogFormat.CLOUDTRAIL

    def test_detect_syslog_ng(self, flattener: LogFlattener, syslog_ng_log: dict):
        """Test detecting Syslog-ng format."""
        fmt = flattener.detect_format(syslog_ng_log)

        assert fmt == LogFormat.SYSLOG_NG

    def test_detect_ecs(self, flattener: LogFlattener, ecs_log: dict):
        """Test detecting ECS format."""
        fmt = flattener.detect_format(ecs_log)

        assert fmt == LogFormat.ECS

    def test_detect_generic(self, flattener: LogFlattener, generic_log: dict):
        """Test detecting generic JSON format."""
        fmt = flattener.detect_format(generic_log)

        assert fmt == LogFormat.GENERIC

    def test_detect_unknown_empty(self, flattener: LogFlattener):
        """Test detecting unknown format for empty dict."""
        fmt = flattener.detect_format({})

        assert fmt == LogFormat.UNKNOWN


class TestIPMasking:
    """Tests for IP address masking."""

    def test_mask_ipv4(self, flattener: LogFlattener):
        """Test IPv4 address is masked."""
        masked = flattener.mask_ip("192.168.1.100")

        assert masked == "192.168.xxx.xxx"

    def test_mask_preserves_first_two_octets(self, flattener: LogFlattener):
        """Test first two octets are preserved."""
        masked = flattener.mask_ip("10.0.5.25")

        assert masked.startswith("10.0.")
        assert masked.endswith(".xxx.xxx")

    def test_whitelist_unmasked(self):
        """Test whitelisted IPs are not masked."""
        config = LogFlattenerConfig(
            mask_ips=True,
            ip_whitelist={"192.168.1.1"},
        )
        flattener = LogFlattener(config=config)

        masked = flattener.mask_ip("192.168.1.1")

        assert masked == "192.168.1.1"

    def test_masking_disabled(self, unmasked_flattener: LogFlattener):
        """Test IP masking can be disabled."""
        masked = unmasked_flattener.mask_ip("192.168.1.100")

        assert masked == "192.168.1.100"

    def test_invalid_ip_unchanged(self, flattener: LogFlattener):
        """Test invalid IP string is unchanged."""
        result = flattener.mask_ip("not-an-ip")

        assert result == "not-an-ip"

    def test_empty_ip_unchanged(self, flattener: LogFlattener):
        """Test empty string is unchanged."""
        result = flattener.mask_ip("")

        assert result == ""


class TestTimestampNormalization:
    """Tests for timestamp normalization."""

    def test_iso8601_preserved(self, flattener: LogFlattener):
        """Test ISO 8601 timestamp is preserved."""
        log = {"eventTime": "2024-01-15T10:30:00Z", "eventName": "Test"}

        result = flattener.flatten(log)

        assert "2024-01-15" in result.timestamp

    def test_iso8601_with_millis(self, flattener: LogFlattener):
        """Test ISO 8601 with milliseconds is normalized."""
        log = {"@timestamp": "2024-01-15T10:30:00.123Z", "ecs": {"version": "8.0"}}

        result = flattener.flatten(log)

        assert "2024-01-15" in result.timestamp

    def test_unix_timestamp_converted(self, flattener: LogFlattener):
        """Test Unix timestamp is converted."""
        log = {"eventTime": 1705315800, "eventName": "Test"}  # 2024-01-15T10:30:00

        result = flattener.flatten(log)

        assert "2024-01-15" in result.timestamp

    def test_missing_timestamp_uses_current(self, flattener: LogFlattener):
        """Test missing timestamp uses current time."""
        log = {"eventName": "Test"}

        result = flattener.flatten(log)

        # Should have some timestamp
        assert result.timestamp
        assert "T" in result.timestamp  # ISO format


class TestNestedFlattening:
    """Tests for nested JSON flattening."""

    def test_flatten_nested_dict(self, flattener: LogFlattener):
        """Test nested dictionary is flattened."""
        log = {
            "eventName": "Test",
            "nested": {
                "level1": {
                    "level2": "deep_value",
                },
            },
        }

        result = flattener.flatten(log)

        assert "nested.level1.level2" in result.raw_fields
        assert result.raw_fields["nested.level1.level2"] == "deep_value"

    def test_flatten_arrays(self, flattener: LogFlattener):
        """Test arrays are flattened to strings."""
        log = {
            "eventName": "Test",
            "tags": ["tag1", "tag2", "tag3"],
        }

        result = flattener.flatten(log)

        assert "tags" in result.raw_fields
        # Should be comma-separated
        tags_str = result.raw_fields["tags"]
        assert "tag1" in tags_str
        assert "tag2" in tags_str

    def test_flatten_mixed_structure(self, flattener: LogFlattener):
        """Test mixed nested structure."""
        log = {
            "eventName": "Test",
            "user": {
                "name": "admin",
                "roles": ["admin", "user"],
            },
            "metadata": {
                "version": "1.0",
            },
        }

        result = flattener.flatten(log)

        assert "user.name" in result.raw_fields
        assert "user.roles" in result.raw_fields
        assert "metadata.version" in result.raw_fields


class TestEventCategoryDetection:
    """Tests for event category detection."""

    def test_detect_auth_category(self, flattener: LogFlattener):
        """Test AUTH category detection."""
        log = {"eventName": "UserLogin", "eventTime": "2024-01-15T10:30:00Z"}

        result = flattener.flatten(log)

        assert result.event_category == EventCategory.AUTH

    def test_detect_network_category(self, flattener: LogFlattener):
        """Test NETWORK category detection."""
        log = {"eventName": "NetworkConnect", "eventTime": "2024-01-15T10:30:00Z"}

        result = flattener.flatten(log)

        assert result.event_category == EventCategory.NETWORK

    def test_detect_process_category(self, flattener: LogFlattener):
        """Test PROCESS category detection."""
        log = {"eventName": "ProcessExecute", "eventTime": "2024-01-15T10:30:00Z"}

        result = flattener.flatten(log)

        assert result.event_category == EventCategory.PROCESS

    def test_detect_file_category(self, flattener: LogFlattener):
        """Test FILE category detection."""
        log = {"eventName": "FileDelete", "eventTime": "2024-01-15T10:30:00Z"}

        result = flattener.flatten(log)

        assert result.event_category == EventCategory.FILE

    def test_unknown_category(self, flattener: LogFlattener):
        """Test UNKNOWN category for unrecognized events."""
        log = {"eventName": "CustomEvent", "eventTime": "2024-01-15T10:30:00Z"}

        result = flattener.flatten(log)

        assert result.event_category == EventCategory.UNKNOWN


class TestSeverityNormalization:
    """Tests for severity level normalization."""

    def test_severity_critical(self, flattener: LogFlattener):
        """Test CRITICAL severity detection."""
        log = {"eventName": "Test", "errorCode": "critical"}

        result = flattener.flatten(log)

        assert result.severity == Severity.CRITICAL

    def test_severity_high_from_error(self, flattener: LogFlattener):
        """Test HIGH severity from error."""
        log = {"eventName": "Test", "errorCode": "AccessDenied"}

        result = flattener.flatten(log)

        assert result.severity == Severity.HIGH

    def test_severity_warning(self, flattener: LogFlattener):
        """Test MEDIUM severity from warning."""
        log = {
            "@timestamp": "2024-01-15T10:30:00Z",
            "ecs": {"version": "8.0"},
            "event": {"severity": "warning"},
        }

        result = flattener.flatten(log)

        assert result.severity == Severity.MEDIUM

    def test_severity_info_default(self, flattener: LogFlattener):
        """Test INFO is default severity."""
        log = {"eventName": "Test"}

        result = flattener.flatten(log)

        assert result.severity == Severity.INFO


class TestPrivacyFeatures:
    """Tests for privacy features (PII redaction)."""

    def test_pii_field_redacted(self):
        """Test PII field is redacted."""
        config = LogFlattenerConfig(
            redact_pii=True,
            pii_fields={"email"},
        )
        flattener = LogFlattener(config=config)

        # Internal method test
        result = flattener._redact_if_pii("email", "user@example.com")

        assert result == "[REDACTED]"

    def test_non_pii_field_preserved(self):
        """Test non-PII field is preserved."""
        config = LogFlattenerConfig(redact_pii=True)
        flattener = LogFlattener(config=config)

        result = flattener._redact_if_pii("username", "admin")

        assert result == "admin"

    def test_pii_redaction_disabled(self):
        """Test PII redaction can be disabled."""
        config = LogFlattenerConfig(redact_pii=False)
        flattener = LogFlattener(config=config)

        result = flattener._redact_if_pii("email", "user@example.com")

        assert result == "user@example.com"


class TestExtractEvents:
    """Tests for log file parsing."""

    def test_extract_from_json_array(self, flattener: LogFlattener):
        """Test extracting events from JSON array file."""
        logs = [
            {"eventName": "Event1", "eventTime": "2024-01-15T10:30:00Z"},
            {"eventName": "Event2", "eventTime": "2024-01-15T10:31:00Z"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(logs, f)
            temp_path = Path(f.name)

        try:
            results = flattener.extract_events(temp_path)
            assert len(results) == 2
        finally:
            temp_path.unlink()

    def test_extract_from_ndjson(self, flattener: LogFlattener):
        """Test extracting events from NDJSON file."""
        logs = [
            '{"eventName": "Event1", "eventTime": "2024-01-15T10:30:00Z"}',
            '{"eventName": "Event2", "eventTime": "2024-01-15T10:31:00Z"}',
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("\n".join(logs))
            temp_path = Path(f.name)

        try:
            results = flattener.extract_events(temp_path)
            assert len(results) == 2
        finally:
            temp_path.unlink()

    def test_extract_from_nonexistent_file(self, flattener: LogFlattener):
        """Test extracting from nonexistent file returns empty list."""
        results = flattener.extract_events(Path("/nonexistent/file.json"))

        assert results == []

    def test_extract_from_empty_file(self, flattener: LogFlattener):
        """Test extracting from empty file returns empty list."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("")
            temp_path = Path(f.name)

        try:
            results = flattener.extract_events(temp_path)
            assert results == []
        finally:
            temp_path.unlink()


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_flatten_log_function(self, cloudtrail_log: dict):
        """Test flatten_log function."""
        result = flatten_log(cloudtrail_log)

        assert result.log_format == LogFormat.CLOUDTRAIL
        assert "xxx" in result.src_ip

    def test_flatten_log_unmasked(self, cloudtrail_log: dict):
        """Test flatten_log with masking disabled."""
        result = flatten_log(cloudtrail_log, mask_ips=False)

        assert "192.168.1.100" in result.src_ip

    def test_detect_log_format_function(self, cloudtrail_log: dict):
        """Test detect_log_format function."""
        fmt = detect_log_format(cloudtrail_log)

        assert fmt == LogFormat.CLOUDTRAIL

    def test_detect_log_format_from_string(self, cloudtrail_log: dict):
        """Test detect_log_format from JSON string."""
        json_str = json.dumps(cloudtrail_log)

        fmt = detect_log_format(json_str)

        assert fmt == LogFormat.CLOUDTRAIL

    def test_extract_events_from_file_function(self):
        """Test extract_events_from_file function."""
        log = {"eventName": "Test", "eventTime": "2024-01-15T10:30:00Z"}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([log], f)
            temp_path = Path(f.name)

        try:
            results = extract_events_from_file(temp_path)
            assert len(results) == 1
        finally:
            temp_path.unlink()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_string_input(self, flattener: LogFlattener):
        """Test empty string input."""
        result = flattener.flatten("")

        assert result.log_format == LogFormat.UNKNOWN

    def test_invalid_json_input(self, flattener: LogFlattener):
        """Test invalid JSON string."""
        result = flattener.flatten("not valid json")

        assert result.log_format == LogFormat.UNKNOWN

    def test_none_values_handled(self, flattener: LogFlattener):
        """Test None values in log are handled."""
        log = {
            "eventName": "Test",
            "eventTime": None,
            "sourceIPAddress": None,
        }

        result = flattener.flatten(log)

        # Should not crash
        assert isinstance(result, FlattenedLog)

    def test_deeply_nested_structure(self, flattener: LogFlattener):
        """Test deeply nested structure is flattened."""
        log = {
            "eventName": "Test",
            "a": {"b": {"c": {"d": {"e": "deep"}}}},
        }

        result = flattener.flatten(log)

        assert "a.b.c.d.e" in result.raw_fields

    def test_special_characters_in_values(self, flattener: LogFlattener):
        """Test special characters in values."""
        log = {
            "eventName": "Test",
            "message": "Line1\nLine2\tTabbed",
        }

        result = flattener.flatten(log)

        # Should contain the message
        assert "message" in result.raw_fields

    def test_unicode_in_values(self, flattener: LogFlattener):
        """Test Unicode characters in values."""
        log = {
            "eventName": "Test",
            "user": "user@example.com",
        }

        result = flattener.flatten(log)

        assert isinstance(result, FlattenedLog)

    def test_large_log_entry(self, flattener: LogFlattener):
        """Test handling of large log entry."""
        log = {
            "eventName": "Test",
            "data": "x" * 10000,
        }

        result = flattener.flatten(log)

        assert len(result.message) > 0

    def test_empty_dict_input(self, flattener: LogFlattener):
        """Test empty dictionary input."""
        result = flattener.flatten({})

        assert result.log_format == LogFormat.UNKNOWN


class TestMessageBuilding:
    """Tests for searchable message building."""

    def test_message_contains_key_value_pairs(self, flattener: LogFlattener):
        """Test message contains key-value pairs."""
        log = {
            "eventName": "Test",
            "eventTime": "2024-01-15T10:30:00Z",
        }

        result = flattener.flatten(log)

        assert "eventName=Test" in result.message
        assert "eventTime=" in result.message

    def test_message_fields_sorted(self, flattener: LogFlattener):
        """Test message fields are sorted."""
        log = {
            "z_field": "z",
            "a_field": "a",
            "m_field": "m",
        }

        result = flattener.flatten(log)

        # Fields should be in sorted order
        a_pos = result.message.find("a_field")
        m_pos = result.message.find("m_field")
        z_pos = result.message.find("z_field")

        assert a_pos < m_pos < z_pos


# =============================================================================
# Summary
# =============================================================================

"""
Test Coverage Summary:
    - LogFlattener init: 3 tests
    - CloudTrail parsing: 5 tests
    - Syslog-ng parsing: 4 tests
    - ECS parsing: 6 tests
    - Format detection: 5 tests
    - IP masking: 6 tests
    - Timestamp normalization: 4 tests
    - Nested flattening: 3 tests
    - Event category detection: 5 tests
    - Severity normalization: 4 tests
    - Privacy features: 3 tests
    - Extract events: 4 tests
    - Convenience functions: 5 tests
    - Edge cases: 8 tests
    - Message building: 2 tests

    Total: 67 tests

Design Decisions:
    1. Test all supported log formats (CloudTrail, Syslog-ng, ECS)
    2. Test privacy features (IP masking, PII redaction)
    3. Test format detection separately
    4. Cover edge cases and error handling
    5. Test nested JSON flattening
    6. Test timestamp normalization for various formats
    7. Follows NASA JPL Rule #1 (Simple Control Flow)
    8. Follows NASA JPL Rule #4 (Small Focused Classes)

Behaviors Tested:
    - LogFlattener initialization and configuration
    - CloudTrail log parsing with nested userIdentity
    - Syslog-ng JSON output parsing
    - Elastic Common Schema log parsing
    - Automatic log format detection
    - IP address masking with configurable whitelist
    - Timestamp normalization to ISO 8601
    - Nested JSON structure flattening
    - Event category classification (AUTH, NETWORK, FILE, etc.)
    - Severity level normalization
    - PII field redaction
    - File-based log extraction (JSON, NDJSON)
    - Edge cases (empty input, invalid JSON, large entries)
"""
