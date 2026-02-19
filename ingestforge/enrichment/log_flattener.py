"""
Log Flattener for Cyber Vertical.

Parses JSON-based logs (CloudTrail, Syslog-ng, ECS format) and extracts
standardized metadata fields for searchable chunks. Part of the CYBER-001
Cyber Vertical implementation.

Architecture Context
--------------------
LogFlattener is an enricher that runs during Stage 4 (Enrich) of the pipeline:

    Split -> Extract -> Refine -> Chunk -> [Enrich: LogFlattener] -> Index

The flattener improves security log analysis by:
1. Normalizing timestamps to ISO 8601
2. Extracting key security metadata (IPs, users, events)
3. Flattening nested JSON for full-text search
4. Masking sensitive data (IPs, PII) by default

Usage Example
-------------
    from ingestforge.enrichment.log_flattener import LogFlattener

    flattener = LogFlattener()

    # Flatten a CloudTrail log entry
    result = flattener.flatten(cloudtrail_json)
    print(result.timestamp)  # ISO 8601 timestamp
    print(result.src_ip)     # Masked: 192.168.xxx.xxx

    # Detect log format
    fmt = flattener.detect_format(log_data)
    # Result: LogFormat.CLOUDTRAIL

    # Extract events from log file
    events = flattener.extract_events(Path("cloudtrail.json"))"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# =============================================================================
# Data Types
# =============================================================================


class LogFormat(Enum):
    """Supported log format types."""

    CLOUDTRAIL = "cloudtrail"
    SYSLOG_NG = "syslog_ng"
    ECS = "ecs"
    GENERIC = "generic"
    UNKNOWN = "unknown"


class EventCategory(Enum):
    """Normalized event categories for security analysis."""

    AUTH = "auth"
    NETWORK = "network"
    PROCESS = "process"
    FILE = "file"
    SYSTEM = "system"
    UNKNOWN = "unknown"


class Severity(Enum):
    """Normalized severity levels."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"
    UNKNOWN = "unknown"


@dataclass
class FlattenedLog:
    """Standardized flattened log record.

    Contains normalized metadata fields extracted from various log formats,
    with privacy protection (IP masking) applied by default.

    Attributes:
        timestamp: ISO 8601 normalized timestamp
        src_ip: Source IP address (masked by default)
        dst_ip: Destination IP address (masked)
        event_id: Event type/name identifier
        event_category: Normalized category (auth, network, process, file)
        severity: Normalized severity level
        user: User/principal identifier (if present)
        host: Source hostname
        message: Flattened log message for search
        raw_fields: Original fields preserved for reference
        log_format: Detected source format
    """

    timestamp: str
    src_ip: str = ""
    dst_ip: str = ""
    event_id: str = ""
    event_category: EventCategory = EventCategory.UNKNOWN
    severity: Severity = Severity.INFO
    user: str = ""
    host: str = ""
    message: str = ""
    raw_fields: Dict[str, Any] = field(default_factory=dict)
    log_format: LogFormat = LogFormat.UNKNOWN


@dataclass
class LogFlattenerConfig:
    """Configuration for LogFlattener.

    Attributes:
        mask_ips: Whether to mask IP addresses (default True)
        ip_whitelist: IPs to leave unmasked
        redact_pii: Whether to redact PII fields
        pii_fields: Field names considered PII
        custom_mappings: Custom field mappings for generic logs
    """

    mask_ips: bool = True
    ip_whitelist: Set[str] = field(default_factory=set)
    redact_pii: bool = True
    pii_fields: Set[str] = field(
        default_factory=lambda: {
            "email",
            "phone",
            "ssn",
            "credit_card",
            "password",
            "secret",
            "token",
            "api_key",
            "private_key",
        }
    )
    custom_mappings: Dict[str, str] = field(default_factory=dict)


# =============================================================================
# Field Mapping Constants
# =============================================================================

CLOUDTRAIL_MAPPINGS: Dict[str, str] = {
    "eventTime": "timestamp",
    "sourceIPAddress": "src_ip",
    "eventName": "event_id",
    "userIdentity.userName": "user",
    "userIdentity.arn": "user",
    "eventType": "event_category",
    "errorCode": "severity",
}

ECS_MAPPINGS: Dict[str, str] = {
    "@timestamp": "timestamp",
    "source.ip": "src_ip",
    "destination.ip": "dst_ip",
    "event.action": "event_id",
    "event.category": "event_category",
    "event.severity": "severity",
    "user.name": "user",
    "host.name": "host",
    "message": "message",
}

SYSLOG_NG_MAPPINGS: Dict[str, str] = {
    "ISODATE": "timestamp",
    "DATE": "timestamp",
    "SOURCEIP": "src_ip",
    "HOST": "host",
    "PROGRAM": "event_id",
    "MSG": "message",
    "MESSAGE": "message",
    "PRIORITY": "severity",
    "FACILITY": "event_category",
}

# Event category detection patterns
CATEGORY_PATTERNS: Dict[EventCategory, List[str]] = {
    EventCategory.AUTH: [
        "login",
        "logout",
        "authenticate",
        "password",
        "credential",
        "session",
        "sso",
        "saml",
        "oauth",
        "mfa",
        "2fa",
    ],
    EventCategory.NETWORK: [
        "connect",
        "disconnect",
        "socket",
        "port",
        "firewall",
        "dns",
        "http",
        "tcp",
        "udp",
        "network",
        "vpn",
    ],
    EventCategory.PROCESS: [
        "process",
        "execute",
        "spawn",
        "fork",
        "kill",
        "terminate",
        "cmd",
        "command",
        "script",
        "binary",
    ],
    EventCategory.FILE: [
        "file",
        "read",
        "write",
        "delete",
        "create",
        "modify",
        "open",
        "close",
        "rename",
        "move",
        "copy",
    ],
    EventCategory.SYSTEM: [
        "system",
        "boot",
        "shutdown",
        "restart",
        "service",
        "kernel",
        "driver",
        "hardware",
        "memory",
        "cpu",
    ],
}

# =============================================================================
# LogFlattener Class
# =============================================================================


@dataclass
class LogFlattener:
    """Flattens JSON logs for security analysis and search.

    Parses various log formats (CloudTrail, Syslog-ng, ECS, generic JSON)
    and extracts standardized metadata fields for consistent querying.

    Examples:
        >>> flattener = LogFlattener()
        >>> result = flattener.flatten('{"eventTime": "2024-01-15T10:30:00Z"}')
        >>> print(result.timestamp)
        2024-01-15T10:30:00Z
    """

    config: LogFlattenerConfig = field(default_factory=LogFlattenerConfig)

    # Regex patterns for IP masking and detection
    _ipv4_pattern: re.Pattern = field(
        default_factory=lambda: re.compile(
            r"\b(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})\b"
        ),
        repr=False,
    )

    def flatten(self, log_data: Union[str, Dict[str, Any]]) -> FlattenedLog:
        """Flatten a log entry to standardized format.

        Args:
            log_data: JSON string or dict containing log data

        Returns:
            FlattenedLog with normalized fields
        """
        parsed = self._parse_input(log_data)
        if parsed is None:
            return self._empty_log()

        log_format = self.detect_format(parsed)
        flattened_dict = self._flatten_dict(parsed)
        extracted = self._extract_fields(flattened_dict, log_format)

        return self._build_log(extracted, flattened_dict, log_format)

    def detect_format(self, log_data: Dict[str, Any]) -> LogFormat:
        """Detect the format of a log entry.

        Args:
            log_data: Parsed log dictionary

        Returns:
            Detected LogFormat enum value
        """
        if self._is_cloudtrail(log_data):
            return LogFormat.CLOUDTRAIL
        if self._is_ecs(log_data):
            return LogFormat.ECS
        if self._is_syslog_ng(log_data):
            return LogFormat.SYSLOG_NG
        if self._has_json_structure(log_data):
            return LogFormat.GENERIC
        return LogFormat.UNKNOWN

    def extract_events(self, log_file: Path) -> List[FlattenedLog]:
        """Extract and flatten events from a log file.

        Supports single JSON objects, JSON arrays, and NDJSON (newline-delimited).

        Args:
            log_file: Path to log file

        Returns:
            List of FlattenedLog records
        """
        if not log_file.exists():
            logger.warning("Log file not found", path=str(log_file))
            return []

        content = log_file.read_text(encoding="utf-8")
        return self._parse_log_content(content)

    def mask_ip(self, ip: str) -> str:
        """Mask an IP address for privacy.

        Masks the last two octets by default: 192.168.1.1 -> 192.168.xxx.xxx

        Args:
            ip: IP address string

        Returns:
            Masked IP address
        """
        if not self.config.mask_ips:
            return ip
        if ip in self.config.ip_whitelist:
            return ip

        match = self._ipv4_pattern.match(ip)
        if not match:
            return ip

        return f"{match.group(1)}.{match.group(2)}.xxx.xxx"

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------

    def _parse_input(self, log_data: Union[str, Dict[str, Any]]) -> Optional[Dict]:
        """Parse input to dictionary."""
        if isinstance(log_data, dict):
            return log_data
        if isinstance(log_data, str):
            try:
                return json.loads(log_data)
            except json.JSONDecodeError as e:
                logger.warning("Failed to parse log JSON", error=str(e))
                return None
        return None

    def _empty_log(self) -> FlattenedLog:
        """Create an empty log record for error cases."""
        return FlattenedLog(
            timestamp=datetime.now(timezone.utc).isoformat(),
            log_format=LogFormat.UNKNOWN,
        )

    def _flatten_dict(
        self, data: Dict[str, Any], prefix: str = "", sep: str = "."
    ) -> Dict[str, Any]:
        """Flatten nested dictionary to dot-notation keys.

        Args:
            data: Nested dictionary
            prefix: Current key prefix
            sep: Separator for nested keys

        Returns:
            Flattened dictionary with dot-notation keys
        """
        items: Dict[str, Any] = {}
        for key, value in data.items():
            new_key = f"{prefix}{sep}{key}" if prefix else key
            if isinstance(value, dict):
                items.update(self._flatten_dict(value, new_key, sep))
            elif isinstance(value, list):
                items[new_key] = self._flatten_list(value)
            else:
                items[new_key] = value
        return items

    def _flatten_list(self, lst: List[Any]) -> str:
        """Flatten a list to searchable string."""
        parts: List[str] = []
        for item in lst:
            if isinstance(item, dict):
                parts.append(json.dumps(item, default=str))
            else:
                parts.append(str(item))
        return ", ".join(parts)

    def _extract_fields(
        self, flattened: Dict[str, Any], log_format: LogFormat
    ) -> Dict[str, Any]:
        """Extract standardized fields based on format."""
        mappings = self._get_mappings(log_format)
        extracted: Dict[str, Any] = {}

        for source_key, target_key in mappings.items():
            if source_key in flattened:
                extracted[target_key] = flattened[source_key]
            # Try partial match for nested keys
            for flat_key, value in flattened.items():
                if flat_key.endswith(source_key):
                    extracted[target_key] = value
                    break

        return extracted

    def _get_mappings(self, log_format: LogFormat) -> Dict[str, str]:
        """Get field mappings for format."""
        if log_format == LogFormat.CLOUDTRAIL:
            return CLOUDTRAIL_MAPPINGS
        if log_format == LogFormat.ECS:
            return ECS_MAPPINGS
        if log_format == LogFormat.SYSLOG_NG:
            return SYSLOG_NG_MAPPINGS
        return self.config.custom_mappings

    def _build_log(
        self,
        extracted: Dict[str, Any],
        raw_fields: Dict[str, Any],
        log_format: LogFormat,
    ) -> FlattenedLog:
        """Build FlattenedLog from extracted fields."""
        timestamp = self._normalize_timestamp(extracted.get("timestamp", ""))
        src_ip = self.mask_ip(str(extracted.get("src_ip", "")))
        dst_ip = self.mask_ip(str(extracted.get("dst_ip", "")))
        event_id = str(extracted.get("event_id", ""))
        category = self._detect_category(event_id, extracted)
        severity = self._normalize_severity(extracted.get("severity"))
        user = self._redact_if_pii("user", str(extracted.get("user", "")))
        host = str(extracted.get("host", ""))
        message = self._build_message(raw_fields)

        return FlattenedLog(
            timestamp=timestamp,
            src_ip=src_ip,
            dst_ip=dst_ip,
            event_id=event_id,
            event_category=category,
            severity=severity,
            user=user,
            host=host,
            message=message,
            raw_fields=raw_fields,
            log_format=log_format,
        )

    def _normalize_timestamp(self, ts: Any) -> str:
        """Normalize timestamp to ISO 8601."""
        if not ts:
            return datetime.now(timezone.utc).isoformat()
        if isinstance(ts, str):
            return self._parse_timestamp_string(ts)
        if isinstance(ts, (int, float)):
            # Unix timestamp
            dt = datetime.fromtimestamp(ts, tz=timezone.utc)
            return dt.isoformat()
        return datetime.now(timezone.utc).isoformat()

    def _parse_timestamp_string(self, ts: str) -> str:
        """Parse various timestamp string formats."""
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
            "%b %d %H:%M:%S",  # Syslog format
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(ts, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt.isoformat()
            except ValueError:
                continue
        # Return as-is if no format matches
        return ts

    def _detect_category(
        self, event_id: str, extracted: Dict[str, Any]
    ) -> EventCategory:
        """Detect event category from event ID and fields."""
        search_text = f"{event_id} {extracted.get('event_category', '')}".lower()

        for category, patterns in CATEGORY_PATTERNS.items():
            if any(pattern in search_text for pattern in patterns):
                return category

        return EventCategory.UNKNOWN

    def _normalize_severity(self, severity: Any) -> Severity:
        """Normalize severity to enum value."""
        if severity is None:
            return Severity.INFO

        sev_str = str(severity).lower()
        if any(s in sev_str for s in ["crit", "fatal", "emergency"]):
            return Severity.CRITICAL
        if any(s in sev_str for s in ["high", "error", "err"]):
            return Severity.HIGH
        if any(s in sev_str for s in ["med", "warn", "warning"]):
            return Severity.MEDIUM
        if any(s in sev_str for s in ["low", "notice"]):
            return Severity.LOW
        if any(s in sev_str for s in ["info", "debug"]):
            return Severity.INFO

        # CloudTrail error codes
        if sev_str and sev_str not in ("none", "null", ""):
            return Severity.HIGH

        return Severity.INFO

    def _redact_if_pii(self, field_name: str, value: str) -> str:
        """Redact value if field is considered PII."""
        if not self.config.redact_pii:
            return value
        if field_name.lower() in self.config.pii_fields:
            return "[REDACTED]"
        return value

    def _build_message(self, fields: Dict[str, Any]) -> str:
        """Build searchable message from flattened fields."""
        parts: List[str] = []
        for key, value in sorted(fields.items()):
            if value is not None and str(value).strip():
                parts.append(f"{key}={value}")
        return " | ".join(parts)

    def _parse_log_content(self, content: str) -> List[FlattenedLog]:
        """Parse log file content in various formats."""
        content = content.strip()
        if not content:
            return []

        # Try JSON array first
        if content.startswith("["):
            return self._parse_json_array(content)

        # Try NDJSON (newline-delimited JSON) - check before single JSON
        # NDJSON has multiple JSON objects on separate lines
        if content.startswith("{") and "\n" in content:
            # Check if it looks like NDJSON (multiple JSON objects)
            lines = [line.strip() for line in content.split("\n") if line.strip()]
            if len(lines) > 1 and all(line.startswith("{") for line in lines):
                return self._parse_ndjson(content)
            # Otherwise treat as single JSON object
            return self._parse_json_object(content)

        # Try single JSON object
        if content.startswith("{"):
            return self._parse_json_object(content)

        # Try NDJSON fallback
        if "\n" in content:
            return self._parse_ndjson(content)

        # Try single JSON object
        return self._parse_single_json(content)

    def _parse_json_array(self, content: str) -> List[FlattenedLog]:
        """Parse JSON array of log entries."""
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [self.flatten(item) for item in data]
            # CloudTrail wraps records
            if isinstance(data, dict) and "Records" in data:
                return [self.flatten(item) for item in data["Records"]]
            return [self.flatten(data)]
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON array", error=str(e))
            return []

    def _parse_json_object(self, content: str) -> List[FlattenedLog]:
        """Parse JSON object, handling CloudTrail Records wrapper."""
        try:
            data = json.loads(content)
            # CloudTrail wraps records in a "Records" array
            if isinstance(data, dict) and "Records" in data:
                records = data["Records"]
                if isinstance(records, list):
                    return [self.flatten(item) for item in records]
            # Single log entry
            return [self.flatten(data)]
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse JSON object", error=str(e))
            return []

    def _parse_ndjson(self, content: str) -> List[FlattenedLog]:
        """Parse newline-delimited JSON."""
        results: List[FlattenedLog] = []
        for line in content.split("\n"):
            line = line.strip()
            if line:
                result = self.flatten(line)
                if result.log_format != LogFormat.UNKNOWN:
                    results.append(result)
        return results

    def _parse_single_json(self, content: str) -> List[FlattenedLog]:
        """Parse single JSON object."""
        result = self.flatten(content)
        return [result] if result.log_format != LogFormat.UNKNOWN else []

    # -------------------------------------------------------------------------
    # Format Detection Methods
    # -------------------------------------------------------------------------

    def _is_cloudtrail(self, data: Dict[str, Any]) -> bool:
        """Check if data matches CloudTrail format."""
        cloudtrail_keys = {"eventVersion", "eventTime", "eventSource", "eventName"}
        return bool(cloudtrail_keys & set(data.keys()))

    def _is_ecs(self, data: Dict[str, Any]) -> bool:
        """Check if data matches ECS format."""
        ecs_keys = {"@timestamp", "ecs", "event", "agent"}
        flat = self._flatten_dict(data)
        flat_keys = set(flat.keys())
        return bool(ecs_keys & set(data.keys())) or any(
            k.startswith("ecs.") or k.startswith("event.") for k in flat_keys
        )

    def _is_syslog_ng(self, data: Dict[str, Any]) -> bool:
        """Check if data matches Syslog-ng JSON output format."""
        syslog_keys = {"HOST", "PROGRAM", "MSG", "FACILITY", "PRIORITY", "ISODATE"}
        return bool(syslog_keys & set(data.keys()))

    def _has_json_structure(self, data: Dict[str, Any]) -> bool:
        """Check if data has valid JSON structure."""
        return isinstance(data, dict) and len(data) > 0


# =============================================================================
# Convenience Functions
# =============================================================================


def flatten_log(
    log_data: Union[str, Dict[str, Any]],
    mask_ips: bool = True,
) -> FlattenedLog:
    """Flatten a log entry to standardized format.

    Args:
        log_data: JSON string or dict containing log data
        mask_ips: Whether to mask IP addresses

    Returns:
        FlattenedLog with normalized fields
    """
    config = LogFlattenerConfig(mask_ips=mask_ips)
    flattener = LogFlattener(config=config)
    return flattener.flatten(log_data)


def detect_log_format(log_data: Union[str, Dict[str, Any]]) -> LogFormat:
    """Detect the format of a log entry.

    Args:
        log_data: JSON string or dict containing log data

    Returns:
        Detected LogFormat enum value
    """
    flattener = LogFlattener()
    if isinstance(log_data, str):
        try:
            parsed = json.loads(log_data)
        except json.JSONDecodeError:
            return LogFormat.UNKNOWN
    else:
        parsed = log_data
    return flattener.detect_format(parsed)


def extract_events_from_file(log_file: Path) -> List[FlattenedLog]:
    """Extract and flatten events from a log file.

    Args:
        log_file: Path to log file

    Returns:
        List of FlattenedLog records
    """
    flattener = LogFlattener()
    return flattener.extract_events(log_file)
