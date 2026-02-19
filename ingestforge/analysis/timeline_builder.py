"""
Incident Timeline Builder for Cyber Vertical (CYBER-004).

Sorts disparate log sources into a unified security timeline with links to
evidence chunks. Generates output in Markdown, JSON, and ASCII table formats.

Architecture Context
--------------------
TimelineBuilder integrates with the LogFlattener enricher to correlate events
across multiple log sources into a coherent chronological timeline:

    [CloudTrail] ──┐
    [Syslog-ng] ───┼─> [LogFlattener] -> [TimelineBuilder] -> Timeline
    [ECS Logs] ────┘

Usage Example
-------------
    from ingestforge.analysis.timeline_builder import TimelineBuilder
    from datetime import datetime

    builder = TimelineBuilder()

    # Add events from flattened logs
    builder.add_event(flattened_log)

    # Add events from chunks
    builder.add_chunk(chunk_dict)

    # Build timeline for date range
    entries = builder.build(
        start=datetime(2024, 1, 1),
        end=datetime(2024, 1, 31),
    )

    # Export to various formats
    markdown = builder.to_markdown()
    json_output = builder.to_json()
    table = builder.to_ascii_table()"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from ingestforge.core.logging import get_logger
from ingestforge.enrichment.log_flattener import FlattenedLog, EventCategory

logger = get_logger(__name__)

# =============================================================================
# Data Types
# =============================================================================


@dataclass
class TimelineEntry:
    """A single event in the security timeline.

    Attributes:
        timestamp: Event timestamp (datetime object)
        event_type: Type of event (e.g., login, file_access)
        description: Human-readable description
        source: Log source (CloudTrail, Syslog, etc.)
        chunk_ids: Reference to evidence chunk IDs
        severity: Event severity level
        actors: Users/IPs involved in event
        attack_techniques: MITRE ATT&CK technique IDs if applicable
    """

    timestamp: datetime
    event_type: str
    description: str
    source: str
    chunk_ids: List[str] = field(default_factory=list)
    severity: str = "info"
    actors: List[str] = field(default_factory=list)
    attack_techniques: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert entry to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "description": self.description,
            "source": self.source,
            "chunk_ids": self.chunk_ids,
            "severity": self.severity,
            "actors": self.actors,
            "attack_techniques": self.attack_techniques,
        }


@dataclass
class CorrelationGroup:
    """A group of correlated events across sources.

    Attributes:
        entries: List of correlated timeline entries
        correlation_type: Type of correlation (actor, timewindow, technique)
        confidence: Confidence score (0.0 - 1.0)
    """

    entries: List[TimelineEntry]
    correlation_type: str
    confidence: float = 0.0


# =============================================================================
# TimelineBuilder Class
# =============================================================================


class TimelineBuilder:
    """Build security incident timelines from disparate log sources.

    Sorts events chronologically, correlates across sources, and generates
    output in multiple formats (Markdown, JSON, ASCII).
    """

    # Time window for correlation (seconds)
    DEFAULT_CORRELATION_WINDOW: int = 300  # 5 minutes

    def __init__(
        self,
        correlation_window: int = DEFAULT_CORRELATION_WINDOW,
    ) -> None:
        """Initialize TimelineBuilder.

        Args:
            correlation_window: Time window in seconds for event correlation
        """
        self._entries: List[TimelineEntry] = []
        self._correlation_window = correlation_window
        self._filtered_entries: Optional[List[TimelineEntry]] = None

    @property
    def entries(self) -> List[TimelineEntry]:
        """Get all timeline entries."""
        return self._entries

    @property
    def entry_count(self) -> int:
        """Get count of timeline entries."""
        return len(self._entries)

    def add_event(self, log: FlattenedLog) -> None:
        """Add an event from a FlattenedLog record.

        Args:
            log: FlattenedLog from the LogFlattener enricher
        """
        timestamp = self._parse_timestamp(log.timestamp)
        actors = self._extract_actors(log)
        techniques = self._detect_techniques(log)

        entry = TimelineEntry(
            timestamp=timestamp,
            event_type=log.event_id or log.event_category.value,
            description=self._build_description(log),
            source=log.log_format.value,
            severity=log.severity.value,
            actors=actors,
            attack_techniques=techniques,
        )
        self._entries.append(entry)
        self._filtered_entries = None  # Reset cached filter

    def add_chunk(self, chunk: Dict[str, Any]) -> None:
        """Add an event from a chunk dictionary.

        Args:
            chunk: Dictionary containing chunk data with metadata
        """
        metadata = chunk.get("metadata", {})
        timestamp = self._extract_chunk_timestamp(chunk, metadata)
        chunk_id = chunk.get("chunk_id", chunk.get("id", ""))

        entry = TimelineEntry(
            timestamp=timestamp,
            event_type=metadata.get("event_type", "unknown"),
            description=self._extract_chunk_description(chunk),
            source=metadata.get("source", metadata.get("source_file", "unknown")),
            chunk_ids=[chunk_id] if chunk_id else [],
            severity=metadata.get("severity", "info"),
            actors=self._extract_chunk_actors(metadata),
            attack_techniques=metadata.get("attack_techniques", []),
        )
        self._entries.append(entry)
        self._filtered_entries = None

    def build(
        self,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> List[TimelineEntry]:
        """Build timeline entries sorted chronologically.

        Args:
            start: Start of date range (inclusive)
            end: End of date range (inclusive)

        Returns:
            List of TimelineEntry sorted by timestamp
        """
        filtered = self._filter_by_range(start, end)
        sorted_entries = sorted(filtered, key=lambda e: e.timestamp)
        self._filtered_entries = sorted_entries
        return sorted_entries

    def to_markdown(self) -> str:
        """Generate Markdown timeline with linked evidence.

        Returns:
            Markdown-formatted timeline string
        """
        entries = self._get_built_entries()
        if not entries:
            return "# Security Timeline\n\nNo events found.\n"

        return self._format_markdown(entries)

    def to_json(self) -> str:
        """Generate JSON output for programmatic access.

        Returns:
            JSON string of timeline data
        """
        entries = self._get_built_entries()
        data = {
            "timeline": [e.to_dict() for e in entries],
            "summary": self._build_summary(entries),
        }
        return json.dumps(data, indent=2, default=str)

    def to_ascii_table(self) -> str:
        """Generate ASCII table for CLI display.

        Returns:
            ASCII table string
        """
        entries = self._get_built_entries()
        if not entries:
            return "No events found."

        return self._format_ascii_table(entries)

    def correlate_events(self) -> List[CorrelationGroup]:
        """Correlate events across multiple data sources.

        Groups events that share actors, occur within the correlation window,
        or reference the same ATT&CK techniques.

        Returns:
            List of CorrelationGroup containing related events
        """
        entries = self._get_built_entries()
        if not entries:
            return []

        groups: List[CorrelationGroup] = []
        groups.extend(self._correlate_by_actor(entries))
        groups.extend(self._correlate_by_timewindow(entries))
        groups.extend(self._correlate_by_technique(entries))

        return self._deduplicate_groups(groups)

    def clear(self) -> None:
        """Clear all timeline entries."""
        self._entries = []
        self._filtered_entries = None

    # -------------------------------------------------------------------------
    # Private Helper Methods - Event Addition
    # -------------------------------------------------------------------------

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse ISO timestamp string to datetime."""
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%S.%f%z",
            "%Y-%m-%dT%H:%M:%S",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(ts_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                return dt
            except ValueError:
                continue

        logger.warning("Could not parse timestamp", ts=ts_str)
        return datetime.now(timezone.utc)

    def _extract_actors(self, log: FlattenedLog) -> List[str]:
        """Extract actors (users, IPs) from FlattenedLog."""
        actors = []
        if log.user and log.user != "[REDACTED]":
            actors.append(log.user)
        if log.src_ip and "xxx" not in log.src_ip:
            actors.append(log.src_ip)
        if log.dst_ip and "xxx" not in log.dst_ip:
            actors.append(log.dst_ip)
        return actors

    def _detect_techniques(self, log: FlattenedLog) -> List[str]:
        """Detect ATT&CK techniques based on event patterns."""
        techniques = []
        event = (log.event_id or "").lower()
        category = log.event_category

        technique_map = self._get_technique_map()
        for pattern, technique_id in technique_map.items():
            if pattern in event:
                techniques.append(technique_id)

        if category == EventCategory.AUTH and "fail" in event:
            techniques.append("T1110")  # Brute Force
        if category == EventCategory.FILE and "delete" in event:
            techniques.append("T1070")  # Indicator Removal

        return list(set(techniques))

    def _get_technique_map(self) -> Dict[str, str]:
        """Get mapping of event patterns to ATT&CK techniques."""
        return {
            "runinstances": "T1578",  # Modify Cloud Compute
            "createuser": "T1136",  # Create Account
            "deleteuser": "T1531",  # Account Access Removal
            "putobject": "T1537",  # Transfer Data to Cloud Account
            "getobject": "T1530",  # Data from Cloud Storage
            "login": "T1078",  # Valid Accounts
            "password": "T1110",  # Brute Force
            "execute": "T1059",  # Command and Scripting
        }

    def _build_description(self, log: FlattenedLog) -> str:
        """Build human-readable description from log."""
        parts = []
        if log.event_id:
            parts.append(log.event_id)
        if log.user and log.user != "[REDACTED]":
            parts.append(f"by {log.user}")
        if log.src_ip:
            parts.append(f"from {log.src_ip}")
        if log.host:
            parts.append(f"on {log.host}")

        if parts:
            return " ".join(parts)
        return log.message[:100] if log.message else "Unknown event"

    # -------------------------------------------------------------------------
    # Private Helper Methods - Chunk Processing
    # -------------------------------------------------------------------------

    def _extract_chunk_timestamp(
        self, chunk: Dict[str, Any], metadata: Dict[str, Any]
    ) -> datetime:
        """Extract timestamp from chunk data."""
        ts_fields = ["timestamp", "event_time", "created_at", "date"]
        for field_name in ts_fields:
            ts_val = metadata.get(field_name) or chunk.get(field_name)
            if ts_val:
                if isinstance(ts_val, datetime):
                    return ts_val
                if isinstance(ts_val, str):
                    return self._parse_timestamp(ts_val)

        return datetime.now(timezone.utc)

    def _extract_chunk_description(self, chunk: Dict[str, Any]) -> str:
        """Extract description from chunk content."""
        content = chunk.get("content", chunk.get("text", ""))
        if content:
            return content[:200].strip()
        return "Event from chunk"

    def _extract_chunk_actors(self, metadata: Dict[str, Any]) -> List[str]:
        """Extract actors from chunk metadata."""
        actors = []
        for field_name in ["user", "actor", "src_ip", "source_ip"]:
            val = metadata.get(field_name)
            if val and val != "[REDACTED]":
                actors.append(str(val))
        return actors

    # -------------------------------------------------------------------------
    # Private Helper Methods - Timeline Building
    # -------------------------------------------------------------------------

    def _filter_by_range(
        self, start: Optional[datetime], end: Optional[datetime]
    ) -> List[TimelineEntry]:
        """Filter entries by date range."""
        if not start and not end:
            return self._entries.copy()

        filtered = []
        for entry in self._entries:
            if start and entry.timestamp < start:
                continue
            if end and entry.timestamp > end:
                continue
            filtered.append(entry)
        return filtered

    def _get_built_entries(self) -> List[TimelineEntry]:
        """Get entries, building if necessary."""
        if self._filtered_entries is None:
            return self.build()
        return self._filtered_entries

    # -------------------------------------------------------------------------
    # Private Helper Methods - Output Formatting
    # -------------------------------------------------------------------------

    def _format_markdown(self, entries: List[TimelineEntry]) -> str:
        """Format entries as Markdown."""
        lines = ["# Security Timeline\n"]
        lines.append(f"**Events:** {len(entries)}\n")
        lines.append(self._build_date_range_header(entries))
        lines.append("\n---\n")

        for entry in entries:
            lines.append(self._format_markdown_entry(entry))

        return "\n".join(lines)

    def _build_date_range_header(self, entries: List[TimelineEntry]) -> str:
        """Build date range header for timeline."""
        if not entries:
            return ""
        start = entries[0].timestamp.strftime("%Y-%m-%d %H:%M:%S")
        end = entries[-1].timestamp.strftime("%Y-%m-%d %H:%M:%S")
        return f"**Range:** {start} to {end}"

    def _format_markdown_entry(self, entry: TimelineEntry) -> str:
        """Format a single entry as Markdown."""
        ts = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        severity_icon = self._get_severity_icon(entry.severity)

        lines = [f"\n## {severity_icon} {ts} - {entry.event_type}\n"]
        lines.append(f"**Source:** {entry.source}  ")
        lines.append(f"**Severity:** {entry.severity}\n")
        lines.append(f"\n{entry.description}\n")

        if entry.actors:
            lines.append(f"\n**Actors:** {', '.join(entry.actors)}")
        if entry.attack_techniques:
            technique_links = self._format_technique_links(entry.attack_techniques)
            lines.append(f"\n**ATT&CK:** {technique_links}")
        if entry.chunk_ids:
            chunk_links = ", ".join(f"[{cid}](#{cid})" for cid in entry.chunk_ids)
            lines.append(f"\n**Evidence:** {chunk_links}")

        return "\n".join(lines)

    def _get_severity_icon(self, severity: str) -> str:
        """Get icon for severity level."""
        icons = {
            "critical": "[!]",
            "high": "[H]",
            "medium": "[M]",
            "low": "[L]",
            "info": "[i]",
        }
        return icons.get(severity.lower(), "[-]")

    def _format_technique_links(self, techniques: List[str]) -> str:
        """Format ATT&CK technique IDs as links."""
        links = []
        for tid in techniques:
            url = f"https://attack.mitre.org/techniques/{tid}/"
            links.append(f"[{tid}]({url})")
        return ", ".join(links)

    def _format_ascii_table(self, entries: List[TimelineEntry]) -> str:
        """Format entries as ASCII table."""
        col_widths = self._calculate_column_widths(entries)
        header = self._build_table_header(col_widths)
        rows = [self._build_table_row(e, col_widths) for e in entries]

        return "\n".join([header, *rows])

    def _calculate_column_widths(
        self, entries: List[TimelineEntry]
    ) -> Tuple[int, int, int, int]:
        """Calculate column widths for ASCII table."""
        ts_width = 19  # Fixed timestamp width
        type_width = max(len(e.event_type) for e in entries)
        type_width = min(max(type_width, 10), 25)
        src_width = max(len(e.source) for e in entries)
        src_width = min(max(src_width, 8), 15)
        sev_width = 8

        return ts_width, type_width, src_width, sev_width

    def _build_table_header(self, widths: Tuple[int, int, int, int]) -> str:
        """Build ASCII table header."""
        ts_w, type_w, src_w, sev_w = widths
        header = f"{'Timestamp':<{ts_w}} | {'Event Type':<{type_w}} | "
        header += f"{'Source':<{src_w}} | {'Severity':<{sev_w}}"
        separator = "-" * len(header)
        return f"{header}\n{separator}"

    def _build_table_row(
        self, entry: TimelineEntry, widths: Tuple[int, int, int, int]
    ) -> str:
        """Build a single table row."""
        ts_w, type_w, src_w, sev_w = widths
        ts = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        event_type = entry.event_type[:type_w]
        source = entry.source[:src_w]

        return f"{ts:<{ts_w}} | {event_type:<{type_w}} | {source:<{src_w}} | {entry.severity:<{sev_w}}"

    def _build_summary(self, entries: List[TimelineEntry]) -> Dict[str, Any]:
        """Build summary statistics for JSON output."""
        if not entries:
            return {"total_events": 0}

        sources = set(e.source for e in entries)
        severities: Dict[str, int] = {}
        for entry in entries:
            severities[entry.severity] = severities.get(entry.severity, 0) + 1

        techniques = set()
        for entry in entries:
            techniques.update(entry.attack_techniques)

        return {
            "total_events": len(entries),
            "sources": list(sources),
            "severity_breakdown": severities,
            "unique_techniques": list(techniques),
            "start_time": entries[0].timestamp.isoformat(),
            "end_time": entries[-1].timestamp.isoformat(),
        }

    # -------------------------------------------------------------------------
    # Private Helper Methods - Correlation
    # -------------------------------------------------------------------------

    def _correlate_by_actor(
        self, entries: List[TimelineEntry]
    ) -> List[CorrelationGroup]:
        """Correlate events by shared actors."""
        actor_map: Dict[str, List[TimelineEntry]] = {}
        for entry in entries:
            for actor in entry.actors:
                actor_map.setdefault(actor, []).append(entry)

        groups = []
        for actor, actor_entries in actor_map.items():
            if len(actor_entries) > 1:
                groups.append(
                    CorrelationGroup(
                        entries=actor_entries,
                        correlation_type="actor",
                        confidence=min(0.9, 0.5 + (len(actor_entries) * 0.1)),
                    )
                )
        return groups

    def _correlate_by_timewindow(
        self, entries: List[TimelineEntry]
    ) -> List[CorrelationGroup]:
        """Correlate events occurring within time window."""
        if len(entries) < 2:
            return []

        groups = []
        i = 0
        while i < len(entries):
            window_entries = [entries[i]]
            j = i + 1
            while j < len(entries):
                delta = (entries[j].timestamp - entries[i].timestamp).total_seconds()
                if delta <= self._correlation_window:
                    window_entries.append(entries[j])
                    j += 1
                else:
                    break

            if len(window_entries) > 1:
                groups.append(
                    CorrelationGroup(
                        entries=window_entries,
                        correlation_type="timewindow",
                        confidence=min(0.8, 0.4 + (len(window_entries) * 0.1)),
                    )
                )
            i = j if j > i + 1 else i + 1

        return groups

    def _correlate_by_technique(
        self, entries: List[TimelineEntry]
    ) -> List[CorrelationGroup]:
        """Correlate events by shared ATT&CK techniques."""
        technique_map: Dict[str, List[TimelineEntry]] = {}
        for entry in entries:
            for technique in entry.attack_techniques:
                technique_map.setdefault(technique, []).append(entry)

        groups = []
        for technique, tech_entries in technique_map.items():
            if len(tech_entries) > 1:
                groups.append(
                    CorrelationGroup(
                        entries=tech_entries,
                        correlation_type="technique",
                        confidence=min(0.95, 0.6 + (len(tech_entries) * 0.1)),
                    )
                )
        return groups

    def _deduplicate_groups(
        self, groups: List[CorrelationGroup]
    ) -> List[CorrelationGroup]:
        """Remove duplicate correlation groups."""
        seen = set()
        unique = []
        for group in groups:
            key = tuple(sorted(id(e) for e in group.entries))
            if key not in seen:
                seen.add(key)
                unique.append(group)
        return unique


# =============================================================================
# Convenience Functions
# =============================================================================


def build_timeline_from_logs(
    logs: List[FlattenedLog],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> List[TimelineEntry]:
    """Build a timeline from a list of FlattenedLog records.

    Args:
        logs: List of FlattenedLog records
        start: Optional start of date range
        end: Optional end of date range

    Returns:
        List of TimelineEntry sorted chronologically
    """
    builder = TimelineBuilder()
    for log in logs:
        builder.add_event(log)
    return builder.build(start=start, end=end)


def build_timeline_from_chunks(
    chunks: List[Dict[str, Any]],
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> List[TimelineEntry]:
    """Build a timeline from a list of chunk dictionaries.

    Args:
        chunks: List of chunk dictionaries
        start: Optional start of date range
        end: Optional end of date range

    Returns:
        List of TimelineEntry sorted chronologically
    """
    builder = TimelineBuilder()
    for chunk in chunks:
        builder.add_chunk(chunk)
    return builder.build(start=start, end=end)
