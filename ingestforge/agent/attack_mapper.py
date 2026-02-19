"""ATT&CK Mapping Agent for Cyber Vertical.

Maps log chunks to MITRE ATT&CK techniques using multi-agent prompts.
Part of CYBER-003 implementation.

Architecture Context
--------------------
ATTACKMapper integrates with the enrichment pipeline to add attack_technique
metadata to chunks during security log analysis:

    LogFlattener -> ATTACKMapper -> Chunk Storage
                         |
                    LLM Client (optional)

The mapper can operate in two modes:
1. Rule-based: Fast pattern matching using built-in technique database
2. LLM-enhanced: Multi-agent prompts for nuanced technique identification

Usage Example
-------------
    from ingestforge.agent.attack_mapper import ATTACKMapper, create_mapper

    # Create mapper with LLM for enhanced detection
    mapper = create_mapper(llm_client=my_llm)

    # Map a single chunk
    mappings = mapper.map_chunk(chunk)

    # Enrich multiple chunks
    enriched = mapper.enrich_chunks(chunks)

    # Get technique info
    info = mapper.get_technique_info("T1059.001")"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from ingestforge.core.logging import get_logger
from ingestforge.enrichment.log_flattener import FlattenedLog

logger = get_logger(__name__)
MAX_MAPPINGS_PER_CHUNK = 10
MAX_INDICATORS_PER_MAPPING = 20
MAX_EVIDENCE_LENGTH = 500
MAX_CHUNKS_BATCH = 100
MAX_PROMPT_LENGTH = 8000

# =============================================================================
# Data Types
# =============================================================================


class Tactic(Enum):
    """MITRE ATT&CK Tactics (Enterprise)."""

    RECONNAISSANCE = "reconnaissance"
    RESOURCE_DEVELOPMENT = "resource-development"
    INITIAL_ACCESS = "initial-access"
    EXECUTION = "execution"
    PERSISTENCE = "persistence"
    PRIVILEGE_ESCALATION = "privilege-escalation"
    DEFENSE_EVASION = "defense-evasion"
    CREDENTIAL_ACCESS = "credential-access"
    DISCOVERY = "discovery"
    LATERAL_MOVEMENT = "lateral-movement"
    COLLECTION = "collection"
    COMMAND_AND_CONTROL = "command-and-control"
    EXFILTRATION = "exfiltration"
    IMPACT = "impact"


@dataclass
class ATTACKMapping:
    """Mapping of a chunk to a MITRE ATT&CK technique.

    Attributes:
        chunk_id: Unique identifier for the mapped chunk
        technique_id: ATT&CK technique ID (e.g., "T1059.001")
        technique_name: Human-readable name (e.g., "PowerShell")
        tactic: Associated tactic category
        confidence: Mapping confidence score (0.0-1.0)
        evidence: Text supporting the mapping
        indicators: Specific IOCs if present
    """

    chunk_id: str
    technique_id: str
    technique_name: str
    tactic: str
    confidence: float
    evidence: str = ""
    indicators: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate and constrain fields."""
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.evidence = self.evidence[:MAX_EVIDENCE_LENGTH]
        self.indicators = self.indicators[:MAX_INDICATORS_PER_MAPPING]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "technique_id": self.technique_id,
            "technique_name": self.technique_name,
            "tactic": self.tactic,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "indicators": self.indicators,
        }


@dataclass
class TechniqueInfo:
    """Information about an ATT&CK technique.

    Attributes:
        technique_id: ATT&CK technique ID
        name: Technique name
        tactic: Primary tactic
        description: Brief description
        detection_hints: Patterns to look for
        keywords: Keywords for rule-based detection
    """

    technique_id: str
    name: str
    tactic: Tactic
    description: str = ""
    detection_hints: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "technique_id": self.technique_id,
            "name": self.name,
            "tactic": self.tactic.value,
            "description": self.description,
            "detection_hints": self.detection_hints,
            "keywords": self.keywords,
        }


# =============================================================================
# Built-in Technique Database
# =============================================================================

# Top 50 most common ATT&CK techniques with detection patterns
TECHNIQUE_DATABASE: List[TechniqueInfo] = [
    # Execution
    TechniqueInfo(
        "T1059.001",
        "PowerShell",
        Tactic.EXECUTION,
        "Adversaries may abuse PowerShell for execution",
        ["powershell.exe", "pwsh.exe", "-encodedcommand", "-enc"],
        ["powershell", "invoke-expression", "iex", "downloadstring"],
    ),
    TechniqueInfo(
        "T1059.003",
        "Windows Command Shell",
        Tactic.EXECUTION,
        "Adversaries may abuse cmd.exe for execution",
        ["cmd.exe /c", "cmd /k", "command prompt"],
        ["cmd", "cmd.exe", "batch", "command shell"],
    ),
    TechniqueInfo(
        "T1059.005",
        "Visual Basic",
        Tactic.EXECUTION,
        "Adversaries may abuse VB for execution",
        ["wscript.exe", "cscript.exe", ".vbs", ".vbe"],
        ["vbscript", "wscript", "cscript", "visual basic"],
    ),
    TechniqueInfo(
        "T1059.007",
        "JavaScript",
        Tactic.EXECUTION,
        "Adversaries may abuse JavaScript for execution",
        ["wscript.exe", "cscript.exe", ".js", ".jse"],
        ["javascript", "jscript", ".js", "node.exe"],
    ),
    TechniqueInfo(
        "T1204.001",
        "Malicious Link",
        Tactic.EXECUTION,
        "User clicks malicious link",
        ["http://", "https://", "clicked link"],
        ["click", "link", "url", "redirect"],
    ),
    TechniqueInfo(
        "T1204.002",
        "Malicious File",
        Tactic.EXECUTION,
        "User opens malicious file",
        ["opened file", "executed attachment"],
        ["attachment", "document", "executable", "macro"],
    ),
    # Initial Access
    TechniqueInfo(
        "T1566.001",
        "Spearphishing Attachment",
        Tactic.INITIAL_ACCESS,
        "Adversaries send malicious attachments via email",
        ["email attachment", "received attachment", "outlook"],
        ["phishing", "attachment", "email", "spearphish"],
    ),
    TechniqueInfo(
        "T1566.002",
        "Spearphishing Link",
        Tactic.INITIAL_ACCESS,
        "Adversaries send malicious links via email",
        ["email link", "clicked link in email"],
        ["phishing", "link", "email", "url"],
    ),
    TechniqueInfo(
        "T1190",
        "Exploit Public-Facing Application",
        Tactic.INITIAL_ACCESS,
        "Adversaries exploit internet-facing services",
        ["exploit", "vulnerability", "web application"],
        ["exploit", "cve", "vulnerability", "rce", "injection"],
    ),
    TechniqueInfo(
        "T1133",
        "External Remote Services",
        Tactic.INITIAL_ACCESS,
        "Adversaries use remote services to access systems",
        ["vpn", "rdp", "ssh", "remote desktop"],
        ["remote", "vpn", "rdp", "ssh", "citrix"],
    ),
    TechniqueInfo(
        "T1078",
        "Valid Accounts",
        Tactic.INITIAL_ACCESS,
        "Adversaries use stolen credentials",
        ["successful login", "logon", "authentication"],
        ["login", "logon", "account", "credential", "authentication"],
    ),
    # Persistence
    TechniqueInfo(
        "T1547.001",
        "Registry Run Keys",
        Tactic.PERSISTENCE,
        "Adversaries add registry run keys for persistence",
        ["HKLM\\Software\\Microsoft\\Windows\\CurrentVersion\\Run"],
        ["registry", "run key", "autorun", "startup"],
    ),
    TechniqueInfo(
        "T1053.005",
        "Scheduled Task",
        Tactic.PERSISTENCE,
        "Adversaries create scheduled tasks for persistence",
        ["schtasks.exe", "at.exe", "taskschd.msc"],
        ["scheduled task", "schtasks", "task scheduler"],
    ),
    TechniqueInfo(
        "T1543.003",
        "Windows Service",
        Tactic.PERSISTENCE,
        "Adversaries create services for persistence",
        ["sc.exe create", "New-Service", "service installation"],
        ["service", "sc.exe", "service create", "daemon"],
    ),
    TechniqueInfo(
        "T1136.001",
        "Local Account",
        Tactic.PERSISTENCE,
        "Adversaries create local accounts",
        ["net user /add", "New-LocalUser", "user creation"],
        ["user add", "account create", "new user", "net user"],
    ),
    # Privilege Escalation
    TechniqueInfo(
        "T1548.002",
        "Bypass User Account Control",
        Tactic.PRIVILEGE_ESCALATION,
        "Adversaries bypass UAC mechanisms",
        ["uac bypass", "elevated privileges"],
        ["uac", "bypass", "elevation", "admin"],
    ),
    TechniqueInfo(
        "T1068",
        "Exploitation for Privilege Escalation",
        Tactic.PRIVILEGE_ESCALATION,
        "Adversaries exploit vulnerabilities to escalate privileges",
        ["privilege escalation", "local exploit"],
        ["exploit", "escalation", "privilege", "root", "admin"],
    ),
    # Defense Evasion
    TechniqueInfo(
        "T1070.001",
        "Clear Windows Event Logs",
        Tactic.DEFENSE_EVASION,
        "Adversaries clear event logs",
        ["wevtutil cl", "Clear-EventLog"],
        ["clear log", "event log", "wevtutil", "audit log"],
    ),
    TechniqueInfo(
        "T1562.001",
        "Disable or Modify Tools",
        Tactic.DEFENSE_EVASION,
        "Adversaries disable security tools",
        ["disable defender", "stop antivirus"],
        ["disable", "antivirus", "defender", "security"],
    ),
    TechniqueInfo(
        "T1027",
        "Obfuscated Files or Information",
        Tactic.DEFENSE_EVASION,
        "Adversaries obfuscate files or data",
        ["base64", "encoded", "obfuscated"],
        ["obfuscate", "encode", "base64", "encrypt"],
    ),
    TechniqueInfo(
        "T1036",
        "Masquerading",
        Tactic.DEFENSE_EVASION,
        "Adversaries masquerade files or processes",
        ["renamed executable", "fake process name"],
        ["masquerade", "rename", "impersonate", "fake"],
    ),
    TechniqueInfo(
        "T1218.011",
        "Rundll32",
        Tactic.DEFENSE_EVASION,
        "Adversaries abuse rundll32.exe for execution",
        ["rundll32.exe", "rundll32"],
        ["rundll32", "dll", "proxy execution"],
    ),
    TechniqueInfo(
        "T1218.005",
        "Mshta",
        Tactic.DEFENSE_EVASION,
        "Adversaries abuse mshta.exe",
        ["mshta.exe", "mshta"],
        ["mshta", "hta", "html application"],
    ),
    # Credential Access
    TechniqueInfo(
        "T1003.001",
        "LSASS Memory",
        Tactic.CREDENTIAL_ACCESS,
        "Adversaries dump LSASS memory for credentials",
        ["lsass.exe", "mimikatz", "procdump"],
        ["lsass", "dump", "mimikatz", "credential", "memory"],
    ),
    TechniqueInfo(
        "T1003.002",
        "Security Account Manager",
        Tactic.CREDENTIAL_ACCESS,
        "Adversaries extract credentials from SAM database",
        ["sam database", "registry hive", "hashdump"],
        ["sam", "hash", "ntds", "password"],
    ),
    TechniqueInfo(
        "T1110.001",
        "Password Guessing",
        Tactic.CREDENTIAL_ACCESS,
        "Adversaries attempt to guess passwords",
        ["failed login", "authentication failure", "brute force"],
        ["brute force", "password guess", "failed auth", "lockout"],
    ),
    TechniqueInfo(
        "T1110.003",
        "Password Spraying",
        Tactic.CREDENTIAL_ACCESS,
        "Adversaries try common passwords across accounts",
        ["multiple failed logins", "spray attack"],
        ["spray", "common password", "multiple accounts"],
    ),
    TechniqueInfo(
        "T1555",
        "Credentials from Password Stores",
        Tactic.CREDENTIAL_ACCESS,
        "Adversaries extract credentials from stores",
        ["credential manager", "keychain", "password vault"],
        ["credential store", "keychain", "vault", "password manager"],
    ),
    # Discovery
    TechniqueInfo(
        "T1087.001",
        "Local Account Discovery",
        Tactic.DISCOVERY,
        "Adversaries enumerate local accounts",
        ["net user", "wmic useraccount"],
        ["enumerate", "user list", "account discovery"],
    ),
    TechniqueInfo(
        "T1083",
        "File and Directory Discovery",
        Tactic.DISCOVERY,
        "Adversaries enumerate files and directories",
        ["dir", "ls", "find", "tree"],
        ["directory", "file", "enumerate", "list"],
    ),
    TechniqueInfo(
        "T1057",
        "Process Discovery",
        Tactic.DISCOVERY,
        "Adversaries enumerate running processes",
        ["tasklist", "ps", "Get-Process"],
        ["process", "tasklist", "running", "pid"],
    ),
    TechniqueInfo(
        "T1082",
        "System Information Discovery",
        Tactic.DISCOVERY,
        "Adversaries gather system information",
        ["systeminfo", "hostname", "uname"],
        ["system", "info", "hostname", "os version"],
    ),
    TechniqueInfo(
        "T1016",
        "System Network Configuration Discovery",
        Tactic.DISCOVERY,
        "Adversaries gather network configuration",
        ["ipconfig", "ifconfig", "route print"],
        ["network", "ip", "config", "adapter", "route"],
    ),
    TechniqueInfo(
        "T1018",
        "Remote System Discovery",
        Tactic.DISCOVERY,
        "Adversaries discover remote systems",
        ["net view", "ping sweep", "nmap"],
        ["scan", "ping", "remote", "network discovery"],
    ),
    # Lateral Movement
    TechniqueInfo(
        "T1021.001",
        "Remote Desktop Protocol",
        Tactic.LATERAL_MOVEMENT,
        "Adversaries use RDP for lateral movement",
        ["rdp", "mstsc.exe", "port 3389"],
        ["rdp", "remote desktop", "3389", "mstsc"],
    ),
    TechniqueInfo(
        "T1021.002",
        "SMB/Windows Admin Shares",
        Tactic.LATERAL_MOVEMENT,
        "Adversaries use SMB for lateral movement",
        ["admin$", "c$", "ipc$", "smb"],
        ["smb", "admin share", "psexec", "445"],
    ),
    TechniqueInfo(
        "T1021.004",
        "SSH",
        Tactic.LATERAL_MOVEMENT,
        "Adversaries use SSH for lateral movement",
        ["ssh", "port 22", "openssh"],
        ["ssh", "22", "openssh", "putty"],
    ),
    TechniqueInfo(
        "T1021.006",
        "Windows Remote Management",
        Tactic.LATERAL_MOVEMENT,
        "Adversaries use WinRM for lateral movement",
        ["winrm", "wsman", "port 5985"],
        ["winrm", "wsman", "remote management", "5985"],
    ),
    TechniqueInfo(
        "T1570",
        "Lateral Tool Transfer",
        Tactic.LATERAL_MOVEMENT,
        "Adversaries transfer tools between systems",
        ["copy", "scp", "xcopy", "robocopy"],
        ["transfer", "copy", "tool", "lateral"],
    ),
    # Collection
    TechniqueInfo(
        "T1560.001",
        "Archive via Utility",
        Tactic.COLLECTION,
        "Adversaries archive data using utilities",
        ["7z", "rar", "zip", "tar"],
        ["archive", "compress", "zip", "rar", "7z"],
    ),
    TechniqueInfo(
        "T1005",
        "Data from Local System",
        Tactic.COLLECTION,
        "Adversaries collect data from local systems",
        ["file access", "data theft", "copy sensitive"],
        ["collect", "data", "sensitive", "exfil"],
    ),
    TechniqueInfo(
        "T1114.001",
        "Local Email Collection",
        Tactic.COLLECTION,
        "Adversaries collect email data locally",
        ["pst", "ost", "email export"],
        ["email", "pst", "outlook", "mailbox"],
    ),
    TechniqueInfo(
        "T1113",
        "Screen Capture",
        Tactic.COLLECTION,
        "Adversaries capture screenshots",
        ["screenshot", "screen capture"],
        ["screenshot", "capture", "screen", "display"],
    ),
    # Command and Control
    TechniqueInfo(
        "T1071.001",
        "Web Protocols",
        Tactic.COMMAND_AND_CONTROL,
        "Adversaries use web protocols for C2",
        ["http callback", "https beacon", "web traffic"],
        ["http", "https", "web", "c2", "beacon"],
    ),
    TechniqueInfo(
        "T1071.004",
        "DNS",
        Tactic.COMMAND_AND_CONTROL,
        "Adversaries use DNS for C2",
        ["dns query", "dns tunnel", "txt record"],
        ["dns", "tunnel", "query", "c2"],
    ),
    TechniqueInfo(
        "T1105",
        "Ingress Tool Transfer",
        Tactic.COMMAND_AND_CONTROL,
        "Adversaries transfer tools into environment",
        ["download", "wget", "curl", "certutil"],
        ["download", "transfer", "wget", "curl", "certutil"],
    ),
    TechniqueInfo(
        "T1573",
        "Encrypted Channel",
        Tactic.COMMAND_AND_CONTROL,
        "Adversaries use encryption for C2",
        ["encrypted", "ssl", "tls", "custom protocol"],
        ["encrypt", "ssl", "tls", "secure channel"],
    ),
    # Exfiltration
    TechniqueInfo(
        "T1041",
        "Exfiltration Over C2 Channel",
        Tactic.EXFILTRATION,
        "Adversaries exfiltrate data over C2",
        ["data exfiltration", "upload", "outbound"],
        ["exfiltrate", "upload", "c2", "outbound"],
    ),
    TechniqueInfo(
        "T1048",
        "Exfiltration Over Alternative Protocol",
        Tactic.EXFILTRATION,
        "Adversaries use alternative protocols for exfiltration",
        ["ftp upload", "dns exfil", "icmp tunnel"],
        ["ftp", "dns", "icmp", "exfil", "alternative"],
    ),
    # Impact
    TechniqueInfo(
        "T1486",
        "Data Encrypted for Impact",
        Tactic.IMPACT,
        "Adversaries encrypt data for ransom",
        ["ransomware", "encrypted files", "ransom note"],
        ["ransomware", "encrypt", "ransom", "locked"],
    ),
    TechniqueInfo(
        "T1489",
        "Service Stop",
        Tactic.IMPACT,
        "Adversaries stop services to cause impact",
        ["service stop", "sc stop", "Stop-Service"],
        ["stop", "service", "disable", "kill"],
    ),
    TechniqueInfo(
        "T1490",
        "Inhibit System Recovery",
        Tactic.IMPACT,
        "Adversaries delete backups and shadow copies",
        ["vssadmin delete shadows", "bcdedit", "wbadmin"],
        ["shadow", "backup", "recovery", "vssadmin"],
    ),
]

# Build lookup indices for fast access
_TECHNIQUE_BY_ID: Dict[str, TechniqueInfo] = {
    t.technique_id: t for t in TECHNIQUE_DATABASE
}
_TECHNIQUE_BY_TACTIC: Dict[Tactic, List[TechniqueInfo]] = {}
for _t in TECHNIQUE_DATABASE:
    _TECHNIQUE_BY_TACTIC.setdefault(_t.tactic, []).append(_t)

# =============================================================================
# LLM Function Type
# =============================================================================

# Type for LLM mapping function
MappingFunction = Callable[[str], str]

# =============================================================================
# ATTACKMapper Class
# =============================================================================


class ATTACKMapper:
    """Maps chunks to MITRE ATT&CK techniques.

    Provides both rule-based pattern matching and optional LLM-enhanced
    mapping for nuanced technique identification.

    Examples:
        >>> mapper = ATTACKMapper()
        >>> chunk = {"id": "chunk1", "text": "powershell.exe -enc base64"}
        >>> mappings = mapper.map_chunk(chunk)
        >>> print(mappings[0].technique_id)
        T1059.001
    """

    def __init__(
        self,
        llm_fn: Optional[MappingFunction] = None,
        min_confidence: float = 0.3,
    ) -> None:
        """Initialize the ATTACKMapper.

        Args:
            llm_fn: Optional function for LLM-based mapping
            min_confidence: Minimum confidence threshold for mappings
        """
        if min_confidence < 0.0 or min_confidence > 1.0:
            min_confidence = 0.3

        self._llm_fn = llm_fn
        self._min_confidence = min_confidence
        self._technique_db = TECHNIQUE_DATABASE

    def map_chunk(self, chunk: Dict[str, Any]) -> List[ATTACKMapping]:
        """Map a chunk to ATT&CK techniques.

        Args:
            chunk: Chunk dictionary with 'id' and 'text' fields

        Returns:
            List of ATTACKMapping objects
        """
        chunk_id = str(chunk.get("id", "unknown"))
        text = str(chunk.get("text", ""))

        if not text.strip():
            return []

        # Rule-based mapping first
        mappings = self._rule_based_map(chunk_id, text)

        # LLM-enhanced mapping if available
        if self._llm_fn and len(mappings) < MAX_MAPPINGS_PER_CHUNK:
            llm_mappings = self._llm_map(chunk_id, text)
            mappings = self._merge_mappings(mappings, llm_mappings)

        # Filter by confidence threshold
        filtered = [m for m in mappings if m.confidence >= self._min_confidence]
        return filtered[:MAX_MAPPINGS_PER_CHUNK]

    def map_log_event(self, log: FlattenedLog) -> List[ATTACKMapping]:
        """Map a flattened log event to ATT&CK techniques.

        Args:
            log: FlattenedLog from log_flattener

        Returns:
            List of ATTACKMapping objects
        """
        # Build chunk-like structure from log
        chunk = {
            "id": f"log_{log.timestamp}_{log.event_id}",
            "text": self._build_log_text(log),
        }
        return self.map_chunk(chunk)

    def enrich_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enrich chunks with ATT&CK technique metadata.

        Adds 'attack_techniques' metadata to each chunk.

        Args:
            chunks: List of chunk dictionaries

        Returns:
            Enriched chunks with attack_techniques metadata
        """
        chunks = chunks[:MAX_CHUNKS_BATCH]
        enriched: List[Dict[str, Any]] = []

        for chunk in chunks:
            mappings = self.map_chunk(chunk)
            enriched_chunk = dict(chunk)
            self._add_technique_metadata(enriched_chunk, mappings)
            enriched.append(enriched_chunk)

        return enriched

    def get_technique_info(self, technique_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific technique.

        Args:
            technique_id: ATT&CK technique ID (e.g., "T1059.001")

        Returns:
            Technique info dictionary or None if not found
        """
        info = _TECHNIQUE_BY_ID.get(technique_id)
        return info.to_dict() if info else None

    def get_techniques_by_tactic(self, tactic: str) -> List[Dict[str, Any]]:
        """Get all techniques for a tactic.

        Args:
            tactic: Tactic name (e.g., "execution")

        Returns:
            List of technique info dictionaries
        """
        try:
            tactic_enum = Tactic(tactic.lower().replace("_", "-"))
        except ValueError:
            return []

        techniques = _TECHNIQUE_BY_TACTIC.get(tactic_enum, [])
        return [t.to_dict() for t in techniques]

    # -------------------------------------------------------------------------
    # Private Helper Methods
    # -------------------------------------------------------------------------

    def _rule_based_map(self, chunk_id: str, text: str) -> List[ATTACKMapping]:
        """Perform rule-based technique mapping."""
        text_lower = text.lower()
        mappings: List[ATTACKMapping] = []

        for technique in self._technique_db:
            confidence, evidence = self._match_technique(technique, text_lower)
            if confidence > 0:
                mapping = ATTACKMapping(
                    chunk_id=chunk_id,
                    technique_id=technique.technique_id,
                    technique_name=technique.name,
                    tactic=technique.tactic.value,
                    confidence=confidence,
                    evidence=evidence,
                    indicators=self._extract_indicators(technique, text),
                )
                mappings.append(mapping)

        return sorted(mappings, key=lambda m: m.confidence, reverse=True)

    def _match_technique(
        self, technique: TechniqueInfo, text: str
    ) -> tuple[float, str]:
        """Match a technique against text and return confidence."""
        keyword_matches = sum(1 for kw in technique.keywords if kw in text)
        hint_matches = sum(1 for h in technique.detection_hints if h.lower() in text)

        if keyword_matches == 0 and hint_matches == 0:
            return 0.0, ""

        # Calculate confidence based on matches
        total_patterns = len(technique.keywords) + len(technique.detection_hints)
        match_count = keyword_matches + hint_matches * 2  # Weight hints higher
        confidence = min(1.0, match_count / max(total_patterns, 1))

        # Build evidence string
        evidence = self._build_evidence(technique, text)

        return confidence, evidence

    def _build_evidence(self, technique: TechniqueInfo, text: str) -> str:
        """Build evidence string for a mapping."""
        matches: List[str] = []
        for kw in technique.keywords:
            if kw in text:
                matches.append(f"keyword:{kw}")
        for hint in technique.detection_hints:
            if hint.lower() in text:
                matches.append(f"pattern:{hint}")
        return ", ".join(matches[:5])

    def _extract_indicators(self, technique: TechniqueInfo, text: str) -> List[str]:
        """Extract specific indicators from text."""
        indicators: List[str] = []
        for hint in technique.detection_hints:
            if hint.lower() in text.lower():
                indicators.append(hint)
        return indicators[:MAX_INDICATORS_PER_MAPPING]

    def _llm_map(self, chunk_id: str, text: str) -> List[ATTACKMapping]:
        """Perform LLM-based technique mapping."""
        if not self._llm_fn:
            return []

        prompt = self._build_llm_prompt(text)
        try:
            response = self._llm_fn(prompt)
            return self._parse_llm_response(chunk_id, response)
        except Exception as e:
            logger.warning("LLM mapping failed", error=str(e))
            return []

    def _build_llm_prompt(self, text: str) -> str:
        """Build prompt for LLM-based mapping."""
        technique_list = self._get_technique_summary()
        prompt = f"""Analyze the following security log or event text and identify any MITRE ATT&CK techniques present.

Log/Event Text:
{text[:MAX_PROMPT_LENGTH]}

Available Techniques:
{technique_list}

Respond in JSON format:
[{{"technique_id": "T1059.001", "confidence": 0.8, "evidence": "powershell execution detected"}}]

If no techniques match, respond with an empty array: []"""
        return prompt

    def _get_technique_summary(self) -> str:
        """Get a summary of techniques for the prompt."""
        lines: List[str] = []
        for t in self._technique_db[:30]:  # Limit for prompt size
            lines.append(f"- {t.technique_id}: {t.name} ({t.tactic.value})")
        return "\n".join(lines)

    def _parse_llm_response(self, chunk_id: str, response: str) -> List[ATTACKMapping]:
        """Parse LLM response into ATTACKMapping objects."""
        mappings: List[ATTACKMapping] = []

        # Extract JSON from response
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if not json_match:
            return []

        try:
            items = json.loads(json_match.group())
        except json.JSONDecodeError:
            return []

        for item in items[:MAX_MAPPINGS_PER_CHUNK]:
            mapping = self._item_to_mapping(chunk_id, item)
            if mapping:
                mappings.append(mapping)

        return mappings

    def _item_to_mapping(
        self, chunk_id: str, item: Dict[str, Any]
    ) -> Optional[ATTACKMapping]:
        """Convert a parsed item to ATTACKMapping."""
        technique_id = item.get("technique_id", "")
        technique_info = _TECHNIQUE_BY_ID.get(technique_id)

        if not technique_info:
            return None

        return ATTACKMapping(
            chunk_id=chunk_id,
            technique_id=technique_id,
            technique_name=technique_info.name,
            tactic=technique_info.tactic.value,
            confidence=float(item.get("confidence", 0.5)),
            evidence=str(item.get("evidence", "")),
        )

    def _merge_mappings(
        self,
        rule_mappings: List[ATTACKMapping],
        llm_mappings: List[ATTACKMapping],
    ) -> List[ATTACKMapping]:
        """Merge rule-based and LLM mappings."""
        seen_ids: set[str] = set()
        merged: List[ATTACKMapping] = []

        # Add rule-based first
        for m in rule_mappings:
            seen_ids.add(m.technique_id)
            merged.append(m)

        # Add unique LLM mappings
        for m in llm_mappings:
            if m.technique_id not in seen_ids:
                seen_ids.add(m.technique_id)
                merged.append(m)

        return merged

    def _build_log_text(self, log: FlattenedLog) -> str:
        """Build searchable text from FlattenedLog."""
        parts = [
            f"event_id:{log.event_id}",
            f"category:{log.event_category.value}",
            f"user:{log.user}",
            f"host:{log.host}",
            log.message,
        ]
        return " ".join(p for p in parts if p)

    def _add_technique_metadata(
        self,
        chunk: Dict[str, Any],
        mappings: List[ATTACKMapping],
    ) -> None:
        """Add attack technique metadata to chunk in-place."""
        metadata = chunk.setdefault("metadata", {})

        if mappings:
            # Primary technique
            metadata["attack_technique"] = mappings[0].technique_id
            metadata["attack_tactic"] = mappings[0].tactic
            metadata["attack_confidence"] = mappings[0].confidence

            # All techniques
            metadata["attack_techniques"] = [
                {
                    "id": m.technique_id,
                    "name": m.technique_name,
                    "tactic": m.tactic,
                    "confidence": m.confidence,
                }
                for m in mappings
            ]


# =============================================================================
# Factory Functions
# =============================================================================


def create_mapper(
    llm_fn: Optional[MappingFunction] = None,
    min_confidence: float = 0.3,
) -> ATTACKMapper:
    """Create an ATTACKMapper instance.

    Args:
        llm_fn: Optional LLM function for enhanced mapping
        min_confidence: Minimum confidence threshold

    Returns:
        Configured ATTACKMapper instance
    """
    return ATTACKMapper(llm_fn=llm_fn, min_confidence=min_confidence)


def create_llm_mapping_function(llm_client: Any) -> MappingFunction:
    """Create a mapping function from an LLM client.

    Args:
        llm_client: LLMClient instance

    Returns:
        Function suitable for ATTACKMapper
    """

    def mapping_fn(prompt: str) -> str:
        return llm_client.generate(prompt)

    return mapping_fn


def map_chunk_to_attack(
    chunk: Dict[str, Any],
    min_confidence: float = 0.3,
) -> List[ATTACKMapping]:
    """Convenience function to map a single chunk.

    Args:
        chunk: Chunk dictionary
        min_confidence: Minimum confidence threshold

    Returns:
        List of ATTACKMapping objects
    """
    mapper = ATTACKMapper(min_confidence=min_confidence)
    return mapper.map_chunk(chunk)


def get_technique_database() -> List[Dict[str, Any]]:
    """Get the full technique database.

    Returns:
        List of technique info dictionaries
    """
    return [t.to_dict() for t in TECHNIQUE_DATABASE]


def get_all_tactics() -> List[str]:
    """Get all tactic names.

    Returns:
        List of tactic names
    """
    return [t.value for t in Tactic]
