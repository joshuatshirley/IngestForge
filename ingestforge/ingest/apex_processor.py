"""
Apex code processor for Salesforce ingestion.

Extracts classes, methods, SOQL queries, and ApexDoc from .cls and .trigger files.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any

from ingestforge.core.provenance import SourceLocation, SourceType
from ingestforge.shared.patterns.processor import IProcessor, ExtractedContent


@dataclass
class ApexMethod:
    """Represents an Apex method."""

    name: str
    signature: str
    access_modifier: str  # public, private, protected, global
    return_type: str
    parameters: List[str]
    docstring: Optional[str] = None
    start_line: int = 0
    end_line: int = 0
    is_static: bool = False
    is_test: bool = False


@dataclass
class ApexClass:
    """Represents an Apex class or trigger."""

    name: str
    class_type: str  # Selector, Service, Handler, Controller, Test, Domain, Trigger, Other
    access_modifier: str  # public, private, global
    sharing_mode: str  # with sharing, without sharing, inherited sharing
    methods: List[ApexMethod] = field(default_factory=list)
    soql_queries: List[str] = field(default_factory=list)
    trigger_events: Optional[
        List[str]
    ] = None  # For triggers: before insert, after update, etc.
    trigger_object: Optional[str] = None  # For triggers: the SObject name
    apexdoc: Dict[str, str] = field(default_factory=dict)  # @description, @group, etc.
    package: str = ""  # Derived from file path (e.g., aie-base-code)
    extends: Optional[str] = None
    implements: List[str] = field(default_factory=list)
    inner_classes: List["ApexClass"] = field(default_factory=list)
    file_path: str = ""
    start_line: int = 1
    end_line: int = 0


class ApexProcessor(IProcessor):
    """
    Process Apex source files (.cls, .trigger).

    Extracts:
    - Class/trigger structure and metadata
    - ApexDoc comments (@description, @param, @return, @group)
    - SOQL queries for searchability
    - Method signatures and access modifiers
    - Trigger events and dispatch patterns

    Example:
        processor = ApexProcessor()
        result = processor.process(Path("AccountsSelector.cls"))
        print(result.metadata["class_type"])  # "Selector"
        print(result.metadata["soql_queries"])  # ["SELECT Id, Name FROM Account..."]
    """

    # Patterns to identify class types based on naming conventions
    CLASS_TYPE_PATTERNS = {
        "Selector": re.compile(r"^[A-Z]\w+Selector$"),
        "Service": re.compile(r"^[A-Z]\w+Service$"),
        "Handler": re.compile(r"^[A-Z]\w+(?:Trigger)?Handler$"),
        "Controller": re.compile(r"^[A-Z]\w+Controller$"),
        "Test": re.compile(r"^[A-Z]\w+Test$|^Test[A-Z]\w+$"),
        "Domain": re.compile(
            r"^[A-Z]\w+Domain$|^[A-Z]\w+s$"
        ),  # e.g., Accounts, Opportunities
        "Batch": re.compile(r"^[A-Z]\w+Batch$"),
        "Queueable": re.compile(r"^[A-Z]\w+Queueable$"),
        "Schedulable": re.compile(r"^[A-Z]\w+Schedule[dr]?$"),
        "DataFactory": re.compile(r"^[A-Z]\w+(?:TDF|TestDataFactory|DataFactory)$"),
        "Invocable": re.compile(r"^[A-Z]\w+Inv(?:ocable)?$"),
        "Wrapper": re.compile(r"^[A-Z]\w+Wrapper$"),
    }

    # Regex patterns for parsing
    APEXDOC_PATTERN = re.compile(r"/\*\*\s*(.*?)\s*\*/", re.DOTALL)
    APEXDOC_TAG_PATTERN = re.compile(r"@(\w+)\s+(.+?)(?=@\w+|$)", re.DOTALL)
    CLASS_PATTERN = re.compile(
        r"(?P<access>public|private|global)?\s*"
        r"(?P<virtual>virtual\s+)?"
        r"(?P<abstract>abstract\s+)?"
        r"(?P<sharing>with\s+sharing|without\s+sharing|inherited\s+sharing)?\s*"
        r"class\s+(?P<name>\w+)"
        r"(?:\s+extends\s+(?P<extends>\w+))?"
        r"(?:\s+implements\s+(?P<implements>[\w,\s]+))?",
        re.IGNORECASE,
    )
    TRIGGER_PATTERN = re.compile(
        r"trigger\s+(?P<name>\w+)\s+on\s+(?P<object>\w+)\s*\(" r"(?P<events>[^)]+)\)",
        re.IGNORECASE,
    )
    METHOD_PATTERN = re.compile(
        r"(?P<annotations>(?:@\w+(?:\([^)]*\))?\s*)*)"
        r"(?P<access>public|private|protected|global)?\s*"
        r"(?P<static>static\s+)?"
        r"(?P<override>override\s+)?"
        r"(?P<virtual>virtual\s+)?"
        r"(?P<return>\w+(?:<[\w,\s<>]+>)?)\s+"
        r"(?P<name>\w+)\s*\("
        r"(?P<params>[^)]*)\)",
        re.IGNORECASE,
    )
    SOQL_PATTERN = re.compile(
        r"\[\s*SELECT\s+.+?\s+FROM\s+\w+.*?\]", re.IGNORECASE | re.DOTALL
    )
    DISPATCH_PATTERN = re.compile(
        r"TriggerDispatcher\.dispatch\s*\(\s*(\w+)\.class\s*\)", re.IGNORECASE
    )

    def __init__(self, sfdx_parser: Any = None) -> None:
        """
        Initialize Apex processor.

        Args:
            sfdx_parser: Optional SFDXProjectParser for package resolution
        """
        self.sfdx_parser = sfdx_parser

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the file."""
        return file_path.suffix.lower() in [".cls", ".trigger"]

    def get_supported_extensions(self) -> List[str]:
        """Return supported file extensions."""
        return [".cls", ".trigger"]

    def process(self, file_path: Path) -> ExtractedContent:
        """
        Process an Apex source file.

        Args:
            file_path: Path to .cls or .trigger file

        Returns:
            ExtractedContent with parsed Apex data
        """
        content = self._read_file(file_path)
        lines = content.split("\n")

        # Determine if it's a trigger or class
        is_trigger = file_path.suffix.lower() == ".trigger"

        if is_trigger:
            apex_class = self._parse_trigger(content, lines, file_path)
        else:
            apex_class = self._parse_class(content, lines, file_path)

        # Resolve package from path
        apex_class.package = self._resolve_package(file_path)

        # Build text content for indexing
        text_content = self._build_text_content(apex_class, content)

        # Build source location
        source_location = SourceLocation(
            source_type=SourceType.CODE,
            title=apex_class.name,
            file_path=str(file_path),
        )

        return ExtractedContent(
            text=text_content,
            metadata={
                "name": apex_class.name,
                "class_type": apex_class.class_type,
                "access_modifier": apex_class.access_modifier,
                "sharing_mode": apex_class.sharing_mode,
                "package": apex_class.package,
                "methods": [self._method_to_dict(m) for m in apex_class.methods],
                "soql_queries": apex_class.soql_queries,
                "trigger_events": apex_class.trigger_events,
                "trigger_object": apex_class.trigger_object,
                "apexdoc": apex_class.apexdoc,
                "extends": apex_class.extends,
                "implements": apex_class.implements,
                "method_count": len(apex_class.methods),
                "is_trigger": is_trigger,
                "file_path": str(file_path),
                "line_count": len(lines),
            },
            sections=[m.name for m in apex_class.methods],
        )

    def _read_file(self, file_path: Path) -> str:
        """Read Apex file with encoding detection."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        return file_path.read_bytes().decode("utf-8", errors="ignore")

    def _parse_class(
        self, content: str, lines: List[str], file_path: Path
    ) -> ApexClass:
        """Parse an Apex class file."""
        # Extract class-level ApexDoc
        apexdoc = self._extract_apexdoc(content, before_pattern=r"\bclass\b")

        # Parse class declaration
        class_match = self.CLASS_PATTERN.search(content)
        if not class_match:
            # Fallback: use filename as class name
            name = file_path.stem
            access = "public"
            sharing = ""
            extends = None
            implements = []
        else:
            name = class_match.group("name")
            access = class_match.group("access") or "public"
            sharing = class_match.group("sharing") or ""
            extends = class_match.group("extends")
            implements_str = class_match.group("implements")
            implements = (
                [i.strip() for i in implements_str.split(",")] if implements_str else []
            )

        # Determine class type
        class_type = self._determine_class_type(name, apexdoc)

        # Extract methods
        methods = self._extract_methods(content, lines)

        # Extract SOQL queries
        soql_queries = self._extract_soql_queries(content)

        return ApexClass(
            name=name,
            class_type=class_type,
            access_modifier=access,
            sharing_mode=sharing,
            methods=methods,
            soql_queries=soql_queries,
            apexdoc=apexdoc,
            extends=extends,
            implements=implements,
            file_path=str(file_path),
            end_line=len(lines),
        )

    def _parse_trigger(
        self, content: str, lines: List[str], file_path: Path
    ) -> ApexClass:
        """Parse an Apex trigger file."""
        trigger_match = self.TRIGGER_PATTERN.search(content)

        if trigger_match:
            name = trigger_match.group("name")
            trigger_object = trigger_match.group("object")
            events_str = trigger_match.group("events")
            trigger_events = [e.strip() for e in events_str.split(",")]
        else:
            name = file_path.stem
            trigger_object = None
            trigger_events = []

        # Check for TriggerDispatcher pattern
        dispatch_match = self.DISPATCH_PATTERN.search(content)
        handler_class = dispatch_match.group(1) if dispatch_match else None

        # Extract class-level ApexDoc
        apexdoc = self._extract_apexdoc(content, before_pattern=r"\btrigger\b")
        if handler_class:
            apexdoc["handler"] = handler_class

        return ApexClass(
            name=name,
            class_type="Trigger",
            access_modifier="trigger",
            sharing_mode="",
            trigger_events=trigger_events,
            trigger_object=trigger_object,
            apexdoc=apexdoc,
            file_path=str(file_path),
            end_line=len(lines),
        )

    def _extract_apexdoc(
        self, content: str, before_pattern: str = None
    ) -> Dict[str, str]:
        """Extract ApexDoc comments."""
        result: dict[str, Any] = {}

        # Find the ApexDoc block before the target pattern
        if before_pattern:
            pattern_match = re.search(before_pattern, content, re.IGNORECASE)
            if pattern_match:
                # Look for /** ... */ before this position
                search_area = content[: pattern_match.start()]
                apexdoc_matches = list(self.APEXDOC_PATTERN.finditer(search_area))
                if apexdoc_matches:
                    doc_content = apexdoc_matches[-1].group(1)
                else:
                    return result
            else:
                return result
        else:
            # Get first ApexDoc in file
            match = self.APEXDOC_PATTERN.search(content)
            if not match:
                return result
            doc_content = match.group(1)

        # Clean up the doc content
        doc_content = re.sub(r"^\s*\*\s?", "", doc_content, flags=re.MULTILINE)

        # Extract tags
        for tag_match in self.APEXDOC_TAG_PATTERN.finditer(doc_content):
            tag_name = tag_match.group(1).lower()
            tag_value = tag_match.group(2).strip()
            tag_value = re.sub(r"\s+", " ", tag_value)  # Normalize whitespace
            result[tag_name] = tag_value

        # If no @description but there's content before first tag, use that
        if "description" not in result:
            first_tag = re.search(r"@\w+", doc_content)
            if first_tag:
                desc = doc_content[: first_tag.start()].strip()
            else:
                desc = doc_content.strip()
            if desc:
                result["description"] = re.sub(r"\s+", " ", desc)

        return result

    def _extract_methods(self, content: str, lines: List[str]) -> List[ApexMethod]:
        """Extract all methods from the class."""
        methods = []

        for match in self.METHOD_PATTERN.finditer(content):
            if self._is_keyword_not_method(match):
                continue

            method = self._create_method_from_match(match, content)
            self._attach_apexdoc_to_method(method, match.start(), content)
            methods.append(method)

        return methods

    def _is_keyword_not_method(self, match: Any) -> bool:
        """Check if match is a keyword, not a method name."""
        return match.group("name").lower() in ["if", "for", "while", "switch", "catch"]

    def _create_method_from_match(self, match: Any, content: str) -> ApexMethod:
        """Create ApexMethod object from regex match."""
        annotations = match.group("annotations") or ""
        is_test = "@isTest" in annotations or "@testMethod" in annotations

        return ApexMethod(
            name=match.group("name"),
            signature=self._build_signature(match),
            access_modifier=match.group("access") or "private",
            return_type=match.group("return"),
            parameters=self._parse_parameters(match.group("params")),
            is_static=bool(match.group("static")),
            is_test=is_test,
            start_line=content[: match.start()].count("\n") + 1,
        )

    def _attach_apexdoc_to_method(
        self, method: ApexMethod, method_pos: int, content: str
    ) -> None:
        """Extract and attach ApexDoc to method if found."""
        preceding = content[:method_pos]
        doc_matches = list(self.APEXDOC_PATTERN.finditer(preceding))

        if not doc_matches:
            return

        last_doc = doc_matches[-1]
        gap = preceding[last_doc.end() :].count("\n")

        if gap <= 3:
            doc_text = self._extract_apexdoc_description(last_doc.group(1))
            if doc_text:
                method.docstring = doc_text

    def _extract_apexdoc_description(self, doc_text: str) -> str:
        """Extract description from ApexDoc text."""
        doc_text = re.sub(r"^\s*\*\s?", "", doc_text, flags=re.MULTILINE)

        # Look for @description tag
        desc_match = re.search(r"@description\s+(.+?)(?=@\w+|$)", doc_text, re.DOTALL)
        if desc_match:
            return re.sub(r"\s+", " ", desc_match.group(1).strip())

        # Extract text before first tag
        if not doc_text.startswith("@"):
            first_tag = re.search(r"@\w+", doc_text)
            desc = doc_text[: first_tag.start()] if first_tag else doc_text
            return re.sub(r"\s+", " ", desc.strip())

        return ""

    def _build_signature(self, match: Any) -> str:
        """Build method signature string."""
        access = match.group("access") or "private"
        static = "static " if match.group("static") else ""
        return_type = match.group("return")
        name = match.group("name")
        params = match.group("params")
        return f"{access} {static}{return_type} {name}({params})"

    def _parse_parameters(self, params_str: str) -> List[str]:
        """Parse parameter list."""
        if not params_str or not params_str.strip():
            return []
        return [p.strip() for p in params_str.split(",") if p.strip()]

    def _extract_soql_queries(self, content: str) -> List[str]:
        """Extract all SOQL queries from the content."""
        queries = []
        for match in self.SOQL_PATTERN.finditer(content):
            query = match.group(0)
            # Normalize whitespace
            query = re.sub(r"\s+", " ", query)
            queries.append(query)
        return queries

    def _determine_class_type(self, name: str, apexdoc: Dict[str, str]) -> str:
        """Determine the class type based on naming convention and ApexDoc."""
        # Check naming patterns
        for class_type, pattern in self.CLASS_TYPE_PATTERNS.items():
            if pattern.match(name):
                return class_type

        # Check ApexDoc @group
        group = apexdoc.get("group", "").lower()
        if "selector" in group:
            return "Selector"
        if "service" in group:
            return "Service"
        if "handler" in group or "trigger" in group:
            return "Handler"
        if "test" in group:
            return "Test"

        return "Other"

    def _resolve_package(self, file_path: Path) -> str:
        """Resolve the package name from file path."""
        # If we have an SFDX parser, use it
        if self.sfdx_parser:
            pkg = self.sfdx_parser.get_package_for_file(file_path)
            if pkg:
                return pkg

        # Fallback: parse from path
        # Look for patterns like /src/aie-base-code/ or /src/package-name/
        path_str = str(file_path)
        match = re.search(r"/src/([a-zA-Z0-9_-]+)/", path_str.replace("\\", "/"))
        if match:
            return match.group(1)

        return ""

    def _build_text_content(self, apex_class: ApexClass, raw_content: str) -> str:
        """Build searchable text content from the Apex class."""
        parts = []

        # Class/trigger name and type
        if apex_class.class_type == "Trigger":
            parts.append(f"Trigger: {apex_class.name}")
            if apex_class.trigger_object:
                parts.append(f"Object: {apex_class.trigger_object}")
            if apex_class.trigger_events:
                parts.append(f"Events: {', '.join(apex_class.trigger_events)}")
        else:
            parts.append(f"Class: {apex_class.name}")
            parts.append(f"Type: {apex_class.class_type}")

        if apex_class.package:
            parts.append(f"Package: {apex_class.package}")

        # ApexDoc description
        if "description" in apex_class.apexdoc:
            parts.append(f"\nDescription: {apex_class.apexdoc['description']}")

        # Methods
        if apex_class.methods:
            parts.append("\nMethods:")
            for method in apex_class.methods:
                method_desc = f"  - {method.name}({', '.join(method.parameters)}): {method.return_type}"
                if method.docstring:
                    method_desc += f" - {method.docstring}"
                parts.append(method_desc)

        # SOQL queries (useful for search)
        if apex_class.soql_queries:
            parts.append("\nSOQL Queries:")
            for query in apex_class.soql_queries[:5]:  # Limit to first 5
                parts.append(f"  {query[:200]}")

        # Include the full raw content for complete indexing
        parts.append("\n\n--- Source Code ---\n")
        parts.append(raw_content)

        return "\n".join(parts)

    def _method_to_dict(self, method: ApexMethod) -> Dict[str, Any]:
        """Convert ApexMethod to dictionary."""
        return {
            "name": method.name,
            "signature": method.signature,
            "access_modifier": method.access_modifier,
            "return_type": method.return_type,
            "parameters": method.parameters,
            "docstring": method.docstring,
            "is_static": method.is_static,
            "is_test": method.is_test,
            "start_line": method.start_line,
        }
