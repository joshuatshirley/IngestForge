"""
Azure DevOps Work Item processor for IngestForge.

Parses ADO work item markdown exports to extract structured metadata,
acceptance criteria, and relationships for semantic search.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Dict, Any

from ingestforge.core.provenance import SourceLocation, SourceType
from ingestforge.shared.patterns.processor import IProcessor, ExtractedContent


@dataclass
class AcceptanceCriteria:
    """Represents a single acceptance criterion in Given-When-Then format."""

    id: str  # e.g., "AC 01"
    given: List[str]
    when: List[str]
    then: List[str]
    raw_text: str


@dataclass
class ChildWorkItem:
    """Reference to a child work item."""

    id: int
    title: str
    work_type: str  # Task, Bug, etc.
    completed: bool


@dataclass
class ADOWorkItem:
    """Represents a complete ADO work item."""

    id: int
    title: str
    work_item_type: str  # Feature, User Story, Bug, Task
    state: str
    assigned_to: Optional[str]
    iteration_path: str
    area_path: str
    parent_id: Optional[int]
    child_ids: List[int]
    story_points: Optional[float]
    priority: Optional[int]
    description: str
    acceptance_criteria: List[AcceptanceCriteria]
    children: List[ChildWorkItem]
    created_date: Optional[str]
    modified_date: Optional[str]
    file_path: str = ""


class ADOProcessor(IProcessor):
    """
    Process Azure DevOps work item markdown exports.

    Extracts:
    - Work item metadata (ID, Type, State, Iteration, etc.)
    - Description content (with HTML cleanup)
    - Given-When-Then acceptance criteria
    - Parent-child relationships
    - Embedded references (Apex classes, packages, etc.)

    Example:
        processor = ADOProcessor()
        result = processor.process(Path("29232_View AIE Support Case.md"))
        print(result.metadata["work_item_type"])  # "User Story"
        print(result.metadata["acceptance_criteria"])
    """

    # Regex patterns for parsing
    HEADER_PATTERN = re.compile(
        r"^#\s+(?P<type>Feature|User Story|Bug|Task|Epic)\s+#(?P<id>\d+)\s*$",
        re.MULTILINE,
    )
    TITLE_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)

    # Table field patterns
    FIELD_PATTERNS = {
        "id": re.compile(r"\|\s*\*\*ID\*\*\s*\|\s*(\d+)\s*\|"),
        "type": re.compile(r"\|\s*\*\*Type\*\*\s*\|\s*(.+?)\s*\|"),
        "state": re.compile(r"\|\s*\*\*State\*\*\s*\|\s*(.+?)\s*\|"),
        "assigned_to": re.compile(r"\|\s*\*\*Assigned To\*\*\s*\|\s*(.+?)\s*\|"),
        "iteration": re.compile(r"\|\s*\*\*Iteration\*\*\s*\|\s*(.+?)\s*\|"),
        "area": re.compile(r"\|\s*\*\*Area\*\*\s*\|\s*(.+?)\s*\|"),
        "parent": re.compile(r"\|\s*\*\*Parent\*\*\s*\|\s*#(\d+)\s*\|"),
        "story_points": re.compile(r"\|\s*\*\*Story Points\*\*\s*\|\s*([\d.]+)\s*\|"),
        "priority": re.compile(r"\|\s*\*\*Priority\*\*\s*\|\s*(\d+)\s*\|"),
    }

    # Section patterns
    DESCRIPTION_SECTION = re.compile(
        r"##\s*Description\s*\n+(.*?)(?=\n##|\n---|\Z)", re.DOTALL | re.IGNORECASE
    )
    ACCEPTANCE_SECTION = re.compile(
        r"##\s*Acceptance Criteria\s*\n+(.*?)(?=\n##|\n---|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    CHILDREN_SECTION = re.compile(
        r"##\s*Child Work Items\s*\n+(.*?)(?=\n##|\n---|\Z)", re.DOTALL | re.IGNORECASE
    )

    # Given-When-Then patterns
    AC_ID_PATTERN = re.compile(r"AC\s*(\d+):?", re.IGNORECASE)
    GWT_GIVEN = re.compile(
        r"Given\s*(.+?)(?=When|Then|AC\s*\d+|$)", re.IGNORECASE | re.DOTALL
    )
    GWT_WHEN = re.compile(r"When\s*(.+?)(?=Then|AC\s*\d+|$)", re.IGNORECASE | re.DOTALL)
    GWT_THEN = re.compile(
        r"Then\s*(.+?)(?=Given|AC\s*\d+|$)", re.IGNORECASE | re.DOTALL
    )

    # Child work item pattern
    CHILD_PATTERN = re.compile(
        r"-\s*\[(?P<done>[xX ])\]\s*\*\*(?P<type>\w+)\s*#(?P<id>\d+)\*\*:\s*(?P<title>.+?)$",
        re.MULTILINE,
    )

    # Date patterns
    DATE_PATTERN = re.compile(
        r"\*(?:Created|Modified):\s*(\d{4}-\d{2}-\d{2}[T\d:\.Z]+)\*"
    )

    def can_process(self, file_path: Path) -> bool:
        """Check if this processor can handle the file."""
        if file_path.suffix.lower() != ".md":
            return False
        # Check if it looks like an ADO export
        try:
            content = file_path.read_text(encoding="utf-8")[:500]
            return bool(self.HEADER_PATTERN.search(content))
        except Exception:
            return False

    def get_supported_extensions(self) -> List[str]:
        """Return supported file extensions."""
        return [".md"]

    def process(self, file_path: Path) -> ExtractedContent:
        """
        Process an ADO work item markdown file.

        Args:
            file_path: Path to the .md file

        Returns:
            ExtractedContent with parsed ADO data
        """
        content = self._read_file(file_path)
        work_item = self._parse_work_item(content, file_path)

        # Build text content for indexing
        text_content = self._build_text_content(work_item)

        # Build source location
        source_location = SourceLocation(
            source_type=SourceType.ADO_WORK_ITEM,
            title=f"[#{work_item.id}] {work_item.title}",
            file_path=str(file_path),
        )

        return ExtractedContent(
            text=text_content,
            metadata={
                "ado_id": work_item.id,
                "title": work_item.title,
                "work_item_type": work_item.work_item_type,
                "state": work_item.state,
                "assigned_to": work_item.assigned_to,
                "iteration_path": work_item.iteration_path,
                "area_path": work_item.area_path,
                "parent_id": work_item.parent_id,
                "child_ids": work_item.child_ids,
                "story_points": work_item.story_points,
                "priority": work_item.priority,
                "acceptance_criteria": [
                    self._ac_to_dict(ac) for ac in work_item.acceptance_criteria
                ],
                "children": [self._child_to_dict(c) for c in work_item.children],
                "created_date": work_item.created_date,
                "modified_date": work_item.modified_date,
                "ac_count": len(work_item.acceptance_criteria),
                "child_count": len(work_item.children),
            },
            sections=[ac.id for ac in work_item.acceptance_criteria],
        )

    def _read_file(self, file_path: Path) -> str:
        """Read file with encoding detection."""
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
        for encoding in encodings:
            try:
                return file_path.read_text(encoding=encoding)
            except (UnicodeDecodeError, LookupError):
                continue
        return file_path.read_bytes().decode("utf-8", errors="ignore")

    def _parse_work_item(self, content: str, file_path: Path) -> ADOWorkItem:
        """
        Parse a work item from markdown content.

        Rule #4: Reduced from 72 lines to <60 lines via helper extraction
        """
        work_id, work_type = self._extract_work_item_id_and_type(content, file_path)

        # Extract title
        title_match = self.TITLE_PATTERN.search(content)
        title = title_match.group(1).strip() if title_match else file_path.stem
        fields = self._extract_work_item_fields(content)
        (
            description,
            acceptance_criteria,
            children,
            child_ids,
        ) = self._extract_work_item_content(content)

        # Extract dates
        created_date, modified_date = self._extract_work_item_dates(content)

        return ADOWorkItem(
            id=work_id,
            title=title,
            work_item_type=work_type,
            state=fields["state"],
            assigned_to=fields["assigned_to"],
            iteration_path=fields["iteration"],
            area_path=fields["area"],
            parent_id=fields["parent_id"],
            child_ids=child_ids,
            story_points=fields["story_points"],
            priority=fields["priority"],
            description=description,
            acceptance_criteria=acceptance_criteria,
            children=children,
            created_date=created_date,
            modified_date=modified_date,
            file_path=str(file_path),
        )

    def _extract_work_item_id_and_type(
        self, content: str, file_path: Path
    ) -> tuple[int, str]:
        """
        Extract work item ID and type.

        Rule #4: Extracted to reduce function size
        """
        header_match = self.HEADER_PATTERN.search(content)
        if header_match:
            return int(header_match.group("id")), header_match.group("type")

        # Try to extract from filename
        filename_match = re.match(r"(\d+)_", file_path.stem)
        work_id = int(filename_match.group(1)) if filename_match else 0
        work_type = self._guess_type_from_path(file_path)
        return work_id, work_type

    def _extract_work_item_fields(self, content: str) -> dict[str, Any]:
        """
        Extract work item fields from content.

        Rule #4: Extracted to reduce function size
        """
        parent_match = self.FIELD_PATTERNS["parent"].search(content)
        story_points_match = self.FIELD_PATTERNS["story_points"].search(content)
        priority_match = self.FIELD_PATTERNS["priority"].search(content)

        return {
            "state": self._extract_field(content, "state", "Unknown"),
            "assigned_to": self._extract_field(content, "assigned_to"),
            "iteration": self._extract_field(content, "iteration", ""),
            "area": self._extract_field(content, "area", ""),
            "parent_id": int(parent_match.group(1)) if parent_match else None,
            "story_points": float(story_points_match.group(1))
            if story_points_match
            else None,
            "priority": int(priority_match.group(1)) if priority_match else None,
        }

    def _extract_work_item_content(self, content: str) -> tuple:
        """
        Extract description, acceptance criteria, and children.

        Rule #4: Extracted to reduce function size
        """
        # Description
        desc_match = self.DESCRIPTION_SECTION.search(content)
        description = self._clean_html(desc_match.group(1)) if desc_match else ""

        # Acceptance criteria
        ac_match = self.ACCEPTANCE_SECTION.search(content)
        acceptance_criteria = []
        if ac_match:
            acceptance_criteria = self._parse_acceptance_criteria(ac_match.group(1))

        # Children
        children_match = self.CHILDREN_SECTION.search(content)
        children = []
        child_ids = []
        if children_match:
            children, child_ids = self._parse_children(children_match.group(1))

        return description, acceptance_criteria, children, child_ids

    def _extract_work_item_dates(self, content: str) -> tuple:
        """
        Extract created and modified dates.

        Rule #4: Extracted to reduce function size
        """
        dates = self.DATE_PATTERN.findall(content)
        created_date = dates[0] if dates else None
        modified_date = dates[1] if len(dates) > 1 else None
        return created_date, modified_date

    def _extract_field(self, content: str, field: str, default: Any = None) -> Any:
        """Extract a field from the table."""
        pattern = self.FIELD_PATTERNS.get(field)
        if pattern:
            match = pattern.search(content)
            if match:
                return match.group(1).strip()
        return default

    def _guess_type_from_path(self, file_path: Path) -> str:
        """Guess work item type from file path."""
        path_str = str(file_path).lower()
        if "feature" in path_str:
            return "Feature"
        if "user_story" in path_str or "userstory" in path_str:
            return "User Story"
        if "bug" in path_str:
            return "Bug"
        if "task" in path_str:
            return "Task"
        if "epic" in path_str:
            return "Epic"
        return "Unknown"

    def _clean_html(self, html: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", html)
        # Decode common entities
        text = text.replace("&nbsp;", " ")
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _parse_acceptance_criteria(self, content: str) -> List[AcceptanceCriteria]:
        """Parse Given-When-Then acceptance criteria.

        Rule #1: Reduced nesting via helper extraction
        """
        criteria = []

        # Clean HTML first
        clean_content = self._clean_html(content)

        # Split by AC markers
        ac_splits = re.split(r"(AC\s*\d+:?)", clean_content, flags=re.IGNORECASE)

        current_ac_id = "AC 00"
        current_text = ""

        for i, part in enumerate(ac_splits):
            ac_id_match = self.AC_ID_PATTERN.match(part.strip())
            if ac_id_match:
                # Save previous AC if exists
                self._save_acceptance_criterion(current_ac_id, current_text, criteria)
                current_ac_id = f"AC {ac_id_match.group(1).zfill(2)}"
                current_text = ""
            else:
                current_text += part

        # Don't forget the last one
        self._save_acceptance_criterion(current_ac_id, current_text, criteria)

        return criteria

    def _save_acceptance_criterion(
        self, ac_id: str, text: str, criteria_list: List[AcceptanceCriteria]
    ) -> None:
        """Save acceptance criterion if it has valid content.

        Rule #1: Extracted to reduce nesting
        Rule #4: Helper function <60 lines
        """
        if not text.strip():
            return

        ac = self._parse_single_ac(ac_id, text)
        if ac:
            criteria_list.append(ac)

    def _parse_single_ac(self, ac_id: str, text: str) -> Optional[AcceptanceCriteria]:
        """Parse a single AC into GWT format."""
        given = []
        when = []
        then = []

        # Extract Given clauses
        given_match = self.GWT_GIVEN.search(text)
        if given_match:
            given_text = given_match.group(1).strip()
            given = [
                g.strip()
                for g in re.split(r"\s*,\s*and\s*|\s*,\s*", given_text)
                if g.strip()
            ]

        # Extract When clauses
        when_match = self.GWT_WHEN.search(text)
        if when_match:
            when_text = when_match.group(1).strip()
            when = [
                w.strip()
                for w in re.split(r"\s*,\s*and\s*|\s*,\s*", when_text)
                if w.strip()
            ]

        # Extract Then clauses
        then_match = self.GWT_THEN.search(text)
        if then_match:
            then_text = then_match.group(1).strip()
            then = [
                t.strip()
                for t in re.split(r"\s*,\s*and\s*|\s*,\s*", then_text)
                if t.strip()
            ]

        # Only return if we have at least one GWT clause
        if given or when or then:
            return AcceptanceCriteria(
                id=ac_id,
                given=given,
                when=when,
                then=then,
                raw_text=text.strip()[:500],  # Limit raw text length
            )
        return None

    def _parse_children(self, content: str) -> tuple:
        """Parse child work item list."""
        children = []
        child_ids = []

        for match in self.CHILD_PATTERN.finditer(content):
            child = ChildWorkItem(
                id=int(match.group("id")),
                title=match.group("title").strip(),
                work_type=match.group("type"),
                completed=match.group("done").lower() == "x",
            )
            children.append(child)
            child_ids.append(child.id)

        return children, child_ids

    def _build_text_content(self, work_item: ADOWorkItem) -> str:
        """Build searchable text content."""
        parts = []

        # Header
        parts.append(f"[{work_item.work_item_type} #{work_item.id}] {work_item.title}")
        parts.append(f"State: {work_item.state}")

        if work_item.iteration_path:
            parts.append(f"Iteration: {work_item.iteration_path}")
        if work_item.area_path:
            parts.append(f"Area: {work_item.area_path}")
        if work_item.assigned_to:
            parts.append(f"Assigned: {work_item.assigned_to}")
        if work_item.story_points:
            parts.append(f"Story Points: {work_item.story_points}")

        # Description
        if work_item.description:
            parts.append("\nDescription:")
            parts.append(work_item.description)

        # Acceptance Criteria
        if work_item.acceptance_criteria:
            parts.append("\nAcceptance Criteria:")
            for ac in work_item.acceptance_criteria:
                parts.append(f"\n{ac.id}:")
                if ac.given:
                    parts.append(f"  GIVEN: {'; '.join(ac.given)}")
                if ac.when:
                    parts.append(f"  WHEN: {'; '.join(ac.when)}")
                if ac.then:
                    parts.append(f"  THEN: {'; '.join(ac.then)}")

        # Children
        if work_item.children:
            parts.append("\nChild Work Items:")
            for child in work_item.children:
                status = "✓" if child.completed else "○"
                parts.append(f"  {status} {child.work_type} #{child.id}: {child.title}")

        return "\n".join(parts)

    def _ac_to_dict(self, ac: AcceptanceCriteria) -> Dict[str, Any]:
        """Convert AcceptanceCriteria to dictionary."""
        return {
            "id": ac.id,
            "given": ac.given,
            "when": ac.when,
            "then": ac.then,
        }

    def _child_to_dict(self, child: ChildWorkItem) -> Dict[str, Any]:
        """Convert ChildWorkItem to dictionary."""
        return {
            "id": child.id,
            "title": child.title,
            "type": child.work_type,
            "completed": child.completed,
        }
