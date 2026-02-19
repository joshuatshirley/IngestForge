"""
GEDCOM processor for Genealogy vertical.

Parses GEDCOM files (.ged) and extracts individuals, families, and events
into standardized metadata.
"""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class GedcomEntity:
    """Base class for GEDCOM entities."""

    id: str
    tag: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    sub_entities: List["GedcomEntity"] = field(default_factory=list)


class GedcomProcessor:
    """
    Parses GEDCOM files and extracts individuals and families.
    """

    def process(self, file_path: Path) -> List[Dict[str, Any]]:
        """Process a GEDCOM file and return list of entity dictionaries."""
        if not file_path.exists():
            return []

        content = file_path.read_text(encoding="utf-8-sig")
        lines = content.splitlines()

        entities = self._parse_lines(lines)
        return [self._to_record(e) for e in entities if e.tag in ("INDI", "FAM")]

    def _parse_lines(self, lines: List[str]) -> List[GedcomEntity]:
        """Parse GEDCOM lines into entities."""
        stack: List[GedcomEntity] = []
        root_entities: List[GedcomEntity] = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            match = re.match(r"^(\d+)\s+(@\w+@)?\s*(\w+)\s*(.*)$", line)
            if not match:
                continue

            level, entity_id, tag, value = match.groups()
            level = int(level)

            # If tag and entity_id are swapped (standard GEDCOM)
            if entity_id and tag not in ("INDI", "FAM", "SOUR", "REPO", "OBJE"):
                # Tag is actually at position 2, ID is at position 3
                # e.g., 0 @I1@ INDI
                pass
            elif not entity_id and value.startswith("@"):
                # Value might be an ID or tag swapped
                pass

            # Handle level popping
            while len(stack) > level:
                stack.pop()

            entity = GedcomEntity(id=entity_id or "", tag=tag)
            if value:
                entity.metadata["value"] = value

            if level == 0:
                root_entities.append(entity)
                stack = [entity]
            else:
                if stack:
                    stack[-1].sub_entities.append(entity)
                    stack.append(entity)

        return root_entities

    def _to_record(self, entity: GedcomEntity) -> Dict[str, Any]:
        """Convert entity to standardized record."""
        record = {
            "id": entity.id,
            "type": entity.tag,
            "name": self._get_sub_value(entity, "NAME"),
            "metadata": {},
        }

        if entity.tag == "INDI":
            record["birth_date"] = self._get_nested_value(entity, "BIRT", "DATE")
            record["death_date"] = self._get_nested_value(entity, "DEAT", "DATE")
            record["sex"] = self._get_sub_value(entity, "SEX")

        return record

    def _get_sub_value(self, entity: GedcomEntity, tag: str) -> Optional[str]:
        for sub in entity.sub_entities:
            if sub.tag == tag:
                return sub.metadata.get("value")
        return None

    def _get_nested_value(
        self, entity: GedcomEntity, parent_tag: str, child_tag: str
    ) -> Optional[str]:
        for sub in entity.sub_entities:
            if sub.tag == parent_tag:
                return self._get_sub_value(sub, child_tag)
        return None
