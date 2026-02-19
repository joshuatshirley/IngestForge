"""
Few-Shot Example Registry.

Few-Shot Registry
Manages storage and retrieval of extraction examples.

JPL Compliance:
- Rule #2: Bounded retrieval and file iteration (MAX_TOTAL_EXAMPLES).
- Rule #4: All functions < 60 lines.
- Rule #7: Checked return values for file operations.
- Rule #9: 100% type hints.
"""

import json
import os
from pathlib import Path
from typing import List, Optional
from ingestforge.learning.models import FewShotExample
from ingestforge.core.config_loaders import load_config
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_EXAMPLES_RETRIEVED = 1000
MAX_TOTAL_EXAMPLES = 10000  # Strict upper bound for file iteration
MAX_LINE_LENGTH = 100_000  # Safety limit for JSON lines


class FewShotRegistry:
    """Registry for storing and querying human-verified extraction pairs."""

    def __init__(self, storage_path: Optional[Path] = None):
        """Initialize registry with a storage file."""
        self.config = load_config()
        self.path = (
            storage_path
            or Path(self.config.get("learning_dir", ".data/learning"))
            / "few_shot.jsonl"
        )
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Create storage directory and file if missing."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.touch()

    def add_example(self, example: FewShotExample) -> bool:
        """Append a new verified example to the registry.

        JPL Rule #7: Return success status.
        """
        # JPL Rule #2: Check capacity before adding
        try:
            count = 0
            if self.path.exists():
                with open(self.path, "r") as f:
                    for _ in f:
                        count += 1

            if count >= MAX_TOTAL_EXAMPLES:
                logger.error("Registry at maximum capacity")
                return False

            line = example.model_dump_json()
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            return True
        except Exception as e:
            logger.error(f"Failed to add example to registry: {e}")
            return False

    def list_examples(
        self, domain: Optional[str] = None, limit: int = 10
    ) -> List[FewShotExample]:
        """List examples with optional domain filtering.

        JPL Rule #2: Strict iteration bound.
        """
        results: List[FewShotExample] = []
        # JPL Rule #2: Safety limit
        max_to_return = min(limit, MAX_EXAMPLES_RETRIEVED)

        if not self.path.exists():
            return results

        try:
            with open(self.path, "r", encoding="utf-8") as f:
                # JPL Rule #2: Use enumerate to bound loop
                for i, line in enumerate(f):
                    if i >= MAX_TOTAL_EXAMPLES or len(results) >= max_to_return:
                        break

                    example = self._parse_line(line)
                    if example and (domain is None or example.domain == domain):
                        results.append(example)
            return results
        except Exception as e:
            logger.error(f"Error reading registry: {e}")
            return []

    def _parse_line(self, line: str) -> Optional[FewShotExample]:
        """Parse a single line from the JSONL file.

        Rule #4: Concise parser.
        """
        if not line.strip() or len(line) > MAX_LINE_LENGTH:
            return None
        try:
            data = json.loads(line)
            return FewShotExample(**data)
        except (json.JSONDecodeError, ValueError):
            return None

    def remove_example(self, example_id: str) -> bool:
        """Remove an example by ID (rewrites file)."""
        temp_path = self.path.with_suffix(".tmp")
        removed = False

        try:
            with open(self.path, "r", encoding="utf-8") as f_in, open(
                temp_path, "w", encoding="utf-8"
            ) as f_out:
                # JPL Rule #2: Bound rewrite loop
                for i, line in enumerate(f_in):
                    if i >= MAX_TOTAL_EXAMPLES:
                        break
                    example = self._parse_line(line)
                    if example and example.id == example_id:
                        removed = True
                        continue
                    f_out.write(line)

            # Atomic swap
            os.replace(temp_path, self.path)
            return removed
        except Exception as e:
            logger.error(f"Failed to remove example: {e}")
            if temp_path.exists():
                os.remove(temp_path)
            return False
