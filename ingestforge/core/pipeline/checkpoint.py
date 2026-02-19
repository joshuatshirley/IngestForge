"""
Checkpoint Manager for IngestForge (IF).

Handles persistence of IFArtifacts between pipeline stages.
Follows NASA JPL Power of Ten rules.
"""

import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from ingestforge.core.pipeline.interfaces import IFArtifact

# JPL Rule #2: Fixed upper bound on checkpoints per document
MAX_CHECKPOINTS_PER_DOCUMENT = 64

logger = logging.getLogger(__name__)


class IFCheckpointManager:
    """
    Manages saving and loading of pipeline artifacts.
    """

    def __init__(self, base_dir: Optional[Path] = None):
        if base_dir is None:
            # Default to .processing/checkpoints in current directory
            base_dir = Path(".processing/checkpoints")
        self.base_dir = base_dir

    def save_checkpoint(
        self, artifact: IFArtifact, document_id: str, stage_name: str
    ) -> bool:
        """
        Save an artifact to disk.

        Rule #7: Check return values of IO operations.
        """
        try:
            doc_dir = self.base_dir / document_id
            doc_dir.mkdir(parents=True, exist_ok=True)

            file_path = doc_dir / f"{stage_name}.json"
            json_data = artifact.model_dump_json(indent=2)

            with open(file_path, "w", encoding="utf-8") as f:
                f.write(json_data)

            # Verify file exists and has content
            if not file_path.exists() or file_path.stat().st_size == 0:
                logger.error(f"Failed to verify checkpoint write: {file_path}")
                return False

            logger.debug(f"Saved checkpoint: {file_path}")
            return True

        except Exception as e:
            logger.error(
                f"Error saving checkpoint for {document_id} at {stage_name}: {e}"
            )
            return False

    def load_checkpoint(
        self, artifact_type: type, document_id: str, stage_name: str
    ) -> Optional[IFArtifact]:
        """
        Load an artifact from disk.
        """
        file_path = self.base_dir / document_id / f"{stage_name}.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # Use Pydantic's validate_python if using artifact_type
            # Or model_validate for specific models
            return artifact_type.model_validate(data)

        except Exception as e:
            logger.error(f"Error loading checkpoint from {file_path}: {e}")
            return None

    def list_checkpoints(self, document_id: str) -> List[str]:
        """
        List all available checkpoint stage names for a document.

        Recovery - Deterministic Resumption.
        Rule #2: Bounded result (MAX_CHECKPOINTS_PER_DOCUMENT).
        Rule #7: Check return values.

        Args:
            document_id: The document identifier.

        Returns:
            List of stage names with checkpoints, sorted by modification time.
        """
        doc_dir = self.base_dir / document_id

        if not doc_dir.exists():
            return []

        try:
            # Find all .json checkpoint files
            checkpoint_files = list(doc_dir.glob("*.json"))
            if len(checkpoint_files) > MAX_CHECKPOINTS_PER_DOCUMENT:
                logger.warning(
                    f"Document {document_id} has {len(checkpoint_files)} checkpoints, "
                    f"exceeds limit of {MAX_CHECKPOINTS_PER_DOCUMENT}"
                )
                checkpoint_files = checkpoint_files[:MAX_CHECKPOINTS_PER_DOCUMENT]

            # Sort by modification time (oldest first)
            checkpoint_files.sort(key=lambda p: p.stat().st_mtime)

            # Extract stage names (remove .json extension)
            return [f.stem for f in checkpoint_files]

        except Exception as e:
            logger.error(f"Error listing checkpoints for {document_id}: {e}")
            return []

    def get_latest_checkpoint(
        self, document_id: str, stage_order: List[str]
    ) -> Optional[Tuple[str, int]]:
        """
        Find the most recent checkpoint based on stage execution order.

        Recovery - Deterministic Resumption.
        Rule #7: Check return values.

        Args:
            document_id: The document identifier.
            stage_order: List of stage names in execution order.

        Returns:
            Tuple of (stage_name, stage_index) for the latest checkpoint,
            or None if no valid checkpoints exist.
        """
        available = self.list_checkpoints(document_id)

        if not available:
            return None

        # Create index map for stage order
        stage_index_map: Dict[str, int] = {
            name: idx for idx, name in enumerate(stage_order)
        }

        # Find the checkpoint with the highest stage index
        latest_stage: Optional[str] = None
        latest_index: int = -1

        for stage_name in available:
            if stage_name in stage_index_map:
                idx = stage_index_map[stage_name]
                if idx > latest_index:
                    latest_index = idx
                    latest_stage = stage_name

        if latest_stage is None:
            logger.warning(
                f"No checkpoints for {document_id} match known stages: {stage_order}"
            )
            return None

        logger.debug(
            f"Latest checkpoint for {document_id}: {latest_stage} (index {latest_index})"
        )
        return (latest_stage, latest_index)

    def clear_checkpoints(self, document_id: str) -> bool:
        """
        Remove all checkpoints for a document.

        Recovery - Deterministic Resumption.
        Rule #7: Check return values.

        Args:
            document_id: The document identifier.

        Returns:
            True if cleared successfully, False on error.
        """
        doc_dir = self.base_dir / document_id

        if not doc_dir.exists():
            return True  # Nothing to clear

        try:
            for checkpoint_file in doc_dir.glob("*.json"):
                checkpoint_file.unlink()

            # Remove directory if empty
            if not any(doc_dir.iterdir()):
                doc_dir.rmdir()

            logger.debug(f"Cleared checkpoints for {document_id}")
            return True

        except Exception as e:
            logger.error(f"Error clearing checkpoints for {document_id}: {e}")
            return False
