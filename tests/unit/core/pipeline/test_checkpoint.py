from pathlib import Path
from ingestforge.core.pipeline.artifacts import IFTextArtifact
from ingestforge.core.pipeline.checkpoint import IFCheckpointManager
from ingestforge.core.pipeline.runner import IFPipelineRunner
from ingestforge.core.pipeline.interfaces import IFStage, IFArtifact


class MockStage(IFStage):
    def __init__(self, name):
        self._name = name

    @property
    def name(self):
        return self._name

    @property
    def input_type(self):
        return IFArtifact

    @property
    def output_type(self):
        return IFArtifact

    def execute(self, artifact):
        return artifact.derive(processor_id=f"stage-{self.name}")


def test_checkpoint_save_and_load(tmp_path):
    """
    GWT:
    Given a CheckpointManager
    When an artifact is saved and then loaded
    Then the loaded artifact must match the original.
    """
    cm = IFCheckpointManager(base_dir=tmp_path)
    art = IFTextArtifact(artifact_id="1", content="hello")

    success = cm.save_checkpoint(art, "doc-1", "stage-A")
    assert success is True

    loaded = cm.load_checkpoint(IFTextArtifact, "doc-1", "stage-A")
    assert loaded is not None
    assert loaded.content == "hello"
    assert loaded.artifact_id == "1"


def test_runner_with_checkpointing(tmp_path):
    """
    GWT:
    Given a Runner with a CheckpointManager
    When run() is called
    Then checkpoints are created on disk.
    """
    cm = IFCheckpointManager(base_dir=tmp_path)
    runner = IFPipelineRunner(checkpoint_manager=cm)
    art = IFTextArtifact(artifact_id="1", content="start")
    stages = [MockStage("A"), MockStage("B")]

    runner.run(art, stages, document_id="doc-123")

    # Check if files exist
    assert (tmp_path / "doc-123" / "A.json").exists()
    assert (tmp_path / "doc-123" / "B.json").exists()


# =============================================================================
# Recovery - Deterministic Resumption Tests
# =============================================================================


class TestListCheckpoints:
    """Tests for IFCheckpointManager.list_checkpoints()"""

    def test_list_checkpoints_empty(self, tmp_path):
        """
        GWT:
        Given no checkpoints exist for a document
        When list_checkpoints() is called
        Then an empty list is returned.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        result = cm.list_checkpoints("nonexistent-doc")
        assert result == []

    def test_list_checkpoints_returns_stage_names(self, tmp_path):
        """
        GWT:
        Given multiple checkpoints exist
        When list_checkpoints() is called
        Then stage names are returned.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        art = IFTextArtifact(artifact_id="1", content="test")

        cm.save_checkpoint(art, "doc-1", "extraction")
        cm.save_checkpoint(art, "doc-1", "chunking")
        cm.save_checkpoint(art, "doc-1", "embedding")

        result = cm.list_checkpoints("doc-1")

        assert "extraction" in result
        assert "chunking" in result
        assert "embedding" in result

    def test_list_checkpoints_sorted_by_time(self, tmp_path):
        """
        GWT:
        Given checkpoints created in sequence
        When list_checkpoints() is called
        Then they are sorted by modification time (oldest first).
        """
        import time

        cm = IFCheckpointManager(base_dir=tmp_path)
        art = IFTextArtifact(artifact_id="1", content="test")

        cm.save_checkpoint(art, "doc-1", "first")
        time.sleep(0.01)  # Small delay to ensure different timestamps
        cm.save_checkpoint(art, "doc-1", "second")
        time.sleep(0.01)
        cm.save_checkpoint(art, "doc-1", "third")

        result = cm.list_checkpoints("doc-1")

        assert result[0] == "first"
        assert result[-1] == "third"


class TestGetLatestCheckpoint:
    """Tests for IFCheckpointManager.get_latest_checkpoint()"""

    def test_get_latest_no_checkpoints(self, tmp_path):
        """
        GWT:
        Given no checkpoints exist
        When get_latest_checkpoint() is called
        Then None is returned.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        result = cm.get_latest_checkpoint("doc-1", ["A", "B", "C"])
        assert result is None

    def test_get_latest_returns_highest_index(self, tmp_path):
        """
        GWT:
        Given checkpoints for multiple stages
        When get_latest_checkpoint() is called
        Then the stage with highest index is returned.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        art = IFTextArtifact(artifact_id="1", content="test")

        cm.save_checkpoint(art, "doc-1", "extraction")
        cm.save_checkpoint(art, "doc-1", "chunking")

        stage_order = ["extraction", "chunking", "embedding", "storage"]
        result = cm.get_latest_checkpoint("doc-1", stage_order)

        assert result is not None
        stage_name, stage_index = result
        assert stage_name == "chunking"
        assert stage_index == 1

    def test_get_latest_ignores_unknown_stages(self, tmp_path):
        """
        GWT:
        Given a checkpoint for an unknown stage
        When get_latest_checkpoint() is called
        Then only known stages are considered.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        art = IFTextArtifact(artifact_id="1", content="test")

        cm.save_checkpoint(art, "doc-1", "extraction")
        cm.save_checkpoint(art, "doc-1", "unknown_stage")

        stage_order = ["extraction", "chunking"]
        result = cm.get_latest_checkpoint("doc-1", stage_order)

        assert result is not None
        stage_name, _ = result
        assert stage_name == "extraction"


class TestClearCheckpoints:
    """Tests for IFCheckpointManager.clear_checkpoints()"""

    def test_clear_removes_all_checkpoints(self, tmp_path):
        """
        GWT:
        Given multiple checkpoints exist
        When clear_checkpoints() is called
        Then all are removed.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        art = IFTextArtifact(artifact_id="1", content="test")

        cm.save_checkpoint(art, "doc-1", "A")
        cm.save_checkpoint(art, "doc-1", "B")

        result = cm.clear_checkpoints("doc-1")

        assert result is True
        assert cm.list_checkpoints("doc-1") == []

    def test_clear_nonexistent_succeeds(self, tmp_path):
        """
        GWT:
        Given no checkpoints exist
        When clear_checkpoints() is called
        Then True is returned (nothing to clear).
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        result = cm.clear_checkpoints("nonexistent")
        assert result is True


class TestResumeFromCheckpoint:
    """Tests for IFPipelineRunner.resume()"""

    def test_resume_no_checkpoint_manager(self, tmp_path):
        """
        GWT:
        Given a runner without checkpoint manager
        When resume() is called
        Then None is returned.
        """
        runner = IFPipelineRunner(checkpoint_manager=None)
        stages = [MockStage("A"), MockStage("B")]

        result = runner.resume(stages, "doc-1")
        assert result is None

    def test_resume_no_checkpoint_returns_none(self, tmp_path):
        """
        GWT:
        Given no checkpoint exists
        When resume() is called
        Then None is returned indicating fresh start.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        runner = IFPipelineRunner(checkpoint_manager=cm)
        stages = [MockStage("A"), MockStage("B")]

        result = runner.resume(stages, "doc-1")
        assert result is None

    def test_resume_returns_artifact_and_remaining_stages(self, tmp_path):
        """
        GWT:
        Given a checkpoint exists at stage "A"
        When resume() is called
        Then artifact and remaining stages [B, C] are returned.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        runner = IFPipelineRunner(checkpoint_manager=cm)

        # Create checkpoint
        art = IFTextArtifact(artifact_id="1", content="after-A")
        cm.save_checkpoint(art, "doc-1", "A")

        stages = [MockStage("A"), MockStage("B"), MockStage("C")]
        result = runner.resume(stages, "doc-1", artifact_type=IFTextArtifact)

        assert result is not None
        loaded_artifact, remaining_stages = result

        assert loaded_artifact.content == "after-A"
        assert len(remaining_stages) == 2
        assert remaining_stages[0].name == "B"
        assert remaining_stages[1].name == "C"

    def test_resume_skips_completed_stages(self, tmp_path):
        """
        GWT:
        Given checkpoint at "chunking" (stage 2 of 4)
        When resume() is called
        Then only stages 3 and 4 remain.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        runner = IFPipelineRunner(checkpoint_manager=cm)

        art = IFTextArtifact(artifact_id="1", content="chunked")
        cm.save_checkpoint(art, "doc-1", "chunking")

        stages = [
            MockStage("extraction"),
            MockStage("chunking"),
            MockStage("embedding"),
            MockStage("storage"),
        ]

        result = runner.resume(stages, "doc-1", artifact_type=IFTextArtifact)

        assert result is not None
        _, remaining = result

        assert len(remaining) == 2
        assert remaining[0].name == "embedding"
        assert remaining[1].name == "storage"


class TestRunWithResume:
    """Tests for IFPipelineRunner.run_with_resume()"""

    def test_run_with_resume_fresh_start(self, tmp_path):
        """
        GWT:
        Given no checkpoint exists
        When run_with_resume() is called
        Then all stages execute from beginning.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        runner = IFPipelineRunner(checkpoint_manager=cm)
        art = IFTextArtifact(artifact_id="1", content="start")
        stages = [MockStage("A"), MockStage("B")]

        result = runner.run_with_resume(art, stages, "doc-1")

        assert "stage-A" in result.provenance
        assert "stage-B" in result.provenance

    def test_run_with_resume_continues_from_checkpoint(self, tmp_path):
        """
        GWT:
        Given a checkpoint exists after stage A
        When run_with_resume() is called
        Then only stage B executes.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        runner = IFPipelineRunner(checkpoint_manager=cm)

        # Create checkpoint after A
        art_after_a = IFTextArtifact(
            artifact_id="1", content="after-A", provenance=["stage-A"]
        )
        cm.save_checkpoint(art_after_a, "doc-1", "A")

        # Initial artifact (should be ignored since checkpoint exists)
        initial = IFTextArtifact(artifact_id="1", content="initial")
        stages = [MockStage("A"), MockStage("B")]

        result = runner.run_with_resume(initial, stages, "doc-1")

        # Should have A from checkpoint and B from execution
        assert "stage-A" in result.provenance
        assert "stage-B" in result.provenance

    def test_run_with_resume_already_complete(self, tmp_path):
        """
        GWT:
        Given checkpoint at final stage
        When run_with_resume() is called
        Then no stages execute and loaded artifact is returned.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        runner = IFPipelineRunner(checkpoint_manager=cm)

        # Create checkpoint at final stage
        final_art = IFTextArtifact(
            artifact_id="1", content="complete", provenance=["stage-A", "stage-B"]
        )
        cm.save_checkpoint(final_art, "doc-1", "B")

        initial = IFTextArtifact(artifact_id="1", content="initial")

        # Use MockStage with IFTextArtifact as output type so checkpoint loading works
        class TextMockStage(IFStage):
            def __init__(self, name):
                self._name = name

            @property
            def name(self):
                return self._name

            @property
            def input_type(self):
                return IFTextArtifact

            @property
            def output_type(self):
                return IFTextArtifact

            def execute(self, artifact):
                return artifact.derive(processor_id=f"stage-{self.name}")

        stages = [TextMockStage("A"), TextMockStage("B")]

        result = runner.run_with_resume(initial, stages, "doc-1")

        assert result.content == "complete"


class TestCheckpointIntegrity:
    """Tests for checkpoint integrity verification"""

    def test_corrupted_checkpoint_returns_none(self, tmp_path):
        """
        GWT:
        Given a corrupted checkpoint file
        When load_checkpoint() is called
        Then None is returned.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)

        # Create corrupted checkpoint
        doc_dir = tmp_path / "doc-1"
        doc_dir.mkdir(parents=True)
        (doc_dir / "A.json").write_text("{invalid json")

        result = cm.load_checkpoint(IFTextArtifact, "doc-1", "A")
        assert result is None

    def test_resume_with_corrupted_checkpoint_returns_none(self, tmp_path):
        """
        GWT:
        Given a corrupted checkpoint file
        When resume() is called
        Then None is returned.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        runner = IFPipelineRunner(checkpoint_manager=cm)

        # Create corrupted checkpoint
        doc_dir = tmp_path / "doc-1"
        doc_dir.mkdir(parents=True)
        (doc_dir / "A.json").write_text("{invalid json")

        stages = [MockStage("A"), MockStage("B")]
        result = runner.resume(stages, "doc-1", artifact_type=IFTextArtifact)

        assert result is None


# =============================================================================
# COMPREHENSIVE COVERAGE TESTS - Additional GWT scenarios for full coverage
# =============================================================================


class TestCheckpointWriteVerification:
    """
    Tests for checkpoint write verification failure (lines 50-52).

    Resilience - Multi-Stage Checkpointing.
    JPL Rule #7: Check return values.
    """

    def test_save_returns_false_if_file_not_created(self, tmp_path, monkeypatch):
        """
        GWT:
        Given a checkpoint manager,
        When the file write appears to succeed but file doesn't exist after,
        Then save_checkpoint returns False.
        """
        from unittest.mock import patch

        cm = IFCheckpointManager(base_dir=tmp_path)
        art = IFTextArtifact(artifact_id="1", content="hello")

        # Patch file operations to simulate write that doesn't persist
        original_open = open
        writes_done = []

        def mock_open_fn(path, mode="r", *args, **kwargs):
            if "w" in mode:
                writes_done.append(str(path))
                # Return a mock file that discards writes
                from io import StringIO

                return StringIO()
            return original_open(path, mode, *args, **kwargs)

        with patch("builtins.open", mock_open_fn):
            result = cm.save_checkpoint(art, "doc-1", "stage-A")

        # File doesn't exist, so should return False
        assert result is False

    def test_save_returns_false_if_file_empty(self, tmp_path, monkeypatch):
        """
        GWT:
        Given a checkpoint manager,
        When the file is created but empty,
        Then save_checkpoint returns False.
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        art = IFTextArtifact(artifact_id="1", content="hello")

        # First, let normal save happen
        success = cm.save_checkpoint(art, "doc-1", "stage-A")
        assert success is True

        # Now truncate the file to empty
        file_path = tmp_path / "doc-1" / "stage-A.json"
        file_path.write_text("")

        # Try to save again - after patching to prevent actual write
        # The verification check should catch the empty file
        # Actually, the save will overwrite... let me think differently

        # Create a scenario where the file gets truncated between write and verify
        from unittest.mock import patch

        original_stat = Path.stat

        def mock_stat(self):
            if "stage-B" in str(self):
                # Simulate empty file
                class FakeStat:
                    st_size = 0

                return FakeStat()
            return original_stat(self)

        with patch.object(Path, "stat", mock_stat):
            with patch.object(Path, "exists", return_value=True):
                result = cm.save_checkpoint(art, "doc-1", "stage-B")

        # Should fail verification due to empty file
        assert result is False


class TestCheckpointSaveException:
    """
    Tests for exception during checkpoint save (lines 57-59).

    Resilience - Multi-Stage Checkpointing.
    JPL Rule #7: Check return values.
    """

    def test_save_handles_write_exception(self, tmp_path):
        """
        GWT:
        Given a checkpoint manager,
        When an exception occurs during file write,
        Then save_checkpoint returns False without crashing.
        """
        from unittest.mock import patch

        cm = IFCheckpointManager(base_dir=tmp_path)
        art = IFTextArtifact(artifact_id="1", content="hello")

        # Patch open to raise exception
        def raise_on_write(*args, **kwargs):
            if "w" in str(args):
                raise IOError("Disk full")
            return open(*args, **kwargs)

        with patch("builtins.open", side_effect=IOError("Disk full")):
            result = cm.save_checkpoint(art, "doc-1", "stage-A")

        assert result is False

    def test_save_handles_serialization_exception(self, tmp_path):
        """
        GWT:
        Given an artifact that cannot be serialized,
        When save_checkpoint is called,
        Then it returns False without crashing.
        """
        from unittest.mock import patch

        cm = IFCheckpointManager(base_dir=tmp_path)
        art = IFTextArtifact(artifact_id="1", content="hello")

        # Patch model_dump_json to raise
        with patch.object(
            IFTextArtifact,
            "model_dump_json",
            side_effect=ValueError("Serialization failed"),
        ):
            result = cm.save_checkpoint(art, "doc-1", "stage-A")

        assert result is False


class TestCheckpointLimitEnforcement:
    """
    Tests for checkpoint limit enforcement (lines 106-111).

    Recovery - Deterministic Resumption.
    JPL Rule #2: Bounded result.
    """

    def test_list_checkpoints_enforces_limit(self, tmp_path):
        """
        GWT:
        Given more checkpoints than MAX_CHECKPOINTS_PER_DOCUMENT,
        When list_checkpoints() is called,
        Then only MAX_CHECKPOINTS_PER_DOCUMENT are returned.
        """
        from ingestforge.core.pipeline.checkpoint import MAX_CHECKPOINTS_PER_DOCUMENT

        cm = IFCheckpointManager(base_dir=tmp_path)
        art = IFTextArtifact(artifact_id="1", content="hello")

        # Create more checkpoints than the limit
        for i in range(MAX_CHECKPOINTS_PER_DOCUMENT + 5):
            cm.save_checkpoint(art, "doc-1", f"stage-{i:03d}")

        result = cm.list_checkpoints("doc-1")

        # Should be capped at MAX_CHECKPOINTS_PER_DOCUMENT
        assert len(result) <= MAX_CHECKPOINTS_PER_DOCUMENT

    def test_list_checkpoints_logs_warning_when_exceeds_limit(self, tmp_path):
        """
        GWT:
        Given more checkpoints than the limit,
        When list_checkpoints() is called,
        Then a warning is logged.
        """
        from ingestforge.core.pipeline.checkpoint import MAX_CHECKPOINTS_PER_DOCUMENT
        from unittest.mock import patch

        cm = IFCheckpointManager(base_dir=tmp_path)
        art = IFTextArtifact(artifact_id="1", content="hello")

        # Create more checkpoints than the limit
        for i in range(MAX_CHECKPOINTS_PER_DOCUMENT + 5):
            cm.save_checkpoint(art, "doc-1", f"stage-{i:03d}")

        with patch("ingestforge.core.pipeline.checkpoint.logger") as mock_logger:
            cm.list_checkpoints("doc-1")
            # Should have logged a warning
            mock_logger.warning.assert_called_once()
            warning_msg = mock_logger.warning.call_args[0][0]
            assert "exceeds limit" in warning_msg.lower()


class TestCheckpointListException:
    """
    Tests for exception during checkpoint listing (lines 119-121).

    Recovery - Deterministic Resumption.
    JPL Rule #7: Check return values.
    """

    def test_list_checkpoints_handles_permission_error(self, tmp_path):
        """
        GWT:
        Given a directory with permission issues,
        When list_checkpoints() is called,
        Then an empty list is returned without crashing.
        """
        from unittest.mock import patch

        cm = IFCheckpointManager(base_dir=tmp_path)

        # Create directory
        doc_dir = tmp_path / "doc-1"
        doc_dir.mkdir(parents=True)

        # Patch glob to raise exception
        with patch.object(Path, "glob", side_effect=PermissionError("Access denied")):
            result = cm.list_checkpoints("doc-1")

        assert result == []

    def test_list_checkpoints_handles_stat_error(self, tmp_path):
        """
        GWT:
        Given a checkpoint file that cannot be stat'd,
        When list_checkpoints() is called,
        Then an empty list is returned without crashing.
        """
        from unittest.mock import patch

        cm = IFCheckpointManager(base_dir=tmp_path)
        art = IFTextArtifact(artifact_id="1", content="hello")

        # Create a checkpoint
        cm.save_checkpoint(art, "doc-1", "stage-A")

        # Patch stat to raise exception during sorting
        original_stat = Path.stat
        call_count = [0]

        def raise_on_stat(self, **kwargs):
            call_count[0] += 1
            if call_count[0] > 2:  # Allow exists() checks but fail during sort
                raise OSError("Cannot stat file")
            return original_stat(self, **kwargs)

        with patch.object(Path, "stat", raise_on_stat):
            result = cm.list_checkpoints("doc-1")

        assert result == []


class TestCheckpointJPLCompliance:
    """
    Additional JPL Power of Ten compliance tests for checkpoint module.
    """

    def test_jpl_rule_2_max_checkpoints_constant_exists(self):
        """
        GWT:
        Given the checkpoint module,
        When imported,
        Then MAX_CHECKPOINTS_PER_DOCUMENT constant is defined.
        """
        from ingestforge.core.pipeline.checkpoint import MAX_CHECKPOINTS_PER_DOCUMENT

        assert isinstance(MAX_CHECKPOINTS_PER_DOCUMENT, int)
        assert MAX_CHECKPOINTS_PER_DOCUMENT > 0

    def test_jpl_rule_7_all_methods_return_explicitly(self, tmp_path):
        """
        GWT:
        Given checkpoint manager methods,
        When called,
        Then they return explicit values (not None from missing return).
        """
        cm = IFCheckpointManager(base_dir=tmp_path)
        art = IFTextArtifact(artifact_id="1", content="hello")

        # save_checkpoint returns bool
        result1 = cm.save_checkpoint(art, "doc-1", "A")
        assert isinstance(result1, bool)

        # load_checkpoint returns IFArtifact or None
        result2 = cm.load_checkpoint(IFTextArtifact, "doc-1", "A")
        assert result2 is None or isinstance(result2, IFArtifact)

        # list_checkpoints returns list
        result3 = cm.list_checkpoints("doc-1")
        assert isinstance(result3, list)

    def test_jpl_rule_9_type_hints_present(self):
        """
        GWT:
        Given checkpoint manager methods,
        When inspecting type hints,
        Then all parameters are annotated.
        """
        import typing

        hints_save = typing.get_type_hints(IFCheckpointManager.save_checkpoint)
        assert "artifact" in hints_save
        assert "document_id" in hints_save
        assert "stage_name" in hints_save
        assert "return" in hints_save

        hints_load = typing.get_type_hints(IFCheckpointManager.load_checkpoint)
        assert "artifact_type" in hints_load
        assert "document_id" in hints_load
        assert "stage_name" in hints_load
        assert "return" in hints_load
