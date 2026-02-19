from ingestforge.core.pipeline.artifacts import IFFileArtifact, IFTextArtifact


def test_if_text_artifact_auto_hashing():
    """
    GWT:
    Given a TextArtifact with content
    When initialized without a hash
    Then it must automatically generate the SHA-256 hash.
    """
    content = "Integrity Check"
    art = IFTextArtifact(artifact_id="t1", content=content)
    assert art.content_hash is not None
    assert len(art.content_hash) == 64  # SHA-256 length


def test_if_artifact_derivation():
    """
    GWT:
    Given a parent artifact
    When derive() is called with a processor ID
    Then the new artifact must have the correct parent_id and updated provenance.
    """
    parent = IFTextArtifact(artifact_id="parent-1", content="original")
    child = parent.derive(processor_id="proc-v1", content="modified")

    assert child.parent_id == "parent-1"
    assert "proc-v1" in child.provenance
    assert child.content == "modified"
    assert (
        child.artifact_id == "parent-1"
    )  # derive defaults to same ID unless overridden


def test_if_file_artifact_hashing(tmp_path):
    """
    GWT:
    Given a FileArtifact pointing to a real file
    When initialized
    Then it must calculate the file hash.
    """
    d = tmp_path / "data"
    d.mkdir()
    f = d / "test.txt"
    f.write_text("file content")

    art = IFFileArtifact(artifact_id="f1", file_path=f)
    assert art.content_hash is not None
