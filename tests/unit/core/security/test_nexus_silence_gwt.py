"""
GWT Unit Tests for Nexus Global Silence (Isolation) - Task 256.
NASA JPL Compliance: Rule #4, Rule #9.
"""

import pytest
from pathlib import Path
from ingestforge.storage.nexus_registry import NexusRegistry
from ingestforge.core.pipeline.nexus_broadcast import NexusBroadcaster
from ingestforge.core.models.search import SearchQuery
from ingestforge.core.config.nexus import NexusConfig


@pytest.fixture
def temp_registry(tmp_path):
    return NexusRegistry(tmp_path)


# =============================================================================
# GIVEN: A system in Global Silence mode
# =============================================================================


def test_silence_given_enabled_when_checked_then_returns_true(temp_registry):
    # When
    temp_registry.set_silence(True)

    # Then
    assert temp_registry.is_silenced() is True


@pytest.mark.asyncio
async def test_silence_given_enabled_when_broadcasting_then_short_circuits(
    temp_registry,
):
    # Given
    temp_registry.set_silence(True)
    mock_config = NexusConfig(
        nexus_id="local", cert_file=Path("c"), key_file=Path("k"), trust_store=Path("t")
    )
    broadcaster = NexusBroadcaster(temp_registry, mock_config)
    query = SearchQuery(text="test", broadcast=True)

    # When
    response = await broadcaster.broadcast(query)

    # Then
    assert len(response.results) == 0
    assert len(response.peer_failures) == 1
    assert "disabled by global silence" in response.peer_failures[0].message


def test_silence_given_disabled_when_checked_then_returns_false(temp_registry):
    # When
    temp_registry.set_silence(False)

    # Then
    assert temp_registry.is_silenced() is False
