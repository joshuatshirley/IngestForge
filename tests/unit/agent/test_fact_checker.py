"""Tests for adversarial fact-checker.

Tests multi-agent debate verification."""

from __future__ import annotations


from ingestforge.agent.fact_checker import (
    VerificationStatus,
    DebateRole,
    Evidence,
    DebateArgument,
    Claim,
    VerificationResult,
    DebateStrategy,
    DebateOrchestrator,
    create_orchestrator,
    simple_proponent,
    simple_critic,
    MAX_DEBATE_ROUNDS,
)

# VerificationStatus tests


class TestVerificationStatus:
    """Tests for VerificationStatus enum."""

    def test_statuses_defined(self) -> None:
        """Test all statuses are defined."""
        statuses = [s.value for s in VerificationStatus]

        assert "verified" in statuses
        assert "refuted" in statuses
        assert "uncertain" in statuses
        assert "contested" in statuses

    def test_status_count(self) -> None:
        """Test correct number of statuses."""
        assert len(VerificationStatus) == 4


# DebateRole tests


class TestDebateRole:
    """Tests for DebateRole enum."""

    def test_roles_defined(self) -> None:
        """Test all roles are defined."""
        roles = [r.value for r in DebateRole]

        assert "proponent" in roles
        assert "critic" in roles
        assert "judge" in roles


# Evidence tests


class TestEvidence:
    """Tests for Evidence dataclass."""

    def test_evidence_creation(self) -> None:
        """Test creating evidence."""
        evidence = Evidence(
            content="Data shows correlation",
            source="study.pdf",
            supports_claim=True,
            confidence=0.9,
        )

        assert evidence.content == "Data shows correlation"
        assert evidence.supports_claim is True

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        evidence = Evidence(
            content="Counter-example",
            source="test",
            supports_claim=False,
        )

        d = evidence.to_dict()

        assert d["supports"] is False
        assert d["source"] == "test"


# DebateArgument tests


class TestDebateArgument:
    """Tests for DebateArgument dataclass."""

    def test_argument_creation(self) -> None:
        """Test creating an argument."""
        arg = DebateArgument(
            role=DebateRole.PROPONENT,
            position="The evidence supports the claim",
        )

        assert arg.role == DebateRole.PROPONENT
        assert "evidence" in arg.position

    def test_argument_with_evidence(self) -> None:
        """Test argument with evidence."""
        evidence = [
            Evidence(content="Point 1", supports_claim=True),
            Evidence(content="Point 2", supports_claim=True),
        ]
        arg = DebateArgument(
            role=DebateRole.PROPONENT,
            position="Multiple supporting facts",
            evidence=evidence,
        )

        assert len(arg.evidence) == 2

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        arg = DebateArgument(
            role=DebateRole.CRITIC,
            position="Lacks evidence",
            round_number=2,
        )

        d = arg.to_dict()

        assert d["role"] == "critic"
        assert d["round"] == 2


# Claim tests


class TestClaim:
    """Tests for Claim dataclass."""

    def test_claim_creation(self) -> None:
        """Test creating a claim."""
        claim = Claim(
            content="The earth is round",
            source="science.txt",
        )

        assert claim.content == "The earth is round"

    def test_claim_truncation(self) -> None:
        """Test long claim is truncated."""
        long_content = "x" * 2000
        claim = Claim(content=long_content)

        assert len(claim.content) == 1000

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        claim = Claim(
            content="Test claim",
            source="test.md",
            context="Background info",
        )

        d = claim.to_dict()

        assert d["content"] == "Test claim"
        assert d["context"] == "Background info"


# VerificationResult tests


class TestVerificationResult:
    """Tests for VerificationResult dataclass."""

    def test_result_creation(self) -> None:
        """Test creating a result."""
        claim = Claim(content="Test")
        result = VerificationResult(
            claim=claim,
            status=VerificationStatus.VERIFIED,
            confidence=0.9,
            proponent_score=0.8,
            critic_score=0.3,
        )

        assert result.is_verified is True
        assert result.confidence == 0.9

    def test_is_verified(self) -> None:
        """Test is_verified property."""
        claim = Claim(content="Test")

        verified = VerificationResult(
            claim=claim,
            status=VerificationStatus.VERIFIED,
            confidence=0.9,
            proponent_score=0.8,
            critic_score=0.3,
        )
        refuted = VerificationResult(
            claim=claim,
            status=VerificationStatus.REFUTED,
            confidence=0.8,
            proponent_score=0.2,
            critic_score=0.7,
        )

        assert verified.is_verified is True
        assert refuted.is_verified is False

    def test_rounds_count(self) -> None:
        """Test rounds_count property."""
        claim = Claim(content="Test")
        args = [
            DebateArgument(role=DebateRole.PROPONENT, position="A"),
            DebateArgument(role=DebateRole.CRITIC, position="B"),
            DebateArgument(role=DebateRole.PROPONENT, position="C"),
            DebateArgument(role=DebateRole.CRITIC, position="D"),
        ]
        result = VerificationResult(
            claim=claim,
            status=VerificationStatus.VERIFIED,
            confidence=0.9,
            proponent_score=0.8,
            critic_score=0.3,
            arguments=args,
        )

        assert result.rounds_count == 2

    def test_to_dict(self) -> None:
        """Test converting to dict."""
        claim = Claim(content="Test claim")
        result = VerificationResult(
            claim=claim,
            status=VerificationStatus.CONTESTED,
            confidence=0.5,
            proponent_score=0.5,
            critic_score=0.5,
            summary="Close debate",
        )

        d = result.to_dict()

        assert d["status"] == "contested"
        assert d["summary"] == "Close debate"


# DebateStrategy tests


class TestDebateStrategy:
    """Tests for DebateStrategy class."""

    def test_proponent_prompt(self) -> None:
        """Test proponent prompt generation."""
        strategy = DebateStrategy()
        claim = Claim(content="Test claim")

        prompt = strategy.proponent_prompt(claim, [])

        assert "Defend" in prompt
        assert "Test claim" in prompt

    def test_critic_prompt(self) -> None:
        """Test critic prompt generation."""
        strategy = DebateStrategy()
        claim = Claim(content="Test claim")

        prompt = strategy.critic_prompt(claim, [])

        assert "Critique" in prompt
        assert "Test claim" in prompt

    def test_judge_prompt(self) -> None:
        """Test judge prompt generation."""
        strategy = DebateStrategy()
        claim = Claim(content="Test claim")

        prompt = strategy.judge_prompt(claim, [])

        assert "Judge" in prompt
        assert "verified" in prompt.lower()


# DebateOrchestrator tests


class TestDebateOrchestrator:
    """Tests for DebateOrchestrator class."""

    def test_orchestrator_creation(self) -> None:
        """Test creating an orchestrator."""
        orchestrator = DebateOrchestrator(
            proponent_fn=simple_proponent,
            critic_fn=simple_critic,
        )

        assert orchestrator is not None

    def test_verify_simple_claim(self) -> None:
        """Test verifying a simple claim."""
        orchestrator = DebateOrchestrator(
            proponent_fn=simple_proponent,
            critic_fn=simple_critic,
            max_rounds=2,
        )
        claim = Claim(content="The sky is blue")

        result = orchestrator.verify(claim)

        assert result is not None
        assert result.status in list(VerificationStatus)
        assert result.arguments

    def test_verify_empty_claim(self) -> None:
        """Test verifying empty claim."""
        orchestrator = DebateOrchestrator(
            proponent_fn=simple_proponent,
            critic_fn=simple_critic,
        )
        claim = Claim(content="")

        result = orchestrator.verify(claim)

        assert result.status == VerificationStatus.UNCERTAIN

    def test_max_rounds_enforced(self) -> None:
        """Test max rounds is enforced."""
        orchestrator = DebateOrchestrator(
            proponent_fn=simple_proponent,
            critic_fn=simple_critic,
            max_rounds=3,
        )
        claim = Claim(content="Test")

        result = orchestrator.verify(claim)

        # Should have at most 6 arguments (3 rounds * 2)
        assert len(result.arguments) <= 6


class TestDebateScoring:
    """Tests for debate scoring."""

    def test_proponent_scores(self) -> None:
        """Test proponent scoring."""
        orchestrator = DebateOrchestrator(
            proponent_fn=simple_proponent,
            critic_fn=simple_critic,
            max_rounds=2,
        )
        claim = Claim(content="Test claim")

        result = orchestrator.verify(claim)

        assert result.proponent_score >= 0.0
        assert result.proponent_score <= 1.0

    def test_critic_scores(self) -> None:
        """Test critic scoring."""
        orchestrator = DebateOrchestrator(
            proponent_fn=simple_proponent,
            critic_fn=simple_critic,
            max_rounds=2,
        )
        claim = Claim(content="Test claim")

        result = orchestrator.verify(claim)

        assert result.critic_score >= 0.0
        assert result.critic_score <= 1.0

    def test_confidence_in_range(self) -> None:
        """Test confidence is in valid range."""
        orchestrator = DebateOrchestrator(
            proponent_fn=simple_proponent,
            critic_fn=simple_critic,
        )
        claim = Claim(content="Test")

        result = orchestrator.verify(claim)

        assert 0.0 <= result.confidence <= 1.0


# Factory function tests


class TestCreateOrchestrator:
    """Tests for create_orchestrator factory."""

    def test_create(self) -> None:
        """Test creating orchestrator."""
        orchestrator = create_orchestrator(
            proponent_fn=simple_proponent,
            critic_fn=simple_critic,
        )

        assert isinstance(orchestrator, DebateOrchestrator)

    def test_create_with_rounds(self) -> None:
        """Test creating with custom rounds."""
        orchestrator = create_orchestrator(
            proponent_fn=simple_proponent,
            critic_fn=simple_critic,
            max_rounds=5,
        )

        assert orchestrator._max_rounds == 5


# Simple participant tests


class TestSimpleParticipants:
    """Tests for simple debate participants."""

    def test_simple_proponent(self) -> None:
        """Test simple proponent function."""
        arg = simple_proponent("Defend claim", [])

        assert arg.role == DebateRole.PROPONENT
        assert arg.position
        assert arg.evidence

    def test_simple_critic(self) -> None:
        """Test simple critic function."""
        arg = simple_critic("Critique claim", [])

        assert arg.role == DebateRole.CRITIC
        assert arg.position
        assert arg.evidence


# Constant tests


class TestConstants:
    """Tests for module constants."""

    def test_max_debate_rounds(self) -> None:
        """Test MAX_DEBATE_ROUNDS is reasonable."""
        assert MAX_DEBATE_ROUNDS > 0
        assert MAX_DEBATE_ROUNDS == 10
