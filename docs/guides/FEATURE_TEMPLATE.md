# Feature Development Checklist

Use this checklist when adding new features to ensure completeness and consistency.

> This checklist covers the **Execution and Verification phases** (Steps 4-9) of the
> [Project Workflow](../planning/WORKFLOW.md). Before starting here, ensure
> Steps 1-3 (Requirements Intake, Prioritization, Assignment) are complete.
> After finishing here, proceed to Steps 10-14 (Approval, Delivery, Closure).

---

## Pre-Requisites (Workflow Steps 1-3)

**Before using this template, verify these are done:**

- [ ] GitHub issue or BACKLOG.md entry exists with problem statement
- [ ] Acceptance criteria defined (specific, testable conditions)
- [ ] Priority assigned (P0-P3) and size estimated (S/M/L/XL)
- [ ] Scope boundaries documented (what's IN and OUT)
- [ ] Owner assigned and feature branch created
- [ ] Scheduled in current sprint or backlog

> See [WORKFLOW.md Steps 1-3](../planning/WORKFLOW.md#phase-1-initiation) for details.

---

## Planning Phase

**Before writing any code, complete these steps:**

- [ ] Feature proposal documented in `BACKLOG.md` or issue
- [ ] Requirements clear and agreed upon with stakeholders
- [ ] User stories or use cases defined
- [ ] Architectural approach decided (ADR written if significant)
- [ ] Interface design reviewed (if adding new interfaces)
- [ ] Dependencies identified (new libraries, tools, services)
- [ ] Breaking changes identified and migration plan outlined
- [ ] Performance impact estimated
- [ ] Security implications reviewed

**Questions to Answer:**
- What problem does this feature solve?
- Who are the users and what's their use case?
- Does this fit with existing architecture?
- Are there similar features in the codebase to follow?
- What's the success metric?

---

## Implementation Phase

**Code quality standards:**

- [ ] Code follows existing patterns and conventions
- [ ] Type hints added to all public functions and methods
- [ ] Docstrings added (class, method, and function level)
- [ ] Error handling implemented with specific exceptions
- [ ] Logging added for key operations (info, warning, error levels)
- [ ] Configuration options added to `config.yaml` schema if needed
- [ ] No hardcoded values (use config or constants)
- [ ] Code is DRY (no duplication)
- [ ] Functions are single-responsibility and focused
- [ ] No TODO or FIXME comments left in code

**Code Example Standards:**

```python
from typing import List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class MyFeature:
    """Brief description of what this class does.

    Longer explanation if needed, including:
    - Key responsibilities
    - Important behavior notes
    - Usage patterns

    Attributes:
        param1: Description of param1
        param2: Description of param2
    """

    param1: str
    param2: Optional[int] = None

    def process(self, input_data: str) -> List[str]:
        """Process input data and return results.

        Args:
            input_data: Description of what this is

        Returns:
            List of processed items

        Raises:
            ValueError: If input_data is empty
            ProcessingError: If processing fails

        Example:
            >>> feature = MyFeature(param1="test")
            >>> result = feature.process("input")
            >>> print(result)
            ['output1', 'output2']
        """
        if not input_data:
            raise ValueError("input_data cannot be empty")

        logger.info(f"Processing data with {self.param1}")

        try:
            # Implementation
            result = self._internal_process(input_data)
            logger.debug(f"Processed {len(result)} items")
            return result
        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise ProcessingError(f"Failed to process: {e}") from e
```

---

## Testing Phase

**Test coverage requirements:**

- [ ] Unit tests written (>80% coverage for new code)
- [ ] Integration tests added (if cross-module feature)
- [ ] Edge cases tested (empty input, None, invalid types)
- [ ] Error paths tested (exceptions, timeouts, failures)
- [ ] Performance tested (if relevant - benchmarks for slow operations)
- [ ] All tests pass locally
- [ ] No test warnings or deprecations

**Test Organization:**

```
tests/
├── unit/
│   └── test_my_feature.py          # Unit tests
├── integration/
│   └── test_my_feature_integration.py  # Integration tests
└── fixtures/
    └── my_feature_test_data.json   # Test data
```

**Test Example:**

```python
import pytest
from ingestforge.mymodule import MyFeature, ProcessingError


class TestMyFeature:
    """Tests for MyFeature class."""

    def test_process_valid_input(self):
        """Test processing with valid input."""
        feature = MyFeature(param1="test")
        result = feature.process("input")

        assert isinstance(result, list)
        assert len(result) > 0

    def test_process_empty_input_raises(self):
        """Test that empty input raises ValueError."""
        feature = MyFeature(param1="test")

        with pytest.raises(ValueError, match="cannot be empty"):
            feature.process("")

    def test_process_with_config(self, config):
        """Test processing with configuration."""
        feature = MyFeature(param1=config.my_feature.param1)
        result = feature.process("input")

        assert result is not None

    @pytest.mark.parametrize("input_data,expected", [
        ("test1", ["result1"]),
        ("test2", ["result2", "result3"]),
    ])
    def test_process_parametrized(self, input_data, expected):
        """Test processing with various inputs."""
        feature = MyFeature(param1="test")
        result = feature.process(input_data)

        assert result == expected
```

---

## Documentation Phase

**Documentation requirements:**

- [ ] Module README.md updated (if new module created)
- [ ] Usage examples added to module README
- [ ] API.md updated (if REST API endpoint added)
- [ ] ARCHITECTURE.md updated (if architecture changed)
- [ ] ADR written (if architectural decision made)
- [ ] CHANGELOG.md updated with new feature
- [ ] Docstrings complete and accurate
- [ ] Configuration documented in comments
- [ ] Migration guide written (if breaking changes)

**Module README Update Checklist:**

```markdown
# My Module

## Purpose
[One-sentence description]

## Key Components

| Component | Purpose | Status |
|-----------|---------|--------|
| MyFeature | Does X | ✅ Complete |  <!-- Add your feature here -->

## Usage Examples

### Example: Using MyFeature
\`\`\`python
from ingestforge.mymodule import MyFeature

# Basic usage
feature = MyFeature(param1="value")
result = feature.process("input")
print(result)
\`\`\`

## Configuration

\`\`\`yaml
# config.yaml
my_feature:
  param1: "default_value"
  enabled: true
\`\`\`
```

**CHANGELOG.md Update:**

```markdown
## [Unreleased]

### Added
- MyFeature for processing X (implements IProcessor interface)
- Configuration option `my_feature.param1` in config.yaml
- 15 new tests for MyFeature

### Changed
- Updated pipeline to optionally use MyFeature
- Improved error messages in processing module

### Dependencies
- Added `new-library>=1.0.0` (optional, for MyFeature)
```

---

## Review Phase

**Pre-PR checklist:**

- [ ] Self-review completed (read your own diff)
- [ ] All checklist items above completed
- [ ] No debug code or print statements left
- [ ] No commented-out code blocks
- [ ] No merge conflicts
- [ ] Branch is up-to-date with main
- [ ] CI/CD checks passing (if available)
- [ ] Performance benchmarks run (if applicable)

**Pull Request Description Template:**

```markdown
## Description

Brief description of what this PR does.

Fixes #123 (if applicable)

## Type of Change

- [ ] New feature
- [ ] Bug fix
- [ ] Breaking change
- [ ] Documentation update
- [ ] Refactoring

## Changes Made

- Added MyFeature class implementing IProcessor
- Updated pipeline to support new feature
- Added 15 unit tests and 3 integration tests
- Updated module README with usage examples

## Testing

- [ ] All existing tests pass
- [ ] New tests added and passing
- [ ] Manual testing completed

**Manual test scenario:**
1. Initialize MyFeature with config
2. Process test input
3. Verify output format
4. Test error handling

## Documentation

- [ ] README updated
- [ ] API docs updated (if applicable)
- [ ] CHANGELOG.md updated
- [ ] ADR written (if architectural change)
- [ ] Docstrings complete

## Breaking Changes

None / List breaking changes and migration path

## Performance Impact

Negligible / Describe impact and benchmarks

## Screenshots (if applicable)

[Add screenshots of CLI output, UI changes, etc.]

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Comments added for complex logic
- [ ] Documentation updated
- [ ] Tests added
- [ ] No new warnings
```

---

## Merge Phase

**Final checks before merging:**

- [ ] All review comments addressed
- [ ] Approvals received (if required)
- [ ] Final tests passing in CI
- [ ] Documentation finalized and reviewed
- [ ] CHANGELOG.md entry confirmed
- [ ] Migration guide tested (if breaking changes)
- [ ] Merged to main branch
- [ ] TODO.md updated (move from in-progress to completed)
- [ ] Issue/ticket closed (if applicable)

**Post-Merge:**

- [ ] Feature announced (if user-facing)
- [ ] Documentation site updated (if applicable)
- [ ] Example code updated in docs
- [ ] Delete feature branch

---

## Delivery Phase (Workflow Steps 11-12)

**After merge, deliver and communicate:**

- [ ] CI/CD pipeline passes on main branch
- [ ] Docker image builds successfully (if applicable)
- [ ] Version bump applied (if releasing)
- [ ] Release tagged in git (if versioned release)
- [ ] Related GitHub issues closed with resolution comment
- [ ] Downstream teams/users notified of breaking changes (if any)
- [ ] BACKLOG.md entry marked complete
- [ ] TODO.md updated

---

## Closure Phase (Workflow Steps 13-14)

**Wrap up and learn:**

- [ ] Feature branch deleted
- [ ] Retrospective notes captured (what went well, what was difficult)
- [ ] Process improvements identified (if any)
- [ ] All tracking artifacts closed and accurate
- [ ] Temporary workarounds documented for future removal (if any)

> See [WORKFLOW.md Steps 13-14](../planning/WORKFLOW.md#phase-5-closure) for retrospective format.

---

## Example: Adding a New Enricher

**Complete worked example showing all phases.**

### 1. Planning

**Feature Proposal (in BACKLOG.md):**

```markdown
### New Feature: Sentiment Analysis Enricher
**Priority:** Medium
**Status:** Planned
**Description:** Add sentiment analysis to chunks using VADER or TextBlob
**Use Case:** Help users find positive/negative sections in arguments
**Estimated Effort:** 2-3 days
```

**Architectural Decision (ADR-007):**

```markdown
# ADR-007: Sentiment Analysis Enricher

## Context
Research papers often contain arguments with positive/negative framing.
Users want to find sections that are supportive vs. critical.

## Decision
Implement SentimentEnricher using VADER (lexicon-based, fast, no training).

## Consequences
✅ Fast (no ML model)
✅ Simple to deploy
⚠️ Less accurate than transformer models
```

### 2. Implementation

**File: `ingestforge/enrichment/sentiment.py`**

```python
"""Sentiment analysis enricher using VADER.

This module provides sentiment analysis for chunk content,
classifying text as positive, negative, or neutral.
"""

from typing import Optional
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from ingestforge.chunking.semantic_chunker import ChunkRecord
from ingestforge.shared.patterns import IEnricher
from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class SentimentEnricher(IEnricher):
    """Add sentiment scores to chunks using VADER.

    VADER (Valence Aware Dictionary and sEntiment Reasoner) is a
    lexicon and rule-based sentiment analysis tool specifically
    tuned to sentiments expressed in social media.

    Attributes:
        threshold: Compound score threshold for classification
                   (default: 0.05, range: 0.0-1.0)

    Example:
        >>> enricher = SentimentEnricher(threshold=0.1)
        >>> chunk = ChunkRecord("1", "This is amazing!")
        >>> enriched = enricher.enrich_chunk(chunk)
        >>> print(enriched.sentiment)
        'positive'
    """

    def __init__(self, threshold: float = 0.05):
        """Initialize sentiment enricher.

        Args:
            threshold: Compound score threshold for pos/neg classification
                      Values above threshold are positive,
                      values below -threshold are negative,
                      values in between are neutral.

        Raises:
            ValueError: If threshold not in range 0.0-1.0
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be 0.0-1.0, got {threshold}")

        self.threshold = threshold
        self._analyzer: Optional[SentimentIntensityAnalyzer] = None
        logger.info(f"SentimentEnricher initialized with threshold={threshold}")

    @property
    def analyzer(self) -> SentimentIntensityAnalyzer:
        """Lazy-load VADER analyzer."""
        if self._analyzer is None:
            try:
                from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
                self._analyzer = SentimentIntensityAnalyzer
                logger.debug("VADER analyzer loaded")
            except ImportError as e:
                logger.error("vaderSentiment not installed")
                raise ImportError(
                    "vaderSentiment required for sentiment analysis. "
                    "Install with: pip install vaderSentiment"
                ) from e
        return self._analyzer

    def enrich_chunk(self, chunk: ChunkRecord) -> ChunkRecord:
        """Add sentiment analysis to chunk.

        Args:
            chunk: Chunk to analyze

        Returns:
            Chunk with sentiment and sentiment_score fields populated

        Example:
            >>> chunk = ChunkRecord("1", "This is terrible.")
            >>> enriched = enricher.enrich_chunk(chunk)
            >>> print(enriched.sentiment)
            'negative'
            >>> print(enriched.sentiment_score)
            -0.54
        """
        logger.debug(f"Analyzing sentiment for chunk {chunk.chunk_id}")

        scores = self.analyzer.polarity_scores(chunk.content)
        compound = scores['compound']

        # Classify based on threshold
        if compound >= self.threshold:
            sentiment = 'positive'
        elif compound <= -self.threshold:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'

        # Attach to chunk
        chunk.sentiment = sentiment
        chunk.sentiment_score = compound

        logger.debug(
            f"Chunk {chunk.chunk_id}: {sentiment} "
            f"(score={compound:.2f})"
        )

        return chunk

    def is_available(self) -> bool:
        """Check if VADER is available.

        Returns:
            True if vaderSentiment is installed, False otherwise
        """
        try:
            _ = self.analyzer
            return True
        except ImportError:
            logger.warning("vaderSentiment not available")
            return False

    def get_metadata(self) -> dict:
        """Return enricher metadata.

        Returns:
            Dictionary with enricher information
        """
        return {
            "name": "SentimentEnricher",
            "version": "1.0.0",
            "library": "vaderSentiment",
            "threshold": self.threshold,
        }
```

### 3. Testing

**File: `tests/unit/test_enrichment_sentiment.py`**

```python
"""Tests for sentiment enricher."""

import pytest
from ingestforge.enrichment.sentiment import SentimentEnricher
from ingestforge.chunking.semantic_chunker import ChunkRecord


class TestSentimentEnricher:
    """Tests for SentimentEnricher class."""

    def test_positive_sentiment(self):
        """Test positive sentiment classification."""
        enricher = SentimentEnricher
        chunk = ChunkRecord("1", "This is amazing and wonderful!")

        result = enricher.enrich_chunk(chunk)

        assert result.sentiment == 'positive'
        assert result.sentiment_score > 0
        assert hasattr(result, 'sentiment')

    def test_negative_sentiment(self):
        """Test negative sentiment classification."""
        enricher = SentimentEnricher
        chunk = ChunkRecord("2", "This is terrible and awful.")

        result = enricher.enrich_chunk(chunk)

        assert result.sentiment == 'negative'
        assert result.sentiment_score < 0

    def test_neutral_sentiment(self):
        """Test neutral sentiment classification."""
        enricher = SentimentEnricher
        chunk = ChunkRecord("3", "The study was conducted in 2024.")

        result = enricher.enrich_chunk(chunk)

        assert result.sentiment == 'neutral'
        assert -0.05 <= result.sentiment_score <= 0.05

    def test_custom_threshold(self):
        """Test custom threshold configuration."""
        enricher = SentimentEnricher(threshold=0.2)
        chunk = ChunkRecord("4", "This is good.")

        result = enricher.enrich_chunk(chunk)

        # With higher threshold, weakly positive may be neutral
        assert result.sentiment in ['positive', 'neutral']

    def test_threshold_validation(self):
        """Test threshold parameter validation."""
        with pytest.raises(ValueError, match="must be 0.0-1.0"):
            SentimentEnricher(threshold=1.5)

        with pytest.raises(ValueError, match="must be 0.0-1.0"):
            SentimentEnricher(threshold=-0.1)

    def test_is_available(self):
        """Test availability check."""
        enricher = SentimentEnricher
        assert enricher.is_available is True

    def test_get_metadata(self):
        """Test metadata retrieval."""
        enricher = SentimentEnricher(threshold=0.1)
        metadata = enricher.get_metadata

        assert metadata['name'] == 'SentimentEnricher'
        assert metadata['threshold'] == 0.1
        assert 'version' in metadata

    def test_batch_processing(self):
        """Test batch enrichment."""
        enricher = SentimentEnricher
        chunks = [
            ChunkRecord("1", "This is great!"),
            ChunkRecord("2", "This is awful."),
            ChunkRecord("3", "This is neutral."),
        ]

        results = enricher.enrich_batch(chunks)

        assert len(results) == 3
        assert results[0].sentiment == 'positive'
        assert results[1].sentiment == 'negative'
        assert results[2].sentiment == 'neutral'

    @pytest.mark.parametrize("text,expected_sentiment", [
        ("I love this!", "positive"),
        ("I hate this!", "negative"),
        ("The sky is blue.", "neutral"),
        ("Absolutely fantastic!", "positive"),
        ("Completely terrible!", "negative"),
    ])
    def test_sentiment_examples(self, text, expected_sentiment):
        """Test sentiment classification with various examples."""
        enricher = SentimentEnricher
        chunk = ChunkRecord("test", text)

        result = enricher.enrich_chunk(chunk)

        assert result.sentiment == expected_sentiment
```

### 4. Documentation

**Update: `ingestforge/enrichment/README.md`**

```markdown
# Enrichment Module

## Key Components

| Component | Purpose | Status |
|-----------|---------|--------|
| EntityExtractor | Extract named entities | ✅ Complete |
| QuestionGenerator | Generate hypothetical questions | ✅ Complete |
| EmbeddingGenerator | Generate embeddings | ✅ Complete |
| **SentimentEnricher** | **Analyze sentiment** | **✅ Complete** |

...

## Usage Examples

### Example 4: Sentiment Analysis
\`\`\`python
from ingestforge.enrichment.sentiment import SentimentEnricher

enricher = SentimentEnricher(threshold=0.1)
chunk = enricher.enrich_chunk(chunk)

print(f"Sentiment: {chunk.sentiment}")  # 'positive', 'negative', 'neutral'
print(f"Score: {chunk.sentiment_score}")  # -1.0 to 1.0
\`\`\`

### Example 5: Pipeline with Sentiment
\`\`\`python
from ingestforge.shared.patterns import EnrichmentPipeline
from ingestforge.enrichment.sentiment import SentimentEnricher

pipeline = EnrichmentPipeline([
    EntityExtractor,
    SentimentEnricher,
    EmbeddingGenerator(config),
])

enriched = pipeline.enrich(chunks)
\`\`\`

## Dependencies

### Optional
- `vaderSentiment>=3.3.2` - For sentiment analysis
```

**Update: `CHANGELOG.md`**

```markdown
## [Unreleased]

### Added
- Sentiment analysis enricher using VADER (ADR-007)
- SentimentEnricher implements IEnricher interface
- Configuration option `enrichment.sentiment.threshold`
- 12 new tests for sentiment analysis

### Dependencies
- Added vaderSentiment>=3.3.2 (optional)
```

**Update: `README.md`** (Features section)

```markdown
### Enrichment
- ✅ Entity extraction (spaCy + regex fallback)
- ✅ Hypothetical question generation (LLM + templates)
- ✅ Embedding generation (sentence-transformers)
- ✅ **Sentiment analysis (VADER)** ← NEW
```

### 5. Pull Request

**PR Title:** `feat: add sentiment analysis enricher`

**PR Description:**

```markdown
## Description

Adds sentiment analysis enricher using VADER for classifying chunk sentiment as positive, negative, or neutral.

Closes #87

## Type of Change

- [x] New feature
- [ ] Bug fix
- [ ] Breaking change
- [ ] Documentation update

## Changes Made

- Added `SentimentEnricher` class implementing `IEnricher`
- Added 12 unit tests with 100% coverage
- Updated enrichment README with usage examples
- Added ADR-007 documenting decision to use VADER
- Updated CHANGELOG.md

## Testing

- [x] All existing tests pass
- [x] New tests added (12 unit tests)
- [x] Manual testing completed

**Manual test scenario:**
1. Created test chunk with positive content
2. Applied SentimentEnricher
3. Verified sentiment='positive' and score>0
4. Tested with negative and neutral content
5. Tested pipeline integration

## Documentation

- [x] enrichment/README.md updated
- [x] CHANGELOG.md updated
- [x] ADR-007 written
- [x] Docstrings complete

## Breaking Changes

None

## Performance Impact

Negligible - VADER is lexicon-based and very fast (~0.1ms per chunk)
```

### 6. Post-Merge

- [x] Feature merged to main
- [x] Issue #87 closed
- [x] BACKLOG.md updated (moved to completed)
- [x] Feature branch deleted

---

## Tips for Success

**Do:**
- ✅ Read existing code in the module before adding new code
- ✅ Follow established patterns (e.g., all enrichers implement IEnricher)
- ✅ Write tests first (TDD) or immediately after code
- ✅ Document as you go, not at the end
- ✅ Ask questions early if unsure about approach
- ✅ Make small, focused PRs (easier to review)

**Don't:**
- ❌ Skip tests "I'll add them later" (you won't)
- ❌ Skip documentation "It's self-explanatory" (it isn't)
- ❌ Make large, multi-feature PRs
- ❌ Introduce dependencies without discussion
- ❌ Break existing tests without fixing them
- ❌ Leave TODO comments in production code

**Code Review Tips:**
- Be respectful and constructive
- Explain the "why" behind suggestions
- Accept that multiple approaches can be valid
- Focus on correctness, readability, and maintainability
- Don't nitpick style (use automated formatters)

---

## Integration with Workflows

**GitHub Issues:**
- Use issue templates for feature requests
- Link PRs to issues with "Closes #123"
- Add labels: `feature`, `enhancement`, `bug`, etc.

**Continuous Integration:**
- All tests must pass before merge
- Code coverage must not decrease
- Linting must pass (flake8, mypy)
- Documentation must build successfully

**Release Process:**
- Features go into `main` branch
- Version bumps follow semantic versioning
- CHANGELOG.md drives release notes
- Breaking changes require major version bump

---

*This template ensures consistent, well-documented features that integrate smoothly with IngestForge's architecture.*
