# Evidence Linker

**Status:** ✅ Implemented (P3-AI-002.2)

## Overview

The Evidence Linker connects claims with supporting/refuting evidence from the knowledge base using semantic similarity and contradiction detection. It's a core component of the fact-checking system.

## Architecture

```
┌─────────────┐
│    Claim    │
└──────┬──────┘
       │
       ↓
┌──────────────────────┐
│  Evidence Linker     │
│  - Search KB         │
│  - Classify Support  │
│  - Score Confidence  │
└──────┬───────────────┘
       │
       ↓
┌──────────────────────┐
│ Linked Evidence      │
│ - Supporting         │
│ - Refuting           │
│ - Neutral            │
└──────────────────────┘
```

## Key Components

### EvidenceLinker

Main class that links claims to evidence.

**Methods:**
- `link_evidence(claim, storage, top_k)`: Search and classify evidence
- `classify_support(claim, evidence)`: Classify single piece of evidence

**Parameters:**
- `support_threshold`: Minimum similarity for supporting evidence (default: 0.6)
- `refute_threshold`: Minimum contradiction score for refutation (default: 0.6)
- `contradiction_detector`: ContradictionDetector instance

### Data Types

#### SupportType (Enum)
- `SUPPORTS`: Evidence supports the claim
- `REFUTES`: Evidence contradicts the claim
- `NEUTRAL`: Evidence neither supports nor refutes

#### LinkedEvidence (Dataclass)
- `evidence_text`: The evidence content
- `source`: Source document ID
- `chunk_id`: Unique evidence identifier
- `relevance_score`: Similarity score (0.0-1.0)
- `support_type`: SupportType classification
- `confidence`: Confidence in classification (0.0-1.0)
- `metadata`: Additional chunk metadata

#### EvidenceLinkResult (Dataclass)
- `claim`: Original claim text
- `linked_evidence`: List of LinkedEvidence items
- `total_support`: Count of supporting evidence
- `total_refute`: Count of refuting evidence
- `total_neutral`: Count of neutral evidence
- `key_entities`: Extracted entities from claim

## Usage

### Basic Evidence Linking

```python
from ingestforge.enrichment import EvidenceLinker
from ingestforge.storage.chromadb import ChromaDBRepository

# Create linker
linker = EvidenceLinker(
    support_threshold=0.6,
    refute_threshold=0.2,
)

# Link evidence to claim
storage = ChromaDBRepository
result = linker.link_evidence(
    claim="The Earth orbits the Sun",
    storage=storage,
    top_k=10,
)

# Examine results
print(f"Support: {result.total_support}")
print(f"Refute: {result.total_refute}")

for evidence in result.linked_evidence:
    print(f"{evidence.support_type.value}: {evidence.evidence_text}")
    print(f"  Confidence: {evidence.confidence:.2f}")
```

### Direct Classification

```python
# Classify a specific claim-evidence pair
classification = linker.classify_support(
    claim="Water freezes at 0°C",
    evidence="Ice forms when water reaches zero degrees Celsius",
)

print(classification)  # SupportType.SUPPORTS
```

### Integration with ContradictionDetector

```python
from ingestforge.enrichment import ContradictionDetector, EvidenceLinker

# Create custom contradiction detector
detector = ContradictionDetector(
    similarity_threshold=0.7,
    negation_boost=0.3,
)

# Use with evidence linker
linker = EvidenceLinker(
    contradiction_detector=detector,
    refute_threshold=0.2,
)
```

## Configuration

### Thresholds

**Support Threshold (0.0-1.0)**
- Minimum semantic similarity to classify as supporting
- Higher = stricter requirements for support
- Recommended: 0.6 for balanced results

**Refute Threshold (0.0-1.0)**
- Minimum contradiction score to classify as refuting
- Note: Contradiction scores are typically lower than similarity scores
- Recommended: 0.2-0.3 for effective contradiction detection

### Fixed Upper Bounds (JPL Rule #2)

- `MAX_EVIDENCE_ITEMS = 100`: Maximum evidence items to process
- `MAX_CLAIM_LENGTH = 5000`: Maximum characters in claim
- `MAX_TOP_K = 50`: Maximum search results to retrieve

## JPL Commandments Compliance

##
All methods use early returns and extracted helper functions to minimize nesting.

##
Constants define maximum iterations:
- MAX_EVIDENCE_ITEMS (100)
- MAX_CLAIM_LENGTH (5000)
- MAX_TOP_K (50)

##
All functions stay under 60 lines through decomposition.

##
Input validation with detailed error messages.

##
Complete validation of claims, evidence, and thresholds.

##
All functions have full type annotations.

## Testing

Comprehensive test suite in `tests/unit/enrichment/test_evidence_linker.py`:

```bash
pytest tests/unit/enrichment/test_evidence_linker.py
```

**Test Coverage:**
- 51 test cases
- 100% passing
- Edge cases, validation, JPL compliance

## Performance Considerations

### Search Optimization
- Uses semantic search for relevance ranking
- Entity extraction improves search precision
- Configurable top_k limits computation

### Confidence Scoring
- Support/refute: scales with relevance score
- Neutral: capped at 0.5 confidence
- Helps downstream systems weigh evidence

## Common Patterns

### Fact-Checking Pipeline

```python
# 1. Extract claims from text
claims = extract_claims(text)

# 2. Link evidence for each claim
linker = EvidenceLinker
for claim in claims:
    result = linker.link_evidence(claim, storage)

    # 3. Make verdict based on evidence
    if result.total_support > result.total_refute:
        print(f"✓ Claim likely true: {claim}")
    elif result.total_refute > result.total_support:
        print(f"✗ Claim likely false: {claim}")
    else:
        print(f"? Insufficient evidence: {claim}")
```

### Evidence Quality Filtering

```python
# Filter high-confidence evidence only
high_conf_evidence = [
    e for e in result.linked_evidence
    if e.confidence > 0.7
]

# Group by support type
supporting = [e for e in result.linked_evidence if e.support_type == SupportType.SUPPORTS]
refuting = [e for e in result.linked_evidence if e.support_type == SupportType.REFUTES]
```

## Integration Points

### With ContradictionDetector (P3-AI-002.1)
- Used internally for refutation detection
- Combines similarity + negation/antonym patterns
- Configurable negation boost and thresholds

### With Storage Layer
- Works with any ChunkRepository implementation
- Supports semantic search via `search` method
- Library filtering for scoped searches

### With Retrieval System
- Can augment query results with support classification
- Helps rank evidence by relevance and type
- Provides confidence scores for ranking

## Future Enhancements

### Planned Improvements
- [ ] Neural claim-evidence entailment models
- [ ] Multi-hop evidence chains
- [ ] Temporal evidence weighting
- [ ] Source credibility scoring
- [ ] Uncertainty quantification

### Integration Opportunities
- [ ] Fact-checker agent (P3-AI-002.3)
- [ ] Claim extraction (P3-AI-002.4)
- [ ] Evidence summarization
- [ ] Contradiction visualization

## See Also

- `contradiction.py`: Contradiction detection engine
- `ner.py`: Named entity recognition for entity extraction
- `knowledge_graph.py`: Graph-based evidence relationships
- `storage/base.py`: ChunkRepository interface

## References

- Task: P3-AI-002.2 Evidence Linker
- Related: P3-AI-002.1 Contradiction Detector
- NASA JPL Commandments for safety-critical code
