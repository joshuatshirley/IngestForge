# Accuracy Framework

> For a research assistant, a wrong answer is worse than no answer.

## Core Principle

**Fail loudly, never fail silently.**

If IngestForge cannot provide accurate, verifiable information, it must:
1. Say "I don't know" or "I'm not confident"
2. Explain why (insufficient sources, conflicting information, low confidence)
3. Suggest what the user can do (add more sources, rephrase query, verify manually)

**Never**:
- Guess when uncertain
- Extrapolate beyond source material
- Present low-confidence results as facts
- Attribute quotes to wrong sources
- Generate citations that don't exist

---

## Accuracy Dimensions

### 1. Source Attribution Accuracy

**Problem**: A quote attributed to the wrong author undermines the entire paper.

**Requirements**:
| Requirement | Implementation |
|-------------|----------------|
| Every claim traceable | `source_location` on every chunk, always |
| No orphan text | Reject chunks without valid provenance |
| Citation verification | Validate citation format matches source metadata |
| Quote exactness | Store original text, detect paraphrasing |

**Verification**:
```python
@dataclass
class VerifiedQuote:
    text: str
    source_location: SourceLocation
    match_type: str  # "exact", "near_exact", "paraphrase"
    confidence: float  # 0.0-1.0
    verification_method: str  # "hash_match", "fuzzy_match", "semantic"

    def is_citable(self) -> bool:
        """Only exact matches are safe to quote."""
        return self.match_type == "exact" and self.confidence >= 0.99
```

**User-Facing**:
```
Query: "What did Smith say about inflation?"

✓ Exact quote found:
  "Inflation erodes purchasing power over time."
  — Smith (2023), Chapter 3, p. 47

⚠ Paraphrase detected (not safe to quote directly):
  Smith discusses how inflation affects buying power.
  [View original context to extract exact quote]

✗ Cannot verify attribution:
  Found similar content but cannot confirm Smith as author.
  Sources: Document A (no author metadata), Document B (author: Jones)
```

---

### 2. Retrieval Accuracy

**Problem**: Semantic search can return plausible-sounding but irrelevant results.

**Requirements**:
| Requirement | Implementation |
|-------------|----------------|
| Relevance threshold | Don't return results below confidence threshold |
| Empty > wrong | Return empty results rather than low-relevance matches |
| Explain gaps | "No results found" with actionable suggestions |
| Multi-signal validation | Require BM25 + semantic agreement for high confidence |

**Confidence Scoring**:
```python
@dataclass
class RetrievalResult:
    chunk: Chunk
    semantic_score: float
    bm25_score: float
    rerank_score: Optional[float]

    @property
    def confidence(self) -> str:
        """High/Medium/Low/Insufficient based on score agreement."""
        if self.semantic_score > 0.8 and self.bm25_score > 0.5:
            return "high"
        elif self.semantic_score > 0.6 and self.bm25_score > 0.3:
            return "medium"
        elif self.semantic_score > 0.4 or self.bm25_score > 0.2:
            return "low"
        return "insufficient"

    def should_display(self, min_confidence: str = "medium") -> bool:
        """Filter out low-confidence results by default."""
        levels = ["insufficient", "low", "medium", "high"]
        return levels.index(self.confidence) >= levels.index(min_confidence)
```

**User-Facing**:
```
Query: "quantum entanglement applications"

Found 3 high-confidence results, 2 medium-confidence results.
[Showing high-confidence only. Use --include-low for all results.]

1. [HIGH] "Quantum entanglement enables..." — Physics Today (2023)
2. [HIGH] "Applications include quantum computing..." — Nature (2022)
3. [HIGH] "Entanglement-based cryptography..." — IEEE (2023)

---

Query: "underwater basket weaving economics"

No high or medium confidence results found.

Suggestions:
- Your corpus may not contain information on this topic
- Try broader terms: "craft economics", "artisan markets"
- Add sources about this specific topic
```

---

### 3. LLM Answer Accuracy

**Problem**: LLMs can hallucinate, especially when sources are thin.

**Requirements**:
| Requirement | Implementation |
|-------------|----------------|
| Grounded answers only | Every LLM claim must cite a retrieved chunk |
| No extrapolation | If sources don't cover it, say so |
| Confidence indicators | Show how well-supported each claim is |
| Source verification | Post-process to verify citations exist |

**Grounding Enforcement**:
```python
class GroundedAnswerGenerator:
    def generate(self, query: str, chunks: list[Chunk]) -> GroundedAnswer:
        # Generate answer with inline citations
        raw_answer = self.llm.generate(
            prompt=GROUNDED_PROMPT,
            context=chunks
        )

        # Verify every citation
        verified = self.verify_citations(raw_answer, chunks)

        # Remove unverifiable claims
        cleaned = self.remove_ungrounded_claims(verified)

        # If nothing left, return "insufficient sources"
        if not cleaned.has_content:
            return GroundedAnswer(
                text="I cannot answer this question based on your sources.",
                reason="insufficient_coverage",
                suggestions=self.suggest_sources(query)
            )

        return cleaned
```

**Grounding Prompt**:
```
You are a research assistant. Answer ONLY based on the provided sources.

RULES:
1. Every factual claim MUST have a citation [1], [2], etc.
2. If sources don't cover something, say "The sources don't address this."
3. Never guess, extrapolate, or use outside knowledge.
4. If sources conflict, present both views with citations.
5. Distinguish between direct quotes and paraphrases.

If you cannot answer from the sources, respond:
"I cannot answer this based on your current sources. Consider adding materials about [specific topic]."
```

**User-Facing**:
```
Query: "What causes inflation according to my sources?"

Based on your sources:

Demand-pull inflation occurs when "aggregate demand exceeds
aggregate supply" [1]. Smith (2023) argues this is the primary
driver in developed economies [1, p. 47].

Cost-push inflation, caused by rising production costs, is
discussed by Jones (2022) who notes that "supply chain
disruptions amplify inflationary pressure" [2, p. 12].

⚠ Your sources do not cover:
- Monetary policy effects on inflation
- Historical inflation data

Consider adding sources on these topics for a complete picture.

---

Sources:
[1] Smith, J. (2023). Macroeconomic Principles. Ch. 3, pp. 45-52.
[2] Jones, M. (2022). Supply Chain Economics. Ch. 1, pp. 10-15.
```

---

### 4. Study Material Accuracy

**Problem**: Flashcards, quizzes, and glossaries can propagate errors at scale.

**Requirements**:
| Requirement | Implementation |
|-------------|----------------|
| Source verification | Every flashcard links to source chunk |
| Definition validation | Definitions must come from "X is Y" patterns, not inference |
| Quiz answer verification | Correct answers must be verbatim from sources |
| Distractor safety | Wrong answers must be clearly wrong, not misleading |

**Flashcard Verification**:
```python
@dataclass
class VerifiedFlashcard:
    term: str
    definition: str
    source_chunk_id: str
    extraction_method: str  # "pattern_match", "llm_extraction"
    confidence: float

    def is_publishable(self) -> bool:
        """Only high-confidence, pattern-matched cards are safe."""
        return (
            self.extraction_method == "pattern_match" and
            self.confidence >= 0.95 and
            self.source_chunk_id is not None
        )

class FlashcardGenerator:
    def generate(self, chunks: list[Chunk]) -> FlashcardSet:
        cards = []
        rejected = []

        for chunk in chunks:
            extracted = self.extract_definitions(chunk)
            for card in extracted:
                if card.is_publishable:
                    cards.append(card)
                else:
                    rejected.append((card, "low_confidence"))

        return FlashcardSet(
            cards=cards,
            rejected_count=len(rejected),
            message=f"Generated {len(cards)} verified flashcards. "
                    f"{len(rejected)} potential cards rejected for quality."
        )
```

**Quiz Answer Verification**:
```python
@dataclass
class QuizQuestion:
    question: str
    correct_answer: str
    correct_answer_source: SourceLocation  # REQUIRED
    distractors: list[str]

    def verify(self, chunks: list[Chunk]) -> bool:
        """Verify correct answer exists verbatim in sources."""
        for chunk in chunks:
            if self.correct_answer in chunk.content:
                return True
        return False
```

---

### 5. Conflict Detection Accuracy

**Problem**: Sources may contradict each other. Presenting one view as fact is misleading.

**Requirements**:
| Requirement | Implementation |
|-------------|----------------|
| Conflict detection | Identify when sources disagree |
| Balanced presentation | Show all sides with citations |
| Recency awareness | Note if newer sources supersede older |
| Authority weighting | Note source credibility differences |

**Conflict Handling**:
```python
@dataclass
class SourceConflict:
    topic: str
    positions: list[ConflictPosition]
    resolution_status: str  # "unresolved", "newer_supersedes", "domain_specific"

@dataclass
class ConflictPosition:
    claim: str
    sources: list[SourceLocation]
    source_dates: list[date]

class ConflictDetector:
    def analyze(self, query: str, chunks: list[Chunk]) -> ConflictAnalysis:
        # Find chunks making claims about same topic
        # Detect semantic opposition
        # Return structured conflict report
```

**User-Facing**:
```
Query: "What is the optimal tax rate?"

⚠ Your sources contain conflicting views on this topic:

POSITION A: "Optimal tax rates should be progressive..."
— Smith (2023), Jones (2022)

POSITION B: "Flat tax rates maximize efficiency..."
— Williams (2021), Brown (2020)

NOTE: The more recent sources (Smith 2023, Jones 2022) favor
Position A, but this remains an active debate in the field.

I cannot provide a single answer. Consider:
- Presenting both views in your paper
- Adding more sources to understand the debate
- Framing your thesis around one position explicitly
```

---

### 6. OCR and Extraction Accuracy

**Problem**: OCR errors propagate through the entire pipeline.

**Requirements**:
| Requirement | Implementation |
|-------------|----------------|
| Confidence thresholds | Flag low-confidence OCR pages |
| Human review queue | Queue uncertain extractions for review |
| Original preservation | Keep original alongside extracted text |
| Error indicators | Mark content that may have errors |

**OCR Quality Tracking**:
```python
@dataclass
class OCRResult:
    text: str
    confidence: float  # Average character confidence
    low_confidence_regions: list[tuple[int, int]]  # Character ranges

    def is_reliable(self) -> bool:
        return self.confidence >= 0.90

    def get_warnings(self) -> list[str]:
        warnings = []
        if self.confidence < 0.90:
            warnings.append(f"OCR confidence {self.confidence:.0%} - verify quotes manually")
        if self.low_confidence_regions:
            warnings.append(f"{len(self.low_confidence_regions)} uncertain regions detected")
        return warnings

@dataclass
class Chunk:
    content: str
    source_location: SourceLocation
    extraction_confidence: float  # NEW
    extraction_warnings: list[str]  # NEW
    needs_human_review: bool  # NEW
```

**User-Facing**:
```
Query: "What does the 1923 document say about..."

Result from: Historical_Document_1923.pdf (scanned)

⚠ OCR CONFIDENCE: 78%
This content was extracted from a scanned document with moderate
confidence. Verify quotes against the original before citing.

"The committee re[?]ommends that all members sub[?]it
their reports by the end of the fi[?]cal year."

[?] = uncertain characters

[View original PDF page] [Mark as verified] [Flag for review]
```

---

## Accuracy Modes

### Default Mode: Strict
- Only high-confidence results shown
- LLM answers require source grounding
- Conflicts explicitly flagged
- "I don't know" preferred over guessing

### Research Mode: Exploratory
```bash
ingestforge ask --mode exploratory
```
- Medium-confidence results included (marked as such)
- Broader semantic matching
- Still no hallucination, but more suggestions
- Useful for discovering what's in corpus

### Verification Mode: Audit
```bash
ingestforge verify --document paper.docx
```
- Check every citation in a document against corpus
- Flag citations that can't be verified
- Identify potential misquotes
- Report coverage gaps

---

## Accuracy Metrics Dashboard

```bash
ingestforge accuracy-report
```

```
=== Corpus Accuracy Report ===

Source Quality:
  Documents with full metadata: 45/50 (90%)
  Documents with author info:   42/50 (84%)
  Average OCR confidence:       94%
  Documents needing review:     3

Retrieval Quality (last 100 queries):
  High-confidence results:      78%
  Medium-confidence results:    18%
  No results (honest "I don't know"): 4%

Answer Quality (last 50 LLM answers):
  Fully grounded answers:       92%
  Partial coverage disclosed:   6%
  "Insufficient sources" responses: 2%

Study Materials:
  Flashcards verified:          234/250 (94%)
  Quiz answers verified:        89/89 (100%)
  Glossary terms with sources:  156/156 (100%)

Recommendations:
  ⚠ 3 documents have low OCR confidence - consider re-scanning
  ⚠ 5 documents missing author metadata - add manually
  ✓ No unverifiable citations detected
```

---

## Implementation Priority

| Priority | Feature | Why |
|----------|---------|-----|
| 1 | **Grounded LLM answers** | Prevents hallucination in Q&A |
| 2 | **Citation verification** | Every quote traceable to source |
| 3 | **Audit trails** | Deep links to source files (see [AUDIT_TRAIL_DESIGN.md](AUDIT_TRAIL_DESIGN.md)) |
| 4 | **Confidence thresholds** | No silent low-quality results |
| 5 | **Conflict detection** | Don't present disputed facts as truth |
| 6 | **OCR confidence tracking** | Know when to trust extracted text |
| 7 | **Study material verification** | Flashcards/quizzes must be accurate |
| 8 | **Accuracy dashboard** | Monitor quality over time |
| 9 | **Verification mode** | Audit documents before submission |

---

## The "I Don't Know" Response

IngestForge should say "I don't know" when:

1. **No relevant sources**: Query topic not in corpus
2. **Low confidence matches**: Semantic similarity too low
3. **Conflicting sources**: Can't determine truth
4. **Insufficient coverage**: Sources mention topic but don't answer question
5. **OCR uncertainty**: Extracted text too unreliable
6. **Missing metadata**: Can't provide proper citation

**Template**:
```
I cannot provide a reliable answer for this query.

Reason: [specific reason]

What you can do:
- [actionable suggestion 1]
- [actionable suggestion 2]

If you need an answer now, here's what I found with LOW confidence:
[optional: show low-confidence results clearly marked]
```

---

*Accuracy is not a feature. It's the foundation.*

*Last updated: 2026-02-07*
