"""Case Conflict Detector for Legal Vertical (LEGAL-004).

Detects contradictory rulings within the internal corpus using
agentic tools and flags inconsistencies in "Precedent" chunks.
Conflict Types
--------------
- DIRECT: Opposite holdings on same legal issue
- JURISDICTIONAL: Different circuits ruling differently
- TEMPORAL: Overruled or superseded cases
- FACTUAL: Similar facts, different outcomes

Usage Example
-------------
    from ingestforge.agent.case_conflict import CaseConflictDetector
    from tests.fixtures.agents import MockLLM

    llm = MockLLM()
    detector = CaseConflictDetector(
        llm_client=llm,
        storage=mock_storage,
    )

    conflicts = detector.detect_conflicts(["doc_1", "doc_2"])
    for conflict in conflicts:
        print(f"{conflict.case_a_id} vs {conflict.case_b_id}: {conflict.conflict_type}")
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

from ingestforge.core.logging import get_logger
from ingestforge.llm.base import GenerationConfig, LLMClient

logger = get_logger(__name__)
MAX_DOCUMENTS_PER_BATCH = 100
MAX_CHUNKS_PER_DOCUMENT = 500
MAX_CONFLICTS_RETURNED = 200
MAX_EXCERPT_LENGTH = 500
MAX_DESCRIPTION_LENGTH = 1000


class ConflictType(Enum):
    """Types of case conflicts that can be detected."""

    DIRECT = "DIRECT"  # Opposite holdings on same legal issue
    JURISDICTIONAL = "JURISDICTIONAL"  # Different circuits ruling differently
    TEMPORAL = "TEMPORAL"  # Overruled or superseded cases
    FACTUAL = "FACTUAL"  # Similar facts, different outcomes
    UNKNOWN = "UNKNOWN"


@dataclass
class CaseConflict:
    """Detected conflict between two cases.

    Attributes:
        case_a_id: First case identifier
        case_b_id: Second case identifier
        conflict_type: Type of conflict detected
        issue: Legal issue in conflict
        description: Explanation of the conflict
        confidence: Conflict confidence score (0.0-1.0)
        case_a_excerpt: Relevant text from case A
        case_b_excerpt: Relevant text from case B
    """

    case_a_id: str
    case_b_id: str
    conflict_type: str
    issue: str
    description: str
    confidence: float
    case_a_excerpt: str
    case_b_excerpt: str

    def __post_init__(self) -> None:
        """Validate and truncate fields."""
        assert self.case_a_id, "case_a_id cannot be empty"
        assert self.case_b_id, "case_b_id cannot be empty"

        # Clamp confidence to valid range
        self.confidence = max(0.0, min(1.0, self.confidence))

        # Truncate excerpts to max length
        self.case_a_excerpt = self.case_a_excerpt[:MAX_EXCERPT_LENGTH]
        self.case_b_excerpt = self.case_b_excerpt[:MAX_EXCERPT_LENGTH]
        self.description = self.description[:MAX_DESCRIPTION_LENGTH]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "case_a_id": self.case_a_id,
            "case_b_id": self.case_b_id,
            "conflict_type": self.conflict_type,
            "issue": self.issue,
            "description": self.description,
            "confidence": self.confidence,
            "case_a_excerpt": self.case_a_excerpt,
            "case_b_excerpt": self.case_b_excerpt,
        }

    @property
    def is_high_confidence(self) -> bool:
        """Check if conflict has high confidence."""
        return self.confidence >= 0.8

    @property
    def is_direct_conflict(self) -> bool:
        """Check if conflict is a direct contradiction."""
        return self.conflict_type == ConflictType.DIRECT.value


class StorageProtocol(Protocol):
    """Protocol for storage backend used by conflict detector."""

    def get_chunks_by_document(self, document_id: str) -> List[Any]:
        """Get all chunks for a document."""
        ...

    def search_semantic(
        self,
        query_embedding: List[float],
        top_k: int = 10,
        **kwargs: Any,
    ) -> List[Any]:
        """Search using embedding vector."""
        ...


class ConflictPrompts:
    """Prompt templates for conflict detection."""

    @staticmethod
    def compare_holdings_prompt(
        holding_a: str,
        holding_b: str,
        context_a: Optional[str] = None,
        context_b: Optional[str] = None,
    ) -> str:
        """Generate prompt for comparing two holdings.

        Args:
            holding_a: First holding text
            holding_b: Second holding text
            context_a: Optional context for first holding
            context_b: Optional context for second holding

        Returns:
            Formatted prompt string
        """
        context_section = ""
        if context_a:
            context_section += f"\nContext A:\n{context_a[:500]}"
        if context_b:
            context_section += f"\nContext B:\n{context_b[:500]}"

        return f"""Compare these two legal holdings for conflicts or contradictions.

Holding A:
{holding_a[:MAX_EXCERPT_LENGTH]}

Holding B:
{holding_b[:MAX_EXCERPT_LENGTH]}
{context_section}

Analyze whether these holdings conflict. Consider:
1. Do they reach opposite conclusions on the same legal issue?
2. Are they from different jurisdictions ruling differently?
3. Does one overrule or supersede the other?
4. Do they apply different rules to similar facts?

Respond in this exact format:
Conflict: [YES/NO]
Type: [DIRECT/JURISDICTIONAL/TEMPORAL/FACTUAL/NONE]
Issue: [Brief description of the legal issue]
Confidence: [0.0-1.0]
Description: [Explanation of the conflict or why there is none]"""

    @staticmethod
    def check_overruled_prompt(case_text: str, citations: List[str]) -> str:
        """Generate prompt for checking if case is overruled.

        Args:
            case_text: Case holding text
            citations: List of citations found in text

        Returns:
            Formatted prompt string
        """
        citations_text = "\n".join(citations[:10]) if citations else "None found"

        return f"""Analyze whether this case has been overruled or superseded.

Case Holding:
{case_text[:MAX_EXCERPT_LENGTH]}

Citations in Text:
{citations_text}

Look for indicators that this case has been:
- Explicitly overruled by a later case
- Superseded by statute or regulation
- Distinguished to the point of being effectively overruled
- Modified or limited in scope

Respond in this exact format:
Overruled: [YES/NO/PARTIAL]
Overruling_Authority: [Citation or "None"]
Confidence: [0.0-1.0]
Reasoning: [Brief explanation]"""

    @staticmethod
    def analyze_jurisdictional_split_prompt(
        issue: str,
        holdings: List[Dict[str, str]],
    ) -> str:
        """Generate prompt for analyzing jurisdictional splits.

        Args:
            issue: Legal issue to analyze
            holdings: List of dicts with 'jurisdiction' and 'holding' keys

        Returns:
            Formatted prompt string
        """
        holdings_text = ""
        for i, h in enumerate(holdings[:10]):
            jurisdiction = h.get("jurisdiction", "Unknown")
            holding = h.get("holding", "")[:200]
            holdings_text += f"\n{i+1}. [{jurisdiction}]: {holding}"

        return f"""Analyze these holdings from different jurisdictions for splits.

Legal Issue:
{issue}

Holdings by Jurisdiction:
{holdings_text}

Identify any circuit splits or jurisdictional conflicts:
1. Which jurisdictions agree with each other?
2. Which jurisdictions disagree?
3. What are the key points of disagreement?

Respond in this exact format:
Split_Detected: [YES/NO]
Agreeing_Jurisdictions: [List of jurisdictions that agree]
Disagreeing_Jurisdictions: [List of jurisdictions that disagree]
Key_Disagreement: [Brief description of the split]
Confidence: [0.0-1.0]"""


class CaseConflictDetector:
    """Detects conflicting rulings within a legal corpus.

    Uses LLM prompts and semantic search to identify:
    - Direct contradictions (opposite holdings on same issue)
    - Jurisdictional conflicts (different circuits ruling differently)
    - Temporal conflicts (overruled or superseded cases)
    - Factual distinctions (similar facts, different outcomes)
    """

    def __init__(
        self,
        llm_client: LLMClient,
        storage: Optional[StorageProtocol] = None,
        contradiction_detector: Optional[Any] = None,
        bluebook_parser: Optional[Any] = None,
        legal_classifier: Optional[Any] = None,
        config: Optional[GenerationConfig] = None,
    ) -> None:
        """Initialize the case conflict detector.

        Args:
            llm_client: LLM client for conflict analysis
            storage: Storage backend for chunk retrieval
            contradiction_detector: Optional ContradictionDetector instance
            bluebook_parser: Optional BluebookParser instance
            legal_classifier: Optional LegalClassifier instance
            config: Optional generation configuration

        Raises:
            ValueError: If llm_client is None
        """
        if llm_client is None:
            raise ValueError("llm_client cannot be None")

        self._llm = llm_client
        self._storage = storage
        self._contradiction_detector = contradiction_detector
        self._bluebook_parser = bluebook_parser
        self._legal_classifier = legal_classifier
        self._config = config or GenerationConfig(
            max_tokens=800,
            temperature=0.2,  # Low for consistent analysis
        )

    def detect_conflicts(
        self,
        document_ids: List[str],
    ) -> List[CaseConflict]:
        """Detect conflicts across multiple documents.

        Args:
            document_ids: List of document IDs to analyze

        Returns:
            List of detected CaseConflict objects
        """
        if not document_ids:
            return []
        document_ids = document_ids[:MAX_DOCUMENTS_PER_BATCH]

        # Collect holdings from all documents
        holdings = self._extract_holdings_from_documents(document_ids)

        if len(holdings) < 2:
            return []

        # Compare holdings pairwise
        conflicts = self._compare_holdings_pairwise(holdings)

        return conflicts[:MAX_CONFLICTS_RETURNED]

    def find_contradictory_holdings(
        self,
        chunk: Dict[str, Any],
    ) -> List[CaseConflict]:
        """Find holdings that contradict a given chunk.

        Args:
            chunk: Chunk dictionary with 'text' and 'document_id' keys

        Returns:
            List of CaseConflict objects
        """
        if not chunk or "text" not in chunk:
            return []

        text = chunk.get("text", "")
        document_id = chunk.get("document_id", "unknown")

        if not text.strip():
            return []

        # Classify the chunk to confirm it's a holding
        if self._legal_classifier:
            classification = self._classify_chunk(chunk)
            if classification.get("role") != "HOLDING":
                return []

        # Search for related holdings in storage
        related_holdings = self._find_related_holdings(text)

        # Compare with each related holding
        conflicts: List[CaseConflict] = []
        for holding in related_holdings[:20]:
            holding_doc_id = holding.get("document_id", "")

            # Skip self-comparison
            if holding_doc_id == document_id:
                continue

            conflict = self._analyze_holding_conflict(
                holding_a=text,
                doc_a_id=document_id,
                holding_b=holding.get("text", ""),
                doc_b_id=holding_doc_id,
            )

            if conflict:
                conflicts.append(conflict)

        return conflicts[:MAX_CONFLICTS_RETURNED]

    def check_overruled(self, case_id: str) -> Optional[str]:
        """Check if a case has been overruled.

        Args:
            case_id: Case document ID to check

        Returns:
            Overruling authority citation if overruled, None otherwise
        """
        if not case_id or not case_id.strip():
            return None

        # Get case chunks
        chunks = self._get_document_chunks(case_id)
        if not chunks:
            return None

        # Extract holding text
        holding_text = self._extract_holding_text(chunks)
        if not holding_text:
            return None

        # Extract citations using BluebookParser
        citations = self._extract_citations(holding_text)

        # Ask LLM to analyze overruled status
        return self._analyze_overruled_status(holding_text, citations)

    def analyze_jurisdictional_split(
        self,
        issue: str,
    ) -> List[CaseConflict]:
        """Analyze holdings for jurisdictional splits on an issue.

        Args:
            issue: Legal issue description to analyze

        Returns:
            List of CaseConflict objects representing splits
        """
        if not issue or not issue.strip():
            return []

        # Search for holdings related to the issue
        related_holdings = self._search_holdings_by_issue(issue)

        if len(related_holdings) < 2:
            return []

        # Group by jurisdiction
        jurisdiction_groups = self._group_by_jurisdiction(related_holdings)

        if len(jurisdiction_groups) < 2:
            return []

        # Analyze for splits
        return self._detect_jurisdiction_splits(issue, jurisdiction_groups)

    def _extract_holdings_from_documents(
        self,
        document_ids: List[str],
    ) -> List[Dict[str, Any]]:
        """Extract holding chunks from documents.

        Args:
            document_ids: List of document IDs

        Returns:
            List of holding dictionaries with text and metadata
        """
        holdings: List[Dict[str, Any]] = []

        for doc_id in document_ids:
            chunks = self._get_document_chunks(doc_id)
            doc_holdings = self._filter_holding_chunks(chunks, doc_id)
            holdings.extend(doc_holdings)
            if len(holdings) >= MAX_CHUNKS_PER_DOCUMENT:
                break

        return holdings

    def _filter_holding_chunks(
        self,
        chunks: List[Any],
        document_id: str,
    ) -> List[Dict[str, Any]]:
        """Filter chunks to only include holdings.

        Args:
            chunks: List of chunk objects
            document_id: Parent document ID

        Returns:
            List of holding dictionaries
        """
        holdings: List[Dict[str, Any]] = []

        for chunk in chunks[:MAX_CHUNKS_PER_DOCUMENT]:
            chunk_dict = self._chunk_to_dict(chunk)

            # Check if chunk is classified as a holding
            if self._is_holding_chunk(chunk_dict):
                chunk_dict["document_id"] = document_id
                holdings.append(chunk_dict)

        return holdings

    def _is_holding_chunk(self, chunk: Dict[str, Any]) -> bool:
        """Determine if chunk contains a legal holding.

        Args:
            chunk: Chunk dictionary

        Returns:
            True if chunk is a holding
        """
        # Check metadata for legal_role if available
        legal_role = chunk.get("legal_role")
        if legal_role == "HOLDING":
            return True

        # Check chunk_type for precedent indicators
        chunk_type = chunk.get("chunk_type", "")
        if "holding" in chunk_type.lower() or "precedent" in chunk_type.lower():
            return True

        # Use classifier if available
        if self._legal_classifier:
            classification = self._classify_chunk(chunk)
            return classification.get("role") == "HOLDING"

        # Default heuristic: look for holding language
        text = chunk.get("text", "").lower()
        holding_indicators = ["we hold", "the court held", "holding:", "we conclude"]
        return any(ind in text for ind in holding_indicators)

    def _compare_holdings_pairwise(
        self,
        holdings: List[Dict[str, Any]],
    ) -> List[CaseConflict]:
        """Compare holdings pairwise for conflicts.

        Args:
            holdings: List of holding dictionaries

        Returns:
            List of detected conflicts
        """
        conflicts: List[CaseConflict] = []
        num_holdings = len(holdings)
        max_comparisons = min(num_holdings * (num_holdings - 1) // 2, 500)
        comparison_count = 0

        for i in range(num_holdings):
            for j in range(i + 1, num_holdings):
                if comparison_count >= max_comparisons:
                    break

                conflict = self._analyze_holding_conflict(
                    holding_a=holdings[i].get("text", ""),
                    doc_a_id=holdings[i].get("document_id", f"doc_{i}"),
                    holding_b=holdings[j].get("text", ""),
                    doc_b_id=holdings[j].get("document_id", f"doc_{j}"),
                )

                if conflict:
                    conflicts.append(conflict)

                comparison_count += 1

            if comparison_count >= max_comparisons:
                break

        return conflicts

    def _analyze_holding_conflict(
        self,
        holding_a: str,
        doc_a_id: str,
        holding_b: str,
        doc_b_id: str,
    ) -> Optional[CaseConflict]:
        """Analyze two holdings for conflicts.

        Args:
            holding_a: First holding text
            doc_a_id: First document ID
            holding_b: Second holding text
            doc_b_id: Second document ID

        Returns:
            CaseConflict if conflict detected, None otherwise
        """
        # Use ContradictionDetector if available for initial screening
        if self._contradiction_detector:
            result = self._contradiction_detector.detect_contradiction(
                holding_a, holding_b
            )
            if result.score < 0.3:
                return None  # Not similar enough to be in conflict

        # Use LLM for detailed analysis
        prompt = ConflictPrompts.compare_holdings_prompt(holding_a, holding_b)

        try:
            response = self._llm.generate(prompt, self._config)
        except Exception as e:
            logger.error(f"LLM conflict analysis failed: {e}")
            return None

        return self._parse_conflict_response(
            response=response,
            doc_a_id=doc_a_id,
            doc_b_id=doc_b_id,
            excerpt_a=holding_a,
            excerpt_b=holding_b,
        )

    def _parse_conflict_response(
        self,
        response: str,
        doc_a_id: str,
        doc_b_id: str,
        excerpt_a: str,
        excerpt_b: str,
    ) -> Optional[CaseConflict]:
        """Parse LLM response into CaseConflict.

        Args:
            response: LLM response text
            doc_a_id: First document ID
            doc_b_id: Second document ID
            excerpt_a: First excerpt
            excerpt_b: Second excerpt

        Returns:
            CaseConflict if conflict detected, None otherwise
        """
        if not response or not response.strip():
            return None

        # Check if conflict was detected
        conflict_match = re.search(r"Conflict:\s*(YES|NO)", response, re.IGNORECASE)
        if not conflict_match or conflict_match.group(1).upper() == "NO":
            return None

        # Extract fields
        conflict_type = self._extract_conflict_type(response)
        issue = self._extract_field(response, "Issue")
        confidence = self._extract_confidence(response)
        description = self._extract_field(response, "Description")

        return CaseConflict(
            case_a_id=doc_a_id,
            case_b_id=doc_b_id,
            conflict_type=conflict_type,
            issue=issue or "Unknown issue",
            description=description or "Conflict detected",
            confidence=confidence,
            case_a_excerpt=excerpt_a,
            case_b_excerpt=excerpt_b,
        )

    def _extract_conflict_type(self, response: str) -> str:
        """Extract conflict type from response.

        Args:
            response: LLM response text

        Returns:
            Conflict type string
        """
        match = re.search(
            r"Type:\s*(\w+)",
            response,
            re.IGNORECASE,
        )
        if match:
            type_str = match.group(1).upper()
            if type_str in [ct.value for ct in ConflictType]:
                return type_str

        return ConflictType.UNKNOWN.value

    def _extract_field(self, response: str, field_name: str) -> str:
        """Extract a field value from response.

        Args:
            response: LLM response text
            field_name: Name of field to extract

        Returns:
            Field value or empty string
        """
        match = re.search(
            rf"{field_name}:\s*(.+?)(?:\n|$)",
            response,
            re.IGNORECASE | re.DOTALL,
        )
        return match.group(1).strip() if match else ""

    def _extract_confidence(self, response: str) -> float:
        """Extract confidence score from response.

        Args:
            response: LLM response text

        Returns:
            Confidence value (0.0-1.0)
        """
        match = re.search(
            r"Confidence:\s*([\d.]+)",
            response,
            re.IGNORECASE,
        )
        if match:
            try:
                return float(match.group(1))
            except ValueError as e:
                logger.debug(f"Failed to parse confidence score: {e}")

        return 0.5  # Default confidence

    def _get_document_chunks(self, document_id: str) -> List[Any]:
        """Get chunks for a document from storage.

        Args:
            document_id: Document ID

        Returns:
            List of chunks
        """
        if not self._storage:
            return []

        try:
            return self._storage.get_chunks_by_document(document_id)
        except Exception as e:
            logger.warning(f"Failed to get chunks for {document_id}: {e}")
            return []

    def _chunk_to_dict(self, chunk: Any) -> Dict[str, Any]:
        """Convert chunk to dictionary.

        Args:
            chunk: Chunk object

        Returns:
            Chunk as dictionary
        """
        if isinstance(chunk, dict):
            return chunk

        if hasattr(chunk, "to_dict"):
            return chunk.to_dict()

        if hasattr(chunk, "content"):
            return {
                "text": getattr(chunk, "content", ""),
                "document_id": getattr(chunk, "document_id", ""),
                "chunk_type": getattr(chunk, "chunk_type", ""),
                "legal_role": getattr(chunk, "legal_role", None),
            }

        return {"text": str(chunk)}

    def _classify_chunk(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Classify a chunk using LegalClassifier.

        Args:
            chunk: Chunk dictionary

        Returns:
            Classification result dictionary
        """
        if not self._legal_classifier:
            return {}

        try:
            result = self._legal_classifier.classify(chunk.get("text", ""))
            return {"role": result.role, "confidence": result.confidence}
        except Exception as e:
            logger.debug(f"Classification failed: {e}")
            return {}

    def _find_related_holdings(self, text: str) -> List[Dict[str, Any]]:
        """Find holdings related to the given text.

        Args:
            text: Text to find related holdings for

        Returns:
            List of related holding dictionaries
        """
        if not self._storage:
            return []

        # Would use semantic search in production
        # For now, return empty list
        return []

    def _extract_holding_text(self, chunks: List[Any]) -> str:
        """Extract holding text from chunks.

        Args:
            chunks: List of chunks

        Returns:
            Concatenated holding text
        """
        holding_texts = []

        for chunk in chunks[:50]:
            chunk_dict = self._chunk_to_dict(chunk)
            if self._is_holding_chunk(chunk_dict):
                holding_texts.append(chunk_dict.get("text", ""))

        return " ".join(holding_texts)[: MAX_EXCERPT_LENGTH * 4]

    def _extract_citations(self, text: str) -> List[str]:
        """Extract legal citations from text.

        Args:
            text: Text to extract citations from

        Returns:
            List of citation strings
        """
        if self._bluebook_parser:
            try:
                citations = self._bluebook_parser.extract_citations(text)
                return [c.raw_text for c in citations]
            except Exception as e:
                logger.debug(f"Citation extraction failed: {e}")

        return []

    def _analyze_overruled_status(
        self,
        holding_text: str,
        citations: List[str],
    ) -> Optional[str]:
        """Analyze if case is overruled using LLM.

        Args:
            holding_text: Case holding text
            citations: List of citations

        Returns:
            Overruling authority if overruled, None otherwise
        """
        prompt = ConflictPrompts.check_overruled_prompt(holding_text, citations)

        try:
            response = self._llm.generate(prompt, self._config)
        except Exception as e:
            logger.error(f"Overruled analysis failed: {e}")
            return None

        # Parse response
        overruled_match = re.search(
            r"Overruled:\s*(YES|NO|PARTIAL)",
            response,
            re.IGNORECASE,
        )
        if not overruled_match or overruled_match.group(1).upper() == "NO":
            return None

        authority_match = re.search(
            r"Overruling_Authority:\s*(.+?)(?:\n|$)",
            response,
            re.IGNORECASE,
        )
        if authority_match:
            authority = authority_match.group(1).strip()
            if authority.lower() != "none":
                return authority

        return "Unknown authority"

    def _search_holdings_by_issue(self, issue: str) -> List[Dict[str, Any]]:
        """Search for holdings related to a legal issue.

        Args:
            issue: Legal issue to search for

        Returns:
            List of holding dictionaries
        """
        # Would use semantic search in production
        return []

    def _group_by_jurisdiction(
        self,
        holdings: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group holdings by jurisdiction.

        Args:
            holdings: List of holding dictionaries

        Returns:
            Dictionary mapping jurisdiction to holdings
        """
        groups: Dict[str, List[Dict[str, Any]]] = {}

        for holding in holdings:
            jurisdiction = holding.get("jurisdiction", "Unknown")
            if jurisdiction not in groups:
                groups[jurisdiction] = []
            groups[jurisdiction].append(holding)

        return groups

    def _detect_jurisdiction_splits(
        self,
        issue: str,
        jurisdiction_groups: Dict[str, List[Dict[str, Any]]],
    ) -> List[CaseConflict]:
        """Detect jurisdictional splits in holdings.

        Args:
            issue: Legal issue being analyzed
            jurisdiction_groups: Holdings grouped by jurisdiction

        Returns:
            List of CaseConflict objects
        """
        # Format holdings for LLM analysis
        holdings_for_prompt = []
        for jurisdiction, holdings in jurisdiction_groups.items():
            for holding in holdings[:3]:  # Max 3 per jurisdiction
                holdings_for_prompt.append(
                    {
                        "jurisdiction": jurisdiction,
                        "holding": holding.get("text", ""),
                    }
                )

        prompt = ConflictPrompts.analyze_jurisdictional_split_prompt(
            issue, holdings_for_prompt
        )

        try:
            response = self._llm.generate(prompt, self._config)
        except Exception as e:
            logger.error(f"Jurisdictional split analysis failed: {e}")
            return []

        # Parse response
        split_match = re.search(r"Split_Detected:\s*(YES|NO)", response, re.IGNORECASE)
        if not split_match or split_match.group(1).upper() == "NO":
            return []

        # Extract disagreement details
        disagreement = self._extract_field(response, "Key_Disagreement")
        confidence = self._extract_confidence(response)

        # Create conflict for each pair of disagreeing jurisdictions
        conflicts: List[CaseConflict] = []
        jurisdictions = list(jurisdiction_groups.keys())

        for i in range(len(jurisdictions)):
            for j in range(i + 1, len(jurisdictions)):
                if len(conflicts) >= MAX_CONFLICTS_RETURNED:
                    break

                jur_a = jurisdictions[i]
                jur_b = jurisdictions[j]

                # Get representative cases
                cases_a = jurisdiction_groups[jur_a]
                cases_b = jurisdiction_groups[jur_b]

                if cases_a and cases_b:
                    conflicts.append(
                        CaseConflict(
                            case_a_id=cases_a[0].get("document_id", f"case_{jur_a}"),
                            case_b_id=cases_b[0].get("document_id", f"case_{jur_b}"),
                            conflict_type=ConflictType.JURISDICTIONAL.value,
                            issue=issue,
                            description=f"Split between {jur_a} and {jur_b}: {disagreement}",
                            confidence=confidence,
                            case_a_excerpt=cases_a[0].get("text", "")[
                                :MAX_EXCERPT_LENGTH
                            ],
                            case_b_excerpt=cases_b[0].get("text", "")[
                                :MAX_EXCERPT_LENGTH
                            ],
                        )
                    )

        return conflicts


def create_case_conflict_detector(
    llm_client: LLMClient,
    storage: Optional[StorageProtocol] = None,
    contradiction_detector: Optional[Any] = None,
    bluebook_parser: Optional[Any] = None,
    legal_classifier: Optional[Any] = None,
    config: Optional[GenerationConfig] = None,
) -> CaseConflictDetector:
    """Factory function to create case conflict detector.

    Args:
        llm_client: LLM client for conflict analysis
        storage: Optional storage backend
        contradiction_detector: Optional ContradictionDetector
        bluebook_parser: Optional BluebookParser
        legal_classifier: Optional LegalClassifier
        config: Optional generation configuration

    Returns:
        Configured CaseConflictDetector
    """
    return CaseConflictDetector(
        llm_client=llm_client,
        storage=storage,
        contradiction_detector=contradiction_detector,
        bluebook_parser=bluebook_parser,
        legal_classifier=legal_classifier,
        config=config,
    )
