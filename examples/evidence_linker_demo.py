"""Evidence Linker Demo.

Demonstrates how to use the EvidenceLinker to find supporting/refuting
evidence for claims from a knowledge base.

This is part of the P3-AI-002.2 fact-checking system.
"""

from ingestforge.enrichment import EvidenceLinker
from ingestforge.storage.jsonl import JSONLRepository
from ingestforge.chunking.semantic_chunker import ChunkRecord


def demo_evidence_linking():
    """Demonstrate evidence linking functionality."""
    print("Evidence Linker Demo")
    print("=" * 60)

    # Create a mock knowledge base with some chunks
    storage = JSONLRepository(".ingestforge/demo_evidence.jsonl")

    # Add some sample chunks to the knowledge base
    chunks = [
        ChunkRecord(
            chunk_id="chunk1",
            document_id="doc1",
            content="The Earth orbits around the Sun in approximately 365.25 days.",
            section_title="Solar System",
            source_file="astronomy.txt",
            word_count=10,
        ),
        ChunkRecord(
            chunk_id="chunk2",
            document_id="doc1",
            content="The Moon orbits around the Earth, not the Sun directly.",
            section_title="Moon Facts",
            source_file="astronomy.txt",
            word_count=10,
        ),
        ChunkRecord(
            chunk_id="chunk3",
            document_id="doc2",
            content="Water freezes at 0 degrees Celsius at standard atmospheric pressure.",
            section_title="Water Properties",
            source_file="chemistry.txt",
            word_count=10,
        ),
        ChunkRecord(
            chunk_id="chunk4",
            document_id="doc2",
            content="Ice melts at temperatures above 0 degrees Celsius.",
            section_title="Phase Changes",
            source_file="chemistry.txt",
            word_count=8,
        ),
    ]

    # Add chunks to storage
    for chunk in chunks:
        storage.add_chunk(chunk)

    print(f"\nAdded {len(chunks)} chunks to knowledge base\n")

    # Create evidence linker with reasonable thresholds
    linker = EvidenceLinker(
        support_threshold=0.6,  # Require 60% similarity for support
        refute_threshold=0.2,  # Lower threshold for contradiction detection
    )

    # Test Case 1: Claim with supporting evidence
    print("\nTest Case 1: Claim with Supporting Evidence")
    print("-" * 60)
    claim1 = "The Earth revolves around the Sun"
    result1 = linker.link_evidence(claim1, storage, top_k=5)

    print(f"Claim: {claim1}")
    print(f"Found {len(result1.linked_evidence)} pieces of evidence")
    print(
        f"Support: {result1.total_support}, Refute: {result1.total_refute}, Neutral: {result1.total_neutral}"
    )

    for i, evidence in enumerate(result1.linked_evidence[:3], 1):
        print(f"\n  Evidence {i}:")
        print(f"    Text: {evidence.evidence_text[:80]}...")
        print(f"    Type: {evidence.support_type.value}")
        print(f"    Relevance: {evidence.relevance_score:.2f}")
        print(f"    Confidence: {evidence.confidence:.2f}")

    # Test Case 2: Claim with refuting evidence
    print("\n\nTest Case 2: Claim with Refuting Evidence")
    print("-" * 60)
    claim2 = "The Moon orbits around the Sun directly"
    result2 = linker.link_evidence(claim2, storage, top_k=5)

    print(f"Claim: {claim2}")
    print(f"Found {len(result2.linked_evidence)} pieces of evidence")
    print(
        f"Support: {result2.total_support}, Refute: {result2.total_refute}, Neutral: {result2.total_neutral}"
    )

    for i, evidence in enumerate(result2.linked_evidence[:3], 1):
        print(f"\n  Evidence {i}:")
        print(f"    Text: {evidence.evidence_text[:80]}...")
        print(f"    Type: {evidence.support_type.value}")
        print(f"    Relevance: {evidence.relevance_score:.2f}")
        print(f"    Confidence: {evidence.confidence:.2f}")

    # Test Case 3: Direct classification
    print("\n\nTest Case 3: Direct Support Classification")
    print("-" * 60)
    test_claim = "Water freezes at zero degrees"
    test_evidence = "Ice forms when water reaches 0 degrees Celsius"

    classification = linker.classify_support(test_claim, test_evidence)
    print(f"Claim: {test_claim}")
    print(f"Evidence: {test_evidence}")
    print(f"Classification: {classification.value}")

    # Cleanup
    storage.clear()
    print("\n\nDemo completed successfully!")


if __name__ == "__main__":
    demo_evidence_linking()
