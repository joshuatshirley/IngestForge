"""
Benchmark semantic chunking quality improvement.

Compares retrieval quality between:
- Jaccard word-based similarity
- Embedding-based cosine similarity

Acceptance criteria: >10% improvement in retrieval quality
"""

import time
from typing import List, Tuple
from ingestforge.chunking.semantic_chunker import SemanticChunker, ChunkRecord


# Multi-topic test corpus for retrieval quality testing
TEST_CORPUS = """
Machine learning is a branch of artificial intelligence that focuses on building applications
that learn from data and improve their accuracy over time without being programmed to do so.
Deep learning is a subset of machine learning that uses neural networks with multiple layers.
These neural networks attempt to simulate the behavior of the human brain, allowing it to
learn from large amounts of data.

Python is a high-level, interpreted programming language known for its clear syntax and
readability. It is widely used in web development, data analysis, artificial intelligence,
scientific computing, and automation. Python's extensive standard library and vast ecosystem
of third-party packages make it a versatile choice for developers.

Climate change refers to long-term shifts in temperatures and weather patterns. These shifts
may be natural, but since the 1800s, human activities have been the main driver of climate
change, primarily due to the burning of fossil fuels like coal, oil, and gas. The effects
include rising sea levels, more frequent extreme weather events, and disruption of ecosystems.

Quantum computing is a type of computation that harnesses the collective properties of quantum
states, such as superposition and entanglement, to perform calculations. Quantum computers
have the potential to solve certain problems much faster than classical computers, particularly
in areas like cryptography, drug discovery, and optimization problems.
"""


# Test queries for each topic
TEST_QUERIES = [
    ("What is deep learning?", "Machine learning"),  # Should retrieve ML paragraph
    ("Tell me about Python programming.", "Python"),  # Should retrieve Python paragraph
    (
        "How does climate change affect the environment?",
        "Climate",
    ),  # Should retrieve climate paragraph
    ("Explain quantum computing.", "Quantum"),  # Should retrieve quantum paragraph
]


def calculate_retrieval_quality(
    chunks: List[ChunkRecord],
    queries: List[Tuple[str, str]],
    chunker: SemanticChunker,
) -> float:
    """
    Calculate retrieval quality by checking if the right chunk is found.

    Returns percentage of queries that retrieved the correct chunk.
    """
    correct = 0

    for query_text, expected_topic in queries:
        # Get query vector
        query_words = query_text.lower().split()
        if chunker.use_embeddings and chunker.embedding_generator:
            try:
                query_vec = chunker.embedding_generator.embed(query_text)
            except Exception:
                query_vec = query_words
        else:
            query_vec = query_words

        # Find best matching chunk
        best_chunk = None
        best_score = -1.0

        for chunk in chunks:
            # Get chunk vector
            if isinstance(query_vec[0], (float, int)):
                # Embedding mode - need to embed chunk too
                try:
                    chunk_vec = chunker.embedding_generator.embed(chunk.content)
                except Exception:
                    chunk_vec = chunk.content.lower().split()
            else:
                chunk_vec = chunk.content.lower().split()

            score = chunker._calculate_similarity(query_vec, chunk_vec)
            if score > best_score:
                best_score = score
                best_chunk = chunk

        # Check if correct chunk was retrieved
        if best_chunk and expected_topic.lower() in best_chunk.content.lower():
            correct += 1

    return (correct / len(queries)) * 100.0


def benchmark():
    """Run retrieval quality benchmark."""
    print("=" * 60)
    print("SEMANTIC CHUNKING RETRIEVAL QUALITY BENCHMARK")
    print("=" * 60)

    # Test 1: Word-based chunking
    print("\nTest 1: Word-based (Jaccard) chunking")
    print("-" * 60)
    chunker_word = SemanticChunker(
        max_chunk_size=800,
        min_chunk_size=100,
        similarity_threshold=0.6,
        use_embeddings=False,
    )

    start = time.time()
    chunks_word = chunker_word.chunk(TEST_CORPUS, "test_doc")
    time_word = time.time() - start

    quality_word = calculate_retrieval_quality(chunks_word, TEST_QUERIES, chunker_word)

    print(f"Chunks created: {len(chunks_word)}")
    print(f"Processing time: {time_word:.3f}s")
    print(f"Retrieval accuracy: {quality_word:.1f}%")

    # Test 2: Embedding-based chunking
    print("\nTest 2: Embedding-based (Cosine) chunking")
    print("-" * 60)

    try:
        chunker_emb = SemanticChunker(
            max_chunk_size=800,
            min_chunk_size=100,
            similarity_threshold=0.7,
            use_embeddings=True,
        )

        start = time.time()
        chunks_emb = chunker_emb.chunk(TEST_CORPUS, "test_doc")
        time_emb = time.time() - start

        quality_emb = calculate_retrieval_quality(chunks_emb, TEST_QUERIES, chunker_emb)

        print(f"Chunks created: {len(chunks_emb)}")
        print(f"Processing time: {time_emb:.3f}s")
        print(f"Retrieval accuracy: {quality_emb:.1f}%")

        # Calculate improvement
        print("\nComparison:")
        print("-" * 60)
        improvement = quality_emb - quality_word
        speed_ratio = time_emb / time_word if time_word > 0 else 0

        print(f"Accuracy improvement: {improvement:+.1f}%")
        print(f"Speed ratio (emb/word): {speed_ratio:.2f}x")

        # Check acceptance criteria
        print("\nAcceptance Criteria:")
        print("-" * 60)
        if improvement >= 10.0:
            print(
                f"✓ PASS: Retrieval quality improved by {improvement:.1f}% (>10% required)"
            )
        else:
            print(
                f"✗ FAIL: Retrieval quality improved by {improvement:.1f}% (<10% required)"
            )

        if time_emb < 10.0:
            print(f"✓ PASS: Processing time {time_emb:.3f}s (<10s acceptable)")
        else:
            print(f"⚠ WARN: Processing time {time_emb:.3f}s (may be slow)")

    except ImportError:
        print("sentence-transformers not installed")
        print("Install with: pip install sentence-transformers")
        print("\nSkipping embedding-based test")
    except Exception as e:
        print(f"Error during embedding test: {e}")
        print("Falling back to word-based mode")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark()
