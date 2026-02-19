"""
Example: Custom Enrichment Pipeline

Description:
    Demonstrates building a custom enrichment pipeline that extends
    IngestForge's base enrichers with domain-specific processors.
    Shows how to create reusable enrichment modules.

Usage:
    python examples/advanced/custom_enrichment.py \
        --input documents/ \
        --enrichers sentiment,ner,keywords \
        --output enriched.jsonl

Expected output:
    - Chunks with custom metadata
    - Enrichment statistics
    - Quality scores

Requirements:
    - Documents to enrich
    - Optional: sentiment analysis library
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from ingestforge.ingest.processor import DocumentProcessor
from ingestforge.chunking.semantic_chunker import SemanticChunker
from ingestforge.storage.jsonl import JSONLStorage


class Enricher(ABC):
    """Base class for custom enrichers."""

    @abstractmethod
    def enrich(self, text: str) -> dict:
        """
        Enrich text with metadata.

        Args:
            text: Input text

        Returns:
            Dictionary with enrichment results
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Enricher name."""
        pass


class SentimentEnricher(Enricher):
    """Sentiment analysis enricher."""

    @property
    def name(self) -> str:
        return "sentiment"

    def enrich(self, text: str) -> dict:
        """Extract sentiment from text."""
        try:
            from textblob import TextBlob

            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity

            # Classify sentiment
            if polarity > 0.1:
                sentiment = "positive"
            elif polarity < -0.1:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            return {
                "sentiment": sentiment,
                "polarity": round(polarity, 3),
                "subjectivity": round(subjectivity, 3),
            }

        except ImportError:
            return {
                "error": "textblob not installed (pip install textblob)",
            }
        except Exception as e:
            return {"error": str(e)}


class KeywordEnricher(Enricher):
    """Extract important keywords."""

    @property
    def name(self) -> str:
        return "keywords"

    def enrich(self, text: str) -> dict:
        """Extract keywords."""
        # Simple implementation: extract capitalized words
        import re

        # Find capitalized phrases (likely proper nouns or important terms)
        pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        keywords = list(set(re.findall(pattern, text)))

        # Sort by frequency in text
        keywords_with_freq = [(kw, text.lower().count(kw.lower())) for kw in keywords]

        keywords_with_freq.sort(key=lambda x: x[1], reverse=True)

        return {
            "keywords": [kw for kw, _ in keywords_with_freq[:10]],
            "keyword_count": len(keywords),
        }


class NamedEntityEnricher(Enricher):
    """Extract named entities (NER)."""

    @property
    def name(self) -> str:
        return "ner"

    def enrich(self, text: str) -> dict:
        """Extract named entities."""
        try:
            import spacy

            # Load model
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                print("[!] spacy model not found. Install with:")
                print("    python -m spacy download en_core_web_sm")
                return {"error": "spacy model not found"}

            doc = nlp(text[:5000])  # Limit to first 5000 chars

            entities = {}
            for ent in doc.ents:
                label = ent.label_

                if label not in entities:
                    entities[label] = []

                entities[label].append(ent.text)

            # Remove duplicates and limit
            for label in entities:
                entities[label] = list(set(entities[label]))[:5]

            return {
                "entities": entities,
                "entity_count": sum(len(v) for v in entities.values()),
            }

        except ImportError:
            return {
                "error": "spacy not installed (pip install spacy)",
            }
        except Exception as e:
            return {"error": str(e)}


class ReadabilityEnricher(Enricher):
    """Calculate text readability metrics."""

    @property
    def name(self) -> str:
        return "readability"

    def enrich(self, text: str) -> dict:
        """Calculate readability metrics."""
        # Simple metrics
        words = text.split()
        sentences = text.count(".") + text.count("!") + text.count("?")
        paragraphs = text.count("\n\n")

        word_count = len(words)
        sentence_count = max(1, sentences)
        paragraph_count = max(1, paragraphs)

        avg_word_length = sum(len(w) for w in words) / word_count if words else 0
        avg_sentence_length = word_count / sentence_count

        # Flesch Kincaid Grade Level (simplified)
        grade_level = (
            max(
                0,
                (
                    0.39 * avg_sentence_length
                    + 11.8 * (sum(len(w) > 3 for w in words) / word_count)
                    - 15.59
                ),
            )
            if words
            else 0
        )

        return {
            "word_count": word_count,
            "sentence_count": sentence_count,
            "paragraph_count": paragraph_count,
            "avg_word_length": round(avg_word_length, 2),
            "avg_sentence_length": round(avg_sentence_length, 2),
            "grade_level": round(grade_level, 1),
        }


class EnrichmentPipeline:
    """Pipeline for applying multiple enrichers."""

    def __init__(self, enrichers: list[str]):
        """
        Initialize pipeline.

        Args:
            enrichers: List of enricher names
        """
        self.enricher_registry = {
            "sentiment": SentimentEnricher(),
            "ner": NamedEntityEnricher(),
            "keywords": KeywordEnricher(),
            "readability": ReadabilityEnricher(),
        }

        self.enrichers = [
            self.enricher_registry[name]
            for name in enrichers
            if name in self.enricher_registry
        ]

        if not self.enrichers:
            print("[!] No valid enrichers specified")

    def enrich(self, chunk: dict) -> dict:
        """
        Enrich a single chunk.

        Args:
            chunk: Chunk dictionary

        Returns:
            Enriched chunk
        """
        text = chunk.get("text", "")

        if "metadata" not in chunk:
            chunk["metadata"] = {}

        enrichment = {}

        for enricher in self.enrichers:
            result = enricher.enrich(text)
            enrichment[enricher.name] = result

        chunk["metadata"]["enrichments"] = enrichment
        return chunk


def enrich_documents(
    input_dir: str,
    output_path: str = "enriched.jsonl",
    enrichers: list[str] = None,
) -> None:
    """
    Enrich documents with custom metadata.

    Args:
        input_dir: Input documents directory
        output_path: Output JSONL file
        enrichers: List of enricher names to apply
    """
    if enrichers is None:
        enrichers = ["sentiment", "keywords", "readability"]

    input_dir = Path(input_dir)

    if not input_dir.exists():
        print(f"Error: Directory not found: {input_dir}")
        return

    print(f"\n[*] Processing documents: {input_dir}")
    print(f"    Enrichers: {', '.join(enrichers)}")
    print(f"    Output: {output_path}")

    # Find documents
    doc_files = (
        list(input_dir.glob("**/*.pdf"))
        + list(input_dir.glob("**/*.txt"))
        + list(input_dir.glob("**/*.md"))
    )

    if not doc_files:
        print("[!] No documents found")
        return

    print(f"\n[*] Found {len(doc_files)} documents")

    # Initialize components
    processor = DocumentProcessor()
    chunker = SemanticChunker(target_size=512, overlap=50)
    pipeline = EnrichmentPipeline(enrichers)
    storage = JSONLStorage(output_path)

    enriched_chunks = []

    # Process documents
    print("\n[*] Processing documents...")

    for i, doc_file in enumerate(doc_files, 1):
        print(f"    [{i}/{len(doc_files)}] {doc_file.name}...", end="", flush=True)

        try:
            # Process
            text = processor.process(doc_file)
            chunks = chunker.chunk(text)

            # Enrich each chunk
            for chunk in chunks:
                chunk = pipeline.enrich(chunk)
                enriched_chunks.append(chunk)

            print(f" OK ({len(chunks)} chunks)")

        except Exception as e:
            print(f" ERROR: {e}")

    # Save results
    print(f"\n[*] Saving {len(enriched_chunks)} enriched chunks...")
    storage.save(enriched_chunks)

    # Print summary
    print(f"\n{'='*70}")
    print("ENRICHMENT SUMMARY")
    print(f"{'='*70}\n")

    print(f"Documents processed: {len(doc_files)}")
    print(f"Chunks enriched:     {len(enriched_chunks)}")
    print(f"Enrichers applied:   {', '.join(enrichers)}")
    print(f"Output:              {output_path}")

    # Show enrichment examples
    if enriched_chunks:
        print("\nExample enrichments:")
        example_chunk = enriched_chunks[0]
        enrichments = example_chunk.get("metadata", {}).get("enrichments", {})

        for enricher_name, result in enrichments.items():
            print(f"\n  {enricher_name}:")
            if "error" in result:
                print(f"    {result['error']}")
            else:
                for key, value in list(result.items())[:3]:
                    print(f"    {key}: {value}")

    print("\n[✓] Enrichment complete!")


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Apply custom enrichments to documents"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input documents directory"
    )
    parser.add_argument(
        "--enrichers",
        "-e",
        default="sentiment,keywords,readability",
        help="Enrichers to apply (sentiment,ner,keywords,readability)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="enriched.jsonl",
        help="Output file (default: enriched.jsonl)",
    )

    args = parser.parse_args()

    enrichers = [e.strip() for e in args.enrichers.split(",")]

    enrich_documents(
        args.input,
        args.output,
        enrichers,
    )


if __name__ == "__main__":
    main()
