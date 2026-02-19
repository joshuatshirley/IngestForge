"""
Example: Generate Flashcards from Documents

Description:
    Automatically generates flashcard-style question-answer pairs from
    any document. Exports to CSV format compatible with Anki, Quizlet,
    and other flashcard applications.

Usage:
    python examples/quickstart/03_generate_flashcards.py document.pdf
    python examples/quickstart/03_generate_flashcards.py --help

Expected output:
    - CSV file with flashcard data
    - Questions and answers extracted from text
    - Front/back card format

Requirements:
    - LLM API key (OPENAI_API_KEY or ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from ingestforge.ingest.processor import DocumentProcessor
from ingestforge.chunking.semantic_chunker import SemanticChunker
from ingestforge.enrichment.questions import QuestionGenerator
from ingestforge.llm.factory import LLMFactory


def generate_flashcards(
    file_path: str,
    output_path: Optional[str] = None,
    questions_per_chunk: int = 2,
    chunk_size: int = 512,
    llm_provider: str = "openai",
) -> None:
    """
    Generate flashcards from a document.

    Args:
        file_path: Path to input document
        output_path: Where to save CSV file (default: {filename}_flashcards.csv)
        questions_per_chunk: Number of questions to generate per chunk
        chunk_size: Target chunk size
        llm_provider: LLM provider to use (openai, anthropic, etc.)
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    if output_path is None:
        output_path = file_path.stem + "_flashcards.csv"

    print("\n[*] Generating flashcards")
    print(f"    Input: {file_path}")
    print(f"    Output: {output_path}")

    try:
        # Step 1: Process document
        print("\n[*] Processing document...")
        processor = DocumentProcessor()
        text = processor.process(file_path)
        print(f"    Extracted {len(text.split())} words")

        # Step 2: Chunk the text
        print("\n[*] Chunking text...")
        chunker = SemanticChunker(target_size=chunk_size, overlap=50)
        chunks = chunker.chunk(text)
        print(f"    Created {len(chunks)} chunks")

        # Step 3: Initialize LLM and question generator
        print(f"\n[*] Initializing LLM ({llm_provider})...")
        try:
            llm = LLMFactory.create(provider=llm_provider)
            question_gen = QuestionGenerator(llm=llm)
        except Exception as e:
            print(f"[!] LLM initialization failed: {e}")
            print(
                "    Skipping flashcard generation. Install API key or use --mock mode"
            )
            return

        # Step 4: Generate questions
        print("\n[*] Generating questions...")
        flashcards = []

        for i, chunk in enumerate(chunks, 1):
            chunk_text = chunk["text"]

            print(
                f"    [{i}/{len(chunks)}] Generating {questions_per_chunk} questions...",
                end="",
                flush=True,
            )

            try:
                # Generate questions for this chunk
                questions = question_gen.generate(
                    chunk_text,
                    num_questions=questions_per_chunk,
                )

                for q_data in questions:
                    flashcards.append(
                        {
                            "front": q_data.get("question", ""),
                            "back": q_data.get("answer", ""),
                            "source": chunk.get("metadata", {}).get(
                                "source", "Unknown"
                            ),
                            "difficulty": q_data.get("difficulty", "medium"),
                            "tags": ";".join(q_data.get("tags", [])),
                        }
                    )

                print(" OK")

            except Exception as e:
                print(f" ERROR: {e}")

        # Step 5: Save to CSV
        print(f"\n[*] Saving {len(flashcards)} flashcards...")
        save_flashcards_csv(flashcards, output_path)

        # Step 6: Print summary
        print(f"\n{'='*70}")
        print("FLASHCARD GENERATION SUMMARY")
        print(f"{'='*70}\n")

        print(f"Total flashcards:    {len(flashcards)}")
        print(f"Questions per chunk: {questions_per_chunk}")
        print(f"Total chunks:        {len(chunks)}")
        print(f"Output file:         {output_path}")

        if flashcards:
            print("\nExample flashcards:")
            for i, card in enumerate(flashcards[:3], 1):
                print(f"\n[Card {i}]")
                print(f"Q: {card['front']}")
                print(f"A: {card['back']}")

        print("\n[✓] Flashcards generated successfully!")
        print("\nImport into Anki:")
        print("  1. In Anki, click 'Import'")
        print(f"  2. Select '{output_path}'")
        print("  3. Field mapping: Front | Back")

    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback

        traceback.print_exc()


def save_flashcards_csv(
    flashcards: list[dict],
    output_path: str,
) -> None:
    """
    Save flashcards to CSV file.

    Args:
        flashcards: List of flashcard dictionaries
        output_path: Path to save CSV file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["front", "back", "source", "difficulty", "tags"],
        )
        writer.writeheader()
        writer.writerows(flashcards)


def load_flashcards_csv(csv_path: str) -> list[dict]:
    """
    Load flashcards from CSV file.

    Args:
        csv_path: Path to CSV file

    Returns:
        List of flashcard dictionaries
    """
    flashcards = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        flashcards.extend(reader)
    return flashcards


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate flashcards from documents")
    parser.add_argument(
        "file",
        nargs="?",
        default="examples/data/sample.pdf",
        help="Path to document (default: examples/data/sample.pdf)",
    )
    parser.add_argument(
        "--output", "-o", help="Output CSV path (default: {filename}_flashcards.csv)"
    )
    parser.add_argument(
        "--questions-per-chunk",
        type=int,
        default=2,
        help="Questions per chunk (default: 2)",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=512, help="Chunk size (default: 512)"
    )
    parser.add_argument(
        "--llm",
        default="openai",
        choices=["openai", "anthropic", "gemini"],
        help="LLM provider (default: openai)",
    )
    parser.add_argument(
        "--mock", action="store_true", help="Mock mode (for testing without API)"
    )

    args = parser.parse_args()

    if args.mock:
        print("[*] Running in mock mode (no LLM calls)")
        # TODO: Implement mock mode with sample flashcards
        return

    generate_flashcards(
        args.file,
        output_path=args.output,
        questions_per_chunk=args.questions_per_chunk,
        chunk_size=args.chunk_size,
        llm_provider=args.llm,
    )


if __name__ == "__main__":
    main()
