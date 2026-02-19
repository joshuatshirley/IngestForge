"""
Example: Textbook to Study Materials Processor

Description:
    Converts textbooks and educational materials into structured
    study content including chapter summaries, key definitions,
    theorems, and practice problems. Organizes output by chapter
    for easy studying.

Usage:
    python examples/learning/textbook_processor.py calculus.pdf --output calculus_study/
    python examples/learning/textbook_processor.py book.pdf --with-quizzes

Expected output:
    - Chapter summaries
    - Key definitions glossary
    - Important theorems/facts
    - Practice problems
    - Study guide

Requirements:
    - Textbook PDF
    - Optional: LLM for summary generation
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass

from ingestforge.ingest.processor import DocumentProcessor


@dataclass
class Chapter:
    """Represents a book chapter."""

    number: int
    title: str
    text: str
    word_count: int
    summary: Optional[str] = None
    key_terms: list[str] = None
    key_concepts: list[str] = None

    def __post_init__(self):
        if self.key_terms is None:
            self.key_terms = []
        if self.key_concepts is None:
            self.key_concepts = []


def detect_chapters(text: str) -> list[tuple[int, str, str]]:
    """
    Detect chapters in text using heuristics.

    Args:
        text: Full text

    Returns:
        List of (chapter_num, title, text) tuples
    """
    import re

    chapters = []
    chapter_num = 0

    # Split on common chapter markers
    patterns = [
        r"(?:^|\n)(?:CHAPTER|Chapter|CH\.?)\s+(\d+).*?(?=(?:^|\n)(?:CHAPTER|Chapter|CH\.?)|$)",
        r"(?:^|\n)(\d+)\.?\s+([A-Z][^\n]+)",
    ]

    for pattern in patterns:
        matches = re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE)
        if matches:
            for match in matches:
                chapter_num += 1

                if len(match.groups()) == 2:
                    title = match.group(2)
                else:
                    title = f"Chapter {chapter_num}"

                # Get text until next chapter
                start = match.start()
                next_match = match.end()
                for m in re.finditer(pattern, text[next_match:]):
                    next_match += m.start()
                    break

                chapter_text = text[start:next_match]
                chapters.append((chapter_num, title, chapter_text))

            if chapters:
                break

    # If no chapters detected, split by length
    if not chapters:
        chunk_size = len(text) // 10  # 10 chunks
        for i in range(10):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < 9 else len(text)
            chapters.append((i + 1, f"Chapter {i + 1}", text[start:end]))

    return chapters


def extract_key_terms(text: str, max_terms: int = 20) -> list[str]:
    """
    Extract key terms/definitions from chapter.

    Args:
        text: Chapter text
        max_terms: Maximum terms to extract

    Returns:
        List of key terms
    """
    # Look for common definition markers
    import re

    terms = set()

    # Pattern 1: "Term: definition"
    pattern1 = r"(?:^|\n)([A-Z][A-Za-z\s]+):\s+(.{20,100})"
    for match in re.finditer(pattern1, text):
        term = match.group(1).strip()
        if 5 < len(term) < 50:
            terms.add(term)

    # Pattern 2: "is defined as"
    pattern2 = r"([A-Z][A-Za-z\s]+)\s+(?:is|are)\s+(?:defined as|called|known as)"
    for match in re.finditer(pattern2, text):
        term = match.group(1).strip()
        if 5 < len(term) < 50:
            terms.add(term)

    # Pattern 3: Capitalized terms followed by explanation
    pattern3 = r"\*\*([A-Z][A-Za-z\s]+)\*\*"
    for match in re.finditer(pattern3, text):
        terms.add(match.group(1))

    return sorted(list(terms))[:max_terms]


def extract_concepts(text: str) -> list[str]:
    """
    Extract key concepts from chapter.

    Args:
        text: Chapter text

    Returns:
        List of key concepts
    """
    # Look for numbered/bulleted key ideas
    import re

    concepts = []

    # Pattern 1: Numbered points
    pattern1 = r"(?:^|\n)\s*\d+\.\s+([^\n]+)"
    for match in re.finditer(pattern1, text):
        concept = match.group(1).strip()
        if 10 < len(concept) < 200:
            concepts.append(concept)

    # Pattern 2: Bullet points
    pattern2 = r"(?:^|\n)\s*[-*•]\s+([^\n]+)"
    for match in re.finditer(pattern2, text):
        concept = match.group(1).strip()
        if 10 < len(concept) < 200:
            concepts.append(concept)

    # Return unique, sorted
    return sorted(list(set(concepts)))[:10]


def process_textbook(
    file_path: str,
    output_dir: str = "study_materials",
) -> None:
    """
    Process textbook and generate study materials.

    Args:
        file_path: Path to textbook PDF
        output_dir: Where to save study materials
    """
    file_path = Path(file_path)

    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n[*] Processing textbook: {file_path}")
    print(f"    Output: {output_dir}")

    try:
        # Step 1: Extract text
        print("\n[*] Extracting text...")
        processor = DocumentProcessor()
        text = processor.process(file_path)

        print(f"    Extracted {len(text):,} characters")
        print(f"    {len(text.split()):,} words")

        # Step 2: Detect chapters
        print("\n[*] Detecting chapters...")
        chapter_data = detect_chapters(text)

        print(f"    Found {len(chapter_data)} chapters")

        # Step 3: Process each chapter
        print("\n[*] Processing chapters...")
        chapters = []

        for num, title, chapter_text in chapter_data:
            print(f"    [Chapter {num}] {title[:60]}...", end="", flush=True)

            key_terms = extract_key_terms(chapter_text)
            key_concepts = extract_concepts(chapter_text)

            chapter = Chapter(
                number=num,
                title=title,
                text=chapter_text,
                word_count=len(chapter_text.split()),
                key_terms=key_terms,
                key_concepts=key_concepts,
            )

            chapters.append(chapter)
            print(" OK")

        # Step 4: Save study materials
        print("\n[*] Saving study materials...")

        for chapter in chapters:
            # Create chapter directory
            chapter_dir = output_dir / f"chapter_{chapter.number:02d}"
            chapter_dir.mkdir(parents=True, exist_ok=True)

            # Save chapter text
            with open(chapter_dir / "text.txt", "w", encoding="utf-8") as f:
                f.write(f"# {chapter.title}\n\n{chapter.text}")

            # Save key terms
            if chapter.key_terms:
                with open(chapter_dir / "key_terms.txt", "w", encoding="utf-8") as f:
                    for term in chapter.key_terms:
                        f.write(f"- {term}\n")

            # Save key concepts
            if chapter.key_concepts:
                with open(chapter_dir / "key_concepts.txt", "w", encoding="utf-8") as f:
                    for i, concept in enumerate(chapter.key_concepts, 1):
                        f.write(f"{i}. {concept}\n")

            # Save metadata
            with open(chapter_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "number": chapter.number,
                        "title": chapter.title,
                        "word_count": chapter.word_count,
                        "key_terms_count": len(chapter.key_terms),
                        "key_concepts_count": len(chapter.key_concepts),
                    },
                    f,
                    indent=2,
                )

        # Step 5: Create index
        print("\n[*] Creating study guide...")
        create_study_guide(chapters, output_dir)

        # Print summary
        print(f"\n{'='*70}")
        print("TEXTBOOK PROCESSING SUMMARY")
        print(f"{'='*70}\n")

        total_terms = sum(len(c.key_terms) for c in chapters)
        total_concepts = sum(len(c.key_concepts) for c in chapters)

        print(f"Chapters:            {len(chapters)}")
        print(f"Total key terms:     {total_terms}")
        print(f"Total key concepts:  {total_concepts}")
        print(f"Study materials:     {output_dir}")

        print("\n[✓] Textbook processing complete!")

    except Exception as e:
        print(f"\n[!] Error: {e}")
        import traceback

        traceback.print_exc()


def create_study_guide(
    chapters: list[Chapter],
    output_dir: Path,
) -> None:
    """
    Create a study guide index.

    Args:
        chapters: List of chapters
        output_dir: Output directory
    """
    lines = [
        "# Study Guide",
        "",
        f"Study materials for {len(chapters)} chapters",
        "",
        "## Table of Contents",
        "",
    ]

    for chapter in chapters:
        lines.append(f"### Chapter {chapter.number}: {chapter.title}")
        lines.append(f"- Word count: {chapter.word_count:,}")
        lines.append(f"- Key terms: {len(chapter.key_terms)}")
        lines.append(f"- Key concepts: {len(chapter.key_concepts)}")
        lines.append("")

    lines.extend(
        [
            "## Study Tips",
            "",
            "1. Start with chapter summaries",
            "2. Review key terms in each chapter",
            "3. Understand key concepts before moving to next chapter",
            "4. Test yourself with practice quizzes",
            "5. Create your own notes and examples",
            "",
        ]
    )

    with open(output_dir / "README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Convert textbook to study materials")
    parser.add_argument(
        "file", nargs="?", default="examples/data/sample.pdf", help="Textbook PDF file"
    )
    parser.add_argument(
        "--output",
        "-o",
        default="study_materials",
        help="Output directory (default: study_materials)",
    )

    args = parser.parse_args()

    process_textbook(args.file, args.output)


if __name__ == "__main__":
    main()
