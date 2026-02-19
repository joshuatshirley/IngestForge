"""Study subcommands.

Provides tools for learning from ingested content:
- quiz: Generate quiz questions for testing knowledge
- flashcards: Create flashcard sets for memorization
- review: Spaced repetition scheduling with SM-2 algorithm
- glossary: Generate term definitions
- notes: Generate study notes

Follows Commandments #4 (Small Functions) and #1 (Simple Control Flow).
"""

from __future__ import annotations

import typer

from ingestforge.cli.study import quiz, flashcards, glossary, notes, review

# Create study subcommand application
app = typer.Typer(
    name="study",
    help="Study and learning tools",
    add_completion=False,
)

# Register legacy commands for backwards compatibility
app.command("quiz")(quiz.command)
app.command("flashcards")(flashcards.command)
app.command("glossary")(glossary.command)
app.command("notes")(notes.command)
app.command("review")(review.command)


@app.callback()
def main() -> None:
    """Study and learning tools for IngestForge.

    Generate study materials from your knowledge base:
    - Quiz questions (multiple choice and open-ended)
    - Flashcard sets for memorization (Anki, Quizlet, Markdown)
    - Spaced repetition scheduling with SM-2 algorithm

    All commands work with ingested documents to create
    personalized study materials.

    Examples:
        # Generate quiz questions
        ingestforge study quiz "Python programming" --count 10

        # Create Anki flashcards
        ingestforge study flashcards "Biology" --format anki -o cards.csv

        # Create spaced repetition schedule
        ingestforge study review "History" --algorithm sm2

        # Save study materials
        ingestforge study quiz "History" -o quiz.json
        ingestforge study flashcards "Math" -o cards.json

    For help on specific commands:
        ingestforge study <command> --help
    """
    pass
