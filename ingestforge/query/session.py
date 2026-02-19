"""
Conversation session management.

Tracks multi-turn conversation state for the interactive ask command,
with word-budget context management for LLM prompts.
"""

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    turn_number: int
    question: str
    answer: Optional[str]
    sources: List[Dict[str, Any]]
    query_type: str
    timestamp: float


@dataclass
class ConversationSession:
    """
    Multi-turn conversation state.

    Manages conversation history with word-budget context management
    that prioritizes recent turns and trims at sentence boundaries.
    """

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    turns: List[ConversationTurn] = field(default_factory=list)
    max_history_words: int = 2000
    created_at: float = field(default_factory=time.time)

    # Follow-up detection patterns
    _PRONOUN_PATTERN = re.compile(
        r"\b(it|that|this|those|these|them|they|the same)\b", re.IGNORECASE
    )
    _FOLLOWUP_PHRASES = re.compile(
        r"(more about|the (first|second|third|last|next)|elaborate|expand|explain further|go deeper|what about)",
        re.IGNORECASE,
    )

    def add_turn(
        self,
        question: str,
        answer: Optional[str],
        sources: List[Dict[str, Any]],
        query_type: str = "unknown",
    ) -> ConversationTurn:
        """Add a new turn to the conversation."""
        turn = ConversationTurn(
            turn_number=len(self.turns) + 1,
            question=question,
            answer=answer,
            sources=sources,
            query_type=query_type,
            timestamp=time.time(),
        )
        self.turns.append(turn)
        return turn

    def get_context_for_llm(self) -> str:
        """
        Build conversation context for LLM prompt.

        Uses a word-budget approach: allocates max_history_words total,
        starts from the most recent turn and walks backward. Each turn
        includes the full question but trims the answer to fit the
        remaining budget. Stops adding turns when budget is exhausted.
        Trims at sentence boundaries to avoid confusing the LLM.
        """
        if not self.turns:
            return ""

        budget = self.max_history_words
        context_parts = []

        # Walk backward from most recent turn
        for turn in reversed(self.turns):
            question_words = len(turn.question.split())

            if budget < question_words + 5:
                # Not enough room for even the question
                break

            answer_text = turn.answer or "(no answer)"
            answer_words = answer_text.split()
            remaining_for_answer = budget - question_words

            if len(answer_words) <= remaining_for_answer:
                trimmed_answer = answer_text
            else:
                trimmed_answer = self._trim_at_sentence_boundary(
                    answer_text, remaining_for_answer
                )

            part = f"Q: {turn.question}\nA: {trimmed_answer}"
            words_used = question_words + len(trimmed_answer.split())
            budget -= words_used

            context_parts.append(part)

            if budget <= 0:
                break

        # Reverse so oldest context comes first
        context_parts.reverse()
        return "\n\n".join(context_parts)

    def is_follow_up(self, query: str) -> bool:
        """
        Detect if a query is a follow-up to the previous turn.

        Detects: short queries (<5 words), pronoun references,
        follow-up phrases like "more about", "the second".
        """
        if not self.turns:
            return False

        words = query.split()

        # Short queries are likely follow-ups
        if len(words) < 5:
            return True

        # Check for pronoun references
        if self._PRONOUN_PATTERN.search(query):
            return True

        # Check for follow-up phrases
        if self._FOLLOWUP_PHRASES.search(query):
            return True

        return False

    def _get_stopwords(self) -> set[str]:
        """
        Get set of common English stopwords.

        Rule #1: Extracted constant to reduce function size
        Rule #4: Function <60 lines
        Rule #6: Smallest scope - only created when needed
        Rule #9: Full type hints

        Returns:
            Set of stopwords to filter
        """
        return {
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "can",
            "shall",
            "to",
            "of",
            "in",
            "for",
            "on",
            "with",
            "at",
            "by",
            "from",
            "as",
            "into",
            "about",
            "between",
            "through",
            "during",
            "before",
            "after",
            "and",
            "but",
            "or",
            "nor",
            "not",
            "so",
            "yet",
            "it",
            "that",
            "this",
            "those",
            "these",
            "them",
            "they",
            "i",
            "me",
            "my",
            "we",
            "us",
            "our",
            "you",
            "your",
            "he",
            "him",
            "his",
            "she",
            "her",
            "what",
            "which",
            "who",
            "how",
            "when",
            "where",
            "why",
            "more",
            "tell",
            "explain",
        }

    def _extract_words_from_text(self, text: str) -> List[str]:
        """
        Extract words of 3+ characters from text.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            text: Text to extract words from

        Returns:
            List of extracted words (lowercase)
        """
        assert text is not None, "Text cannot be None"
        assert isinstance(text, str), "Text must be string"
        words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
        return words

    def _filter_keywords(
        self,
        words: List[str],
        stopwords: set[str],
        existing_keywords: List[str],
    ) -> List[str]:
        """
        Filter words to get valid keywords (not stopwords, not duplicates).

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            words: Words to filter
            stopwords: Set of stopwords to exclude
            existing_keywords: Already collected keywords (for deduplication)

        Returns:
            List of new keywords
        """
        assert words is not None, "Words cannot be None"
        assert stopwords is not None, "Stopwords cannot be None"
        assert existing_keywords is not None, "Existing keywords cannot be None"
        new_keywords = [
            word
            for word in words
            if word not in stopwords and word not in existing_keywords
        ]

        return new_keywords

    def _collect_keywords_from_turns(self) -> List[str]:
        """
        Collect keywords from all conversation turns.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Returns:
            List of unique keywords from conversation history
        """
        MAX_TURNS: int = 1000  # Hard limit on conversation length
        turns_processed: int = 0

        stopwords = self._get_stopwords()
        keywords: List[str] = []
        for turn in self.turns:
            turns_processed += 1
            if turns_processed > MAX_TURNS:
                logger.warning(f"Safety limit: processed {MAX_TURNS} turns")
                break

            # Process question and answer
            for text in [turn.question, turn.answer or ""]:
                words = self._extract_words_from_text(text)
                new_keywords = self._filter_keywords(words, stopwords, keywords)
                keywords.extend(new_keywords)

        return keywords

    def get_retrieval_augmentation(self, query: str) -> str:
        """
        Augment a follow-up query with keywords from conversation history.

        Rule #1: Zero nesting - all logic extracted to helpers
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Extracts meaningful keywords from previous questions and answers,
        filters out stopwords, and prepends them to the current query
        to improve retrieval quality for follow-up questions.

        Args:
            query: Current query to augment

        Returns:
            Augmented query with historical keywords prepended
        """
        assert query is not None, "Query cannot be None"
        assert isinstance(query, str), "Query must be string"
        if not self.turns:
            return query
        keywords = self._collect_keywords_from_turns()
        if not keywords:
            return query
        # Prepend up to 5 keywords to the query
        augmentation = " ".join(keywords[:5])
        result = f"{augmentation} {query}"
        assert query in result, "Result must contain original query"

        return result

    def export_markdown(self) -> str:
        """Export conversation as markdown."""
        lines = [
            f"# Conversation: {self.summary_topic}",
            f"Session: {self.session_id}",
            f"Turns: {self.turn_count}",
            "",
        ]

        for turn in self.turns:
            lines.append(f"## Q{turn.turn_number}: {turn.question}")
            lines.append("")
            if turn.answer:
                lines.append(turn.answer)
            else:
                lines.append("*(no answer generated)*")
            lines.append("")

            if turn.sources:
                lines.append("**Sources:**")
                for i, src in enumerate(turn.sources, 1):
                    title = src.get("section_title", src.get("source_file", "Unknown"))
                    lines.append(f"- [{i}] {title}")
                lines.append("")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize session to dictionary."""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "max_history_words": self.max_history_words,
            "turns": [
                {
                    "turn_number": t.turn_number,
                    "question": t.question,
                    "answer": t.answer,
                    "sources": t.sources,
                    "query_type": t.query_type,
                    "timestamp": t.timestamp,
                }
                for t in self.turns
            ],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ConversationSession":
        """Deserialize session from dictionary."""
        session = cls(
            session_id=data["session_id"],
            created_at=data.get("created_at", time.time()),
            max_history_words=data.get("max_history_words", 2000),
        )
        for t in data.get("turns", []):
            session.turns.append(
                ConversationTurn(
                    turn_number=t["turn_number"],
                    question=t["question"],
                    answer=t.get("answer"),
                    sources=t.get("sources", []),
                    query_type=t.get("query_type", "unknown"),
                    timestamp=t.get("timestamp", 0.0),
                )
            )
        return session

    @property
    def turn_count(self) -> int:
        """Number of turns in the conversation."""
        return len(self.turns)

    @property
    def summary_topic(self) -> str:
        """Derive topic from first question."""
        if self.turns:
            first_q = self.turns[0].question
            # Truncate long questions
            if len(first_q) > 60:
                return first_q[:57] + "..."
            return first_q
        return "New conversation"

    @staticmethod
    def _trim_at_sentence_boundary(text: str, max_words: int) -> str:
        """
        Trim text to fit within word budget, respecting sentence boundaries.

        Splits on '. ' and builds up sentences until budget is reached.
        """
        words = text.split()
        if len(words) <= max_words:
            return text

        # Split into sentences
        sentences = re.split(r"(?<=\.)\s+", text)
        result: list[Any] = []
        word_count = 0

        for sentence in sentences:
            sentence_words = len(sentence.split())
            if word_count + sentence_words > max_words:
                break
            result.append(sentence)
            word_count += sentence_words

        if result:
            return " ".join(result)

        # If even the first sentence is too long, truncate by words
        return " ".join(words[:max_words]) + "..."
