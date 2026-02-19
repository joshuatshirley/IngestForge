"""Tests for conversation session management."""
import time

import pytest

from ingestforge.query.session import ConversationSession, ConversationTurn


class TestConversationTurn:
    """Tests for ConversationTurn dataclass."""

    def test_create_turn(self):
        """Test creating a conversation turn."""
        turn = ConversationTurn(
            turn_number=1,
            question="What is Python?",
            answer="Python is a programming language",
            sources=[{"file": "intro.txt"}],
            query_type="factoid",
            timestamp=time.time(),
        )

        assert turn.turn_number == 1
        assert turn.question == "What is Python?"
        assert turn.answer == "Python is a programming language"
        assert len(turn.sources) == 1
        assert turn.query_type == "factoid"


class TestConversationSession:
    """Tests for ConversationSession."""

    @pytest.fixture
    def session(self) -> ConversationSession:
        """Create a test session."""
        return ConversationSession(max_history_words=100)

    @pytest.fixture
    def populated_session(self) -> ConversationSession:
        """Create a session with multiple turns."""
        session = ConversationSession(max_history_words=100)
        session.add_turn(
            "What is Python?",
            "Python is a high-level programming language known for its simplicity.",
            [{"file": "intro.txt"}],
            "factoid",
        )
        session.add_turn(
            "What are its main features?",
            "Python features include dynamic typing, garbage collection, and extensive standard library.",
            [{"file": "features.txt"}],
            "list",
        )
        return session

    def test_session_initialization(self, session: ConversationSession):
        """Test session is initialized correctly."""
        assert session.session_id is not None
        assert len(session.session_id) == 8  # UUID hex[:8]
        assert session.turns == []
        assert session.max_history_words == 100
        assert session.created_at > 0

    def test_add_turn(self, session: ConversationSession):
        """Test adding a turn to the session."""
        turn = session.add_turn(
            question="What is Python?",
            answer="Python is a programming language",
            sources=[{"file": "test.txt"}],
            query_type="factoid",
        )

        assert turn.turn_number == 1
        assert turn.question == "What is Python?"
        assert turn.answer == "Python is a programming language"
        assert len(session.turns) == 1

    def test_add_multiple_turns(self, session: ConversationSession):
        """Test adding multiple turns."""
        session.add_turn("Q1?", "A1", [], "type1")
        session.add_turn("Q2?", "A2", [], "type2")
        session.add_turn("Q3?", "A3", [], "type3")

        assert len(session.turns) == 3
        assert session.turns[0].turn_number == 1
        assert session.turns[1].turn_number == 2
        assert session.turns[2].turn_number == 3

    def test_add_turn_with_no_answer(self, session: ConversationSession):
        """Test adding a turn with no answer."""
        turn = session.add_turn("What is X?", None, [], "unknown")
        assert turn.answer is None

    def test_turn_count_property(self, populated_session: ConversationSession):
        """Test turn_count property."""
        assert populated_session.turn_count == 2

    def test_turn_count_empty(self, session: ConversationSession):
        """Test turn_count on empty session."""
        assert session.turn_count == 0

    def test_summary_topic_from_first_question(
        self, populated_session: ConversationSession
    ):
        """Test summary_topic derives from first question."""
        assert populated_session.summary_topic == "What is Python?"

    def test_summary_topic_truncated(self, session: ConversationSession):
        """Test summary_topic truncates long questions."""
        long_q = "This is a very long question " * 10
        session.add_turn(long_q, "answer", [], "type")

        topic = session.summary_topic
        assert len(topic) <= 60
        assert topic.endswith("...")

    def test_summary_topic_empty_session(self, session: ConversationSession):
        """Test summary_topic on empty session."""
        assert session.summary_topic == "New conversation"


class TestContextGeneration:
    """Tests for LLM context generation."""

    def test_get_context_empty_session(self):
        """Test context generation with no turns."""
        session = ConversationSession()
        context = session.get_context_for_llm()
        assert context == ""

    def test_get_context_single_turn(self):
        """Test context generation with one turn."""
        session = ConversationSession(max_history_words=1000)
        session.add_turn("What is Python?", "Python is a language", [], "factoid")

        context = session.get_context_for_llm()
        assert "Q: What is Python?" in context
        assert "A: Python is a language" in context

    def test_get_context_multiple_turns(self):
        """Test context generation with multiple turns."""
        session = ConversationSession(max_history_words=1000)
        session.add_turn("Q1?", "A1", [], "type1")
        session.add_turn("Q2?", "A2", [], "type2")
        session.add_turn("Q3?", "A3", [], "type3")

        context = session.get_context_for_llm()
        assert "Q: Q1?" in context
        assert "Q: Q2?" in context
        assert "Q: Q3?" in context
        assert context.index("Q1") < context.index("Q2") < context.index("Q3")

    def test_get_context_respects_word_budget(self):
        """Test context generation respects word budget."""
        session = ConversationSession(max_history_words=20)

        # Add turns with long answers
        for i in range(5):
            session.add_turn(
                f"Question {i}?",
                "This is a long answer " * 20,  # 100 words
                [],
                "type",
            )

        context = session.get_context_for_llm()
        word_count = len(context.split())

        # Should not exceed budget significantly
        assert word_count <= 25  # Small margin for variation

    def test_get_context_prioritizes_recent_turns(self):
        """Test context generation prioritizes recent turns."""
        session = ConversationSession(max_history_words=30)
        session.add_turn("Old question?", "Old answer", [], "type")
        session.add_turn("Recent question?", "Recent answer", [], "type")

        context = session.get_context_for_llm()

        # Recent turn should be included
        assert "Recent question" in context

        # Old turn might be excluded due to budget
        # (but check at least recent is there)

    def test_get_context_with_no_answer(self):
        """Test context generation when turn has no answer."""
        session = ConversationSession(max_history_words=1000)
        session.add_turn("Question?", None, [], "type")

        context = session.get_context_for_llm()
        assert "Q: Question?" in context
        assert "A: (no answer)" in context

    def test_get_context_trims_long_answers(self):
        """Test context generation trims long answers."""
        session = ConversationSession(max_history_words=50)

        long_answer = "Sentence one. " * 20 + "Sentence two. " * 20
        session.add_turn("Q?", long_answer, [], "type")

        context = session.get_context_for_llm()

        # Answer should be trimmed
        assert "Q: Q?" in context
        assert len(context.split()) <= 55  # Small margin


class TestTrimSentenceBoundary:
    """Tests for sentence boundary trimming."""

    def test_trim_no_trimming_needed(self):
        """Test trim when text is within budget."""
        text = "Short text here."
        result = ConversationSession._trim_at_sentence_boundary(text, 100)
        assert result == text

    def test_trim_at_sentence_boundary(self):
        """Test trimming at sentence boundary."""
        text = "First sentence. Second sentence. Third sentence."
        result = ConversationSession._trim_at_sentence_boundary(text, 5)

        # Should include sentences that fit in budget
        assert "First sentence" in result
        # May include "Second sentence" if it fits (2 words + 2 words = 4 words total)
        # Third should not be included
        assert "Third sentence" not in result

    def test_trim_multiple_sentences_fit(self):
        """Test multiple sentences fit in budget."""
        text = "One. Two. Three. Four."
        result = ConversationSession._trim_at_sentence_boundary(text, 10)

        assert "One" in result
        # Exact count depends on splitting, but should fit multiple

    def test_trim_first_sentence_too_long(self):
        """Test trimming when even first sentence exceeds budget."""
        text = "This is a very long sentence with many words."
        result = ConversationSession._trim_at_sentence_boundary(text, 3)

        # Should truncate by words
        assert result.endswith("...")
        word_count = len(result.replace("...", "").split())
        assert word_count == 3

    def test_trim_preserves_complete_sentences(self):
        """Test trimming preserves sentence structure."""
        text = "Sentence one. Sentence two. Sentence three."
        result = ConversationSession._trim_at_sentence_boundary(text, 6)

        # Should end with period (complete sentence)
        assert not result.endswith("...")


class TestFollowUpDetection:
    """Tests for follow-up query detection."""

    def test_is_followup_empty_session(self):
        """Test follow-up detection with no history."""
        session = ConversationSession()
        assert not session.is_follow_up("What is Python?")

    def test_is_followup_short_query(self):
        """Test short queries are detected as follow-ups."""
        session = ConversationSession()
        session.add_turn("What is Python?", "A language", [], "factoid")

        assert session.is_follow_up("More details")
        assert session.is_follow_up("Why?")
        assert session.is_follow_up("How?")

    def test_is_followup_pronoun_references(self):
        """Test pronoun references indicate follow-ups."""
        session = ConversationSession()
        session.add_turn("What is Python?", "A language", [], "factoid")

        assert session.is_follow_up("Tell me more about it")
        assert session.is_follow_up("What about that feature?")
        assert session.is_follow_up("Explain this concept")
        assert session.is_follow_up("Those are interesting")

    def test_is_followup_followup_phrases(self):
        """Test follow-up phrases are detected."""
        session = ConversationSession()
        session.add_turn("What is Python?", "A language", [], "factoid")

        assert session.is_follow_up("Tell me more about Python")
        assert session.is_follow_up("What about the first point?")
        assert session.is_follow_up("Can you elaborate?")
        assert session.is_follow_up("Explain further please")
        assert session.is_follow_up("Go deeper into this topic")

    def test_is_followup_long_query_without_indicators(self):
        """Test long queries without indicators are not follow-ups."""
        session = ConversationSession()
        session.add_turn("What is Python?", "A language", [], "factoid")

        assert not session.is_follow_up(
            "What are the best practices for writing clean code?"
        )


class TestRetrievalAugmentation:
    """Tests for retrieval query augmentation."""

    def test_augmentation_empty_session(self):
        """Test augmentation with no history."""
        session = ConversationSession()
        query = "What is Python?"
        augmented = session.get_retrieval_augmentation(query)
        assert augmented == query

    def test_augmentation_extracts_keywords(self):
        """Test augmentation extracts keywords from history."""
        session = ConversationSession()
        session.add_turn(
            "What is Python programming?",
            "Python is a high-level language with dynamic typing.",
            [],
            "factoid",
        )

        augmented = session.get_retrieval_augmentation("How to start?")

        # Should include keywords from history
        # (keywords exclude stopwords)
        assert len(augmented) > len("How to start?")

    def test_augmentation_filters_stopwords(self):
        """Test augmentation filters out stopwords."""
        session = ConversationSession()
        session.add_turn("What is the Python?", "It is the language", [], "type")

        augmented = session.get_retrieval_augmentation("Query")

        # Should not include stopwords like "the", "is", "it"
        assert "the" not in augmented.lower().split()[:5]

    def test_augmentation_limits_keywords(self):
        """Test augmentation limits keywords to 5."""
        session = ConversationSession()

        # Add many keywords
        long_answer = " ".join([f"keyword{i}" for i in range(20)])
        session.add_turn("Q?", long_answer, [], "type")

        augmented = session.get_retrieval_augmentation("query")
        parts = augmented.split()

        # Should have query + up to 5 keywords
        assert len(parts) <= 6  # 5 keywords + "query"

    def test_augmentation_no_duplicate_keywords(self):
        """Test augmentation doesn't duplicate keywords."""
        session = ConversationSession()
        session.add_turn("Q?", "keyword1 keyword2 keyword3", [], "type")
        session.add_turn("Q2?", "keyword1 keyword4", [], "type")

        augmented = session.get_retrieval_augmentation("query")
        words = augmented.split()

        # Keywords should not be duplicated in the augmentation part
        # Count how many times each keyword appears
        keyword_counts = {}
        for word in words[:-1]:  # Exclude "query" at the end
            keyword_counts[word] = keyword_counts.get(word, 0) + 1

        # No keyword should appear more than once in augmentation
        for count in keyword_counts.values():
            assert count <= 1

    def test_augmentation_preserves_query(self):
        """Test augmentation preserves original query."""
        session = ConversationSession()
        session.add_turn("Q?", "Some answer here", [], "type")

        original = "my query here"
        augmented = session.get_retrieval_augmentation(original)

        assert original in augmented


class TestSerialization:
    """Tests for session serialization."""

    def test_to_dict_empty(self):
        """Test serializing empty session."""
        session = ConversationSession()
        data = session.to_dict()

        assert "session_id" in data
        assert "created_at" in data
        assert "max_history_words" in data
        assert data["turns"] == []

    def test_to_dict_with_turns(self):
        """Test serializing session with turns."""
        session = ConversationSession()
        session.add_turn("Q1?", "A1", [{"file": "test.txt"}], "factoid")
        session.add_turn("Q2?", "A2", [], "list")

        data = session.to_dict()

        assert len(data["turns"]) == 2
        assert data["turns"][0]["question"] == "Q1?"
        assert data["turns"][0]["answer"] == "A1"
        assert data["turns"][1]["question"] == "Q2?"

    def test_from_dict_empty(self):
        """Test deserializing empty session."""
        data = {
            "session_id": "test123",
            "created_at": 123.45,
            "max_history_words": 500,
            "turns": [],
        }

        session = ConversationSession.from_dict(data)

        assert session.session_id == "test123"
        assert session.created_at == 123.45
        assert session.max_history_words == 500
        assert len(session.turns) == 0

    def test_from_dict_with_turns(self):
        """Test deserializing session with turns."""
        data = {
            "session_id": "test123",
            "created_at": 123.45,
            "max_history_words": 500,
            "turns": [
                {
                    "turn_number": 1,
                    "question": "Q1?",
                    "answer": "A1",
                    "sources": [{"file": "test.txt"}],
                    "query_type": "factoid",
                    "timestamp": 100.0,
                },
                {
                    "turn_number": 2,
                    "question": "Q2?",
                    "answer": None,
                    "sources": [],
                    "query_type": "unknown",
                    "timestamp": 101.0,
                },
            ],
        }

        session = ConversationSession.from_dict(data)

        assert len(session.turns) == 2
        assert session.turns[0].question == "Q1?"
        assert session.turns[0].answer == "A1"
        assert session.turns[1].answer is None

    def test_serialization_roundtrip(self):
        """Test serialize and deserialize preserves data."""
        original = ConversationSession()
        original.add_turn("Q1?", "A1", [{"file": "f1"}], "type1")
        original.add_turn("Q2?", "A2", [{"file": "f2"}], "type2")

        data = original.to_dict()
        restored = ConversationSession.from_dict(data)

        assert restored.session_id == original.session_id
        assert restored.turn_count == original.turn_count
        assert restored.turns[0].question == original.turns[0].question
        assert restored.turns[1].answer == original.turns[1].answer


class TestExportMarkdown:
    """Tests for markdown export."""

    def test_export_markdown_empty(self):
        """Test markdown export of empty session."""
        session = ConversationSession()
        markdown = session.export_markdown()

        assert "Conversation:" in markdown
        assert "Session:" in markdown
        assert "Turns: 0" in markdown

    def test_export_markdown_with_turns(self):
        """Test markdown export with turns."""
        session = ConversationSession()
        session.add_turn("Q1?", "A1 here", [{"source_file": "f1.txt"}], "type1")
        session.add_turn("Q2?", "A2 here", [], "type2")

        markdown = session.export_markdown()

        assert "## Q1: Q1?" in markdown
        assert "A1 here" in markdown
        assert "## Q2: Q2?" in markdown
        assert "A2 here" in markdown

    def test_export_markdown_with_sources(self):
        """Test markdown export includes sources."""
        session = ConversationSession()
        session.add_turn(
            "Question?",
            "Answer",
            [{"section_title": "Chapter 1"}, {"source_file": "doc.txt"}],
            "type",
        )

        markdown = session.export_markdown()

        assert "**Sources:**" in markdown
        assert "[1] Chapter 1" in markdown
        assert "[2] doc.txt" in markdown

    def test_export_markdown_no_answer(self):
        """Test markdown export when turn has no answer."""
        session = ConversationSession()
        session.add_turn("Question?", None, [], "type")

        markdown = session.export_markdown()

        assert "*(no answer generated)*" in markdown


class TestHelperMethods:
    """Tests for internal helper methods."""

    def test_get_stopwords(self):
        """Test stopwords set contains common words."""
        session = ConversationSession()
        stopwords = session._get_stopwords()

        assert "the" in stopwords
        assert "is" in stopwords
        assert "and" in stopwords
        assert "to" in stopwords

    def test_extract_words_from_text(self):
        """Test word extraction from text."""
        session = ConversationSession()
        words = session._extract_words_from_text("The Python programming language")

        # Should extract words, lowercase
        assert "python" in words
        assert "programming" in words
        assert "language" in words

    def test_extract_words_filters_short(self):
        """Test word extraction filters short words."""
        session = ConversationSession()
        words = session._extract_words_from_text("I am a Python programmer")

        # Short words (< 3 chars) should be excluded
        assert "am" not in words
        assert "python" in words

    def test_filter_keywords_removes_stopwords(self):
        """Test keyword filtering removes stopwords."""
        session = ConversationSession()
        words = ["the", "python", "is", "great"]
        stopwords = {"the", "is"}

        keywords = session._filter_keywords(words, stopwords, [])

        assert "python" in keywords
        assert "great" in keywords
        assert "the" not in keywords
        assert "is" not in keywords

    def test_filter_keywords_removes_duplicates(self):
        """Test keyword filtering removes duplicates."""
        session = ConversationSession()
        words = ["python", "java", "python"]
        existing = ["python"]

        keywords = session._filter_keywords(words, set(), existing)

        assert "java" in keywords
        assert keywords.count("python") == 0  # Already in existing

    def test_collect_keywords_from_turns(self):
        """Test collecting keywords from conversation history."""
        session = ConversationSession()
        session.add_turn("What is Python?", "Python is great", [], "type")
        session.add_turn("How to learn?", "Start with tutorials", [], "type")

        keywords = session._collect_keywords_from_turns()

        # Should extract meaningful keywords
        assert "python" in keywords
        assert "tutorials" in keywords

    def test_collect_keywords_safety_limit(self):
        """Test keyword collection respects safety limit."""
        session = ConversationSession()

        # Mock having many turns (more than MAX_TURNS)
        for i in range(1500):
            session.turns.append(
                ConversationTurn(i + 1, f"Q{i}", f"A{i}", [], "type", time.time())
            )

        # Should not hang or crash
        keywords = session._collect_keywords_from_turns()

        # Should still return keywords
        assert isinstance(keywords, list)


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_question(self):
        """Test handling empty question."""
        session = ConversationSession()
        turn = session.add_turn("", "answer", [], "type")
        assert turn.question == ""

    def test_empty_answer(self):
        """Test handling empty answer."""
        session = ConversationSession()
        turn = session.add_turn("question", "", [], "type")
        assert turn.answer == ""

    def test_very_small_word_budget(self):
        """Test context generation with tiny budget."""
        session = ConversationSession(max_history_words=1)
        session.add_turn("Question?", "Answer", [], "type")

        context = session.get_context_for_llm()

        # Should handle gracefully
        assert isinstance(context, str)

    def test_unicode_handling(self):
        """Test handling unicode characters."""
        session = ConversationSession()
        session.add_turn("Qu\u00e9 es Python?", "Python es un lenguaje", [], "type")

        context = session.get_context_for_llm()
        assert "Python" in context

    def test_special_characters(self):
        """Test handling special characters."""
        session = ConversationSession()
        session.add_turn("What is @Python?", "Python is #1!", [], "type")

        keywords = session._collect_keywords_from_turns()
        assert "python" in keywords
