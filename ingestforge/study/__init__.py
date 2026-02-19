"""Study utilities package.

Provides core utilities for study functionality:
- due_check: Lightweight due cards notification system (SRS-004)
- session_tracker: Review session tracking with undo (SRS-002.2)
- exam_timer: Countdown timer for timed sessions (QUIZ-001.1)
- grading: Deferred grading engine (QUIZ-001.2)
- grading_semantic: Semantic answer scoring (NLP-002.1)
- stats: Study statistics aggregator (UX-019.1)
- explanations: LLM-powered answer explanations (QUIZ-002.1)
- related_chunks: Semantic context linker (QUIZ-002.2)
- scheduler: SM-2 spaced repetition algorithm (TICKET-102)
"""

from __future__ import annotations

from ingestforge.study.session_tracker import (
    SessionTracker,
    ReviewAction,
)

from ingestforge.study.exam_timer import (
    ExamTimer,
    TimerState,
    TimedSession,
)

from ingestforge.study.grading import (
    GradingEngine,
    GradeReport,
    QuestionResult,
    LetterGrade,
    calculate_grade,
    format_grade_display,
    get_grade_color,
)

from ingestforge.study.stats import (
    StudyStats,
    TopicStats,
    MasteryLevel,
    StatsAggregator,
    get_study_stats,
)

from ingestforge.study.explanations import (
    Explanation,
    ExplanationRequest,
    ExplanationGenerator,
    explain_answer,
)

from ingestforge.study.related_chunks import (
    RelatedChunk,
    RelatedChunksResult,
    RelatedChunksLinker,
    find_related_chunks,
)

from ingestforge.study.grading_semantic import (
    GradeLevel,
    SemanticScore,
    GradingThresholds,
    SemanticGrader,
    grade_answer_semantic,
)

from ingestforge.study.scheduler import (
    calculate_next_interval,
    get_review_count,
    MIN_EASE_FACTOR,
    FIRST_SUCCESS_INTERVAL,
    SECOND_SUCCESS_INTERVAL,
    FAIL_RESET_INTERVAL,
)

__all__ = [
    # Due check (SRS-004)
    "due_check",
    # Session tracker (SRS-002.2)
    "SessionTracker",
    "ReviewAction",
    # Exam timer (QUIZ-001.1)
    "ExamTimer",
    "TimerState",
    "TimedSession",
    # Grading (QUIZ-001.2)
    "GradingEngine",
    "GradeReport",
    "QuestionResult",
    "LetterGrade",
    "calculate_grade",
    "format_grade_display",
    "get_grade_color",
    # Stats (UX-019.1)
    "StudyStats",
    "TopicStats",
    "MasteryLevel",
    "StatsAggregator",
    "get_study_stats",
    # Explanations (QUIZ-002.1)
    "Explanation",
    "ExplanationRequest",
    "ExplanationGenerator",
    "explain_answer",
    # Related chunks (QUIZ-002.2)
    "RelatedChunk",
    "RelatedChunksResult",
    "RelatedChunksLinker",
    "find_related_chunks",
    # Semantic grading (NLP-002.1)
    "GradeLevel",
    "SemanticScore",
    "GradingThresholds",
    "SemanticGrader",
    "grade_answer_semantic",
    # Scheduler (TICKET-102)
    "calculate_next_interval",
    "get_review_count",
    "MIN_EASE_FACTOR",
    "FIRST_SUCCESS_INTERVAL",
    "SECOND_SUCCESS_INTERVAL",
    "FAIL_RESET_INTERVAL",
]
