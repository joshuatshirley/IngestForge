"""UI components for IngestForge CLI.

This package provides reusable UI components for the CLI:
- tips: Contextual post-action suggestions (UX-016)
- summary_panel: Study session summary display (SRS-003)
- rating_prompt: Difficulty rating input with color feedback (SRS-002.1)
- dashboard: Study progress dashboard (UX-019.2)
- quiz_feedback: Semantic answer feedback display (NLP-002.2)

Usage:
    from ingestforge.cli.ui.tips import show_tip, show_tips_panel
    from ingestforge.cli.ui.summary_panel import SessionStats, show_session_summary
    from ingestforge.cli.ui.rating_prompt import prompt_rating, Rating, RATINGS
    from ingestforge.cli.ui.dashboard import show_dashboard
    from ingestforge.cli.ui.quiz_feedback import show_answer_feedback, get_feedback_panel
"""

from __future__ import annotations

from ingestforge.cli.ui.tips import (
    show_tip,
    show_tips_panel,
    get_tips_for_command,
)

from ingestforge.cli.ui.summary_panel import (
    SessionStats,
    show_session_summary,
    get_motivational_message,
    format_duration,
)

from ingestforge.cli.ui.rating_prompt import (
    Rating,
    RATINGS,
    SHORTCUTS,
    prompt_rating,
    display_rating_feedback,
    get_sm2_quality,
    get_rating_prompt_panel,
    get_rating_stats_display,
)

from ingestforge.cli.ui.dashboard import (
    show_dashboard,
    create_overview_panel,
    create_mastery_chart,
    create_topics_table,
    get_dashboard_summary,
)

from ingestforge.cli.ui.quiz_feedback import (
    FeedbackStyle,
    GRADE_COLORS,
    GRADE_EMOJI,
    get_grade_color,
    get_grade_indicator,
    create_similarity_bar,
    get_similarity_color,
    format_similarity_display,
    get_feedback_panel,
    show_answer_feedback,
    show_similarity_bar,
    get_comparison_table,
    format_score_summary,
    show_batch_results,
)

__all__ = [
    # Tips (UX-016)
    "show_tip",
    "show_tips_panel",
    "get_tips_for_command",
    # Summary panel (SRS-003)
    "SessionStats",
    "show_session_summary",
    "get_motivational_message",
    "format_duration",
    # Rating prompt (SRS-002.1)
    "Rating",
    "RATINGS",
    "SHORTCUTS",
    "prompt_rating",
    "display_rating_feedback",
    "get_sm2_quality",
    "get_rating_prompt_panel",
    "get_rating_stats_display",
    # Dashboard (UX-019.2)
    "show_dashboard",
    "create_overview_panel",
    "create_mastery_chart",
    "create_topics_table",
    "get_dashboard_summary",
    # Quiz feedback (NLP-002.2)
    "FeedbackStyle",
    "GRADE_COLORS",
    "GRADE_EMOJI",
    "get_grade_color",
    "get_grade_indicator",
    "create_similarity_bar",
    "get_similarity_color",
    "format_similarity_display",
    "get_feedback_panel",
    "show_answer_feedback",
    "show_similarity_bar",
    "get_comparison_table",
    "format_score_summary",
    "show_batch_results",
]
