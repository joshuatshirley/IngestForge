"""Tests for exam_timer module (QUIZ-001.1).

Tests the countdown timer functionality:
- Timer state management
- Start/stop/reset operations
- Time remaining calculation
- Display generation
- Timed session management
"""

import time
from rich.panel import Panel

from ingestforge.study.exam_timer import (
    ExamTimer,
    TimerState,
    TimedSession,
    create_timer_display,
)


class TestTimerState:
    """Test TimerState dataclass."""

    def test_state_creation(self) -> None:
        """TimerState should be creatable."""
        state = TimerState(total_seconds=60)
        assert state.total_seconds == 60
        assert state.elapsed_seconds == 0.0
        assert state.is_running is False
        assert state.is_expired is False


class TestExamTimer:
    """Test ExamTimer class."""

    def test_timer_creation(self) -> None:
        """Timer should initialize with defaults."""
        timer = ExamTimer()
        assert timer.time_limit == ExamTimer.DEFAULT_TIME_LIMIT
        assert timer.is_running is False
        assert timer.is_expired is False

    def test_custom_time_limit(self) -> None:
        """Timer should accept custom time limit."""
        timer = ExamTimer(time_limit=120)
        assert timer.time_limit == 120

    def test_minimum_time_limit(self) -> None:
        """Time limit should be at least 1 second."""
        timer = ExamTimer(time_limit=0)
        assert timer.time_limit == 1

    def test_start(self) -> None:
        """start() should set timer running."""
        timer = ExamTimer()
        timer.start()
        assert timer.is_running is True

    def test_stop(self) -> None:
        """stop() should pause the timer."""
        timer = ExamTimer()
        timer.start()
        timer.stop()
        assert timer.is_running is False

    def test_reset(self) -> None:
        """reset() should clear timer state."""
        timer = ExamTimer(time_limit=30)
        timer.start()
        time.sleep(0.1)
        timer.reset()
        assert timer.is_running is False
        assert timer.remaining_seconds == 30

    def test_reset_with_new_limit(self) -> None:
        """reset() can set new time limit."""
        timer = ExamTimer(time_limit=30)
        timer.reset(new_time_limit=60)
        assert timer.time_limit == 60


class TestTimerRemainingTime:
    """Test remaining time calculation."""

    def test_remaining_equals_total_initially(self) -> None:
        """Remaining should equal total before start."""
        timer = ExamTimer(time_limit=60)
        assert timer.remaining_seconds == 60

    def test_remaining_decreases_after_start(self) -> None:
        """Remaining should decrease while running."""
        timer = ExamTimer(time_limit=60)
        timer.start()
        time.sleep(0.5)  # Longer sleep for reliable timing
        timer.tick()  # Force state update
        remaining = timer.remaining_seconds
        assert remaining < 60
        timer.stop()

    def test_remaining_not_negative(self) -> None:
        """Remaining should never be negative."""
        timer = ExamTimer(time_limit=1)
        timer.start()
        time.sleep(1.5)
        assert timer.remaining_seconds >= 0


class TestTimerExpiry:
    """Test timer expiry behavior."""

    def test_timer_expires(self) -> None:
        """Timer should expire when time runs out."""
        timer = ExamTimer(time_limit=1)
        timer.start()
        time.sleep(1.2)
        timer.tick()  # Update state
        assert timer.is_expired is True

    def test_expiry_callback(self) -> None:
        """on_expire callback should be called."""
        callback_called = [False]

        def on_expire() -> None:
            callback_called[0] = True

        timer = ExamTimer(time_limit=1, on_expire=on_expire)
        timer.start()
        time.sleep(1.2)
        timer.tick()

        assert callback_called[0] is True


class TestTimerDisplay:
    """Test timer display generation."""

    def test_get_display_returns_panel(self) -> None:
        """get_display should return Rich Panel."""
        timer = ExamTimer()
        panel = timer.get_display()
        assert isinstance(panel, Panel)

    def test_format_remaining(self) -> None:
        """format_remaining should return time string."""
        timer = ExamTimer(time_limit=90)
        formatted = timer.format_remaining()
        assert formatted == "01:30"

    def test_format_remaining_under_minute(self) -> None:
        """format_remaining works for short times."""
        timer = ExamTimer(time_limit=45)
        formatted = timer.format_remaining()
        assert formatted == "00:45"


class TestCreateTimerDisplay:
    """Test create_timer_display helper."""

    def test_creates_panel(self) -> None:
        """Should create combined display panel."""
        timer = ExamTimer()
        panel = create_timer_display(
            timer,
            question_text="What is 2+2?",
            question_number=1,
            total_questions=5,
        )
        assert isinstance(panel, Panel)


class TestTimedSession:
    """Test TimedSession class."""

    def test_session_creation(self) -> None:
        """Session should initialize properly."""
        session = TimedSession(time_per_question=30)
        assert session.time_per_question == 30
        assert session.current_index == 0

    def test_add_question(self) -> None:
        """add_question should store questions."""
        session = TimedSession()
        session.add_question("Q1?")
        session.add_question("Q2?")
        assert len(session.questions) == 2

    def test_current_question(self) -> None:
        """current_question should return active question."""
        session = TimedSession()
        session.add_question("What is 2+2?")
        assert session.current_question == "What is 2+2?"

    def test_start_question(self) -> None:
        """start_question should start timer."""
        session = TimedSession()
        session.add_question("Q1?")
        result = session.start_question()
        assert result is True
        assert session.timer.is_running is True
        session.timer.stop()

    def test_start_question_no_questions(self) -> None:
        """start_question returns False when no questions."""
        session = TimedSession()
        result = session.start_question()
        assert result is False

    def test_submit_answer(self) -> None:
        """submit_answer should record answer."""
        session = TimedSession()
        session.add_question("Q1?")
        session.start_question()
        time.sleep(0.1)
        time_taken = session.submit_answer("4")

        assert session.answers[0] == "4"
        assert time_taken > 0

    def test_next_question(self) -> None:
        """next_question should advance index."""
        session = TimedSession()
        session.add_question("Q1?")
        session.add_question("Q2?")
        session.start_question()
        session.submit_answer("A1")

        result = session.next_question()
        assert result is True
        assert session.current_index == 1
        assert session.current_question == "Q2?"

    def test_next_question_at_end(self) -> None:
        """next_question returns False at end."""
        session = TimedSession()
        session.add_question("Q1?")
        result = session.next_question()
        assert result is False

    def test_is_complete(self) -> None:
        """is_complete should check all answered."""
        session = TimedSession()
        session.add_question("Q1?")
        session.start_question()
        session.submit_answer("A1")

        assert session.is_complete is True

    def test_get_session_summary(self) -> None:
        """get_session_summary should return stats."""
        session = TimedSession()
        session.add_question("Q1?")
        session.start_question()
        time.sleep(0.1)
        session.submit_answer("A1")

        summary = session.get_session_summary()
        assert summary["total_questions"] == 1
        assert summary["questions_answered"] == 1
        assert summary["total_time_seconds"] > 0
