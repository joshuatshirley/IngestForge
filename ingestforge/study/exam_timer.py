"""Exam timer - Countdown timer for timed quiz sessions.

QUIZ-001.1: Timed Session Loop with per-question time limits.

Uses Rich Live display for real-time countdown visualization."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Callable
from rich.panel import Panel
from rich.text import Text


@dataclass
class TimerState:
    """Represents the current state of an exam timer.

    Attributes:
        total_seconds: Total time allowed
        elapsed_seconds: Time elapsed
        is_running: Whether timer is active
        is_expired: Whether time has run out
        start_time: When the timer started
    """

    total_seconds: int
    elapsed_seconds: float = 0.0
    is_running: bool = False
    is_expired: bool = False
    start_time: Optional[datetime] = None


class ExamTimer:
    """Countdown timer for exam sessions.

    Provides a visual countdown timer that can be displayed
    alongside quiz questions using Rich Live display.

    Attributes:
        time_limit: Time limit per question in seconds
        warning_threshold: Seconds remaining to show warning color
        on_expire: Optional callback when timer expires
    """

    DEFAULT_TIME_LIMIT = 60  # seconds
    DEFAULT_WARNING = 10  # seconds

    def __init__(
        self,
        time_limit: int = DEFAULT_TIME_LIMIT,
        warning_threshold: int = DEFAULT_WARNING,
        on_expire: Optional[Callable[[], None]] = None,
    ) -> None:
        """Initialize exam timer.

        Args:
            time_limit: Time limit per question in seconds
            warning_threshold: Seconds to show warning color
            on_expire: Callback when timer expires
        """
        self.time_limit = max(1, time_limit)
        self.warning_threshold = max(1, min(warning_threshold, time_limit - 1))
        self.on_expire = on_expire
        self._state = TimerState(total_seconds=time_limit)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()

    @property
    def remaining_seconds(self) -> float:
        """Get remaining time in seconds."""
        with self._lock:
            return max(0, self._state.total_seconds - self._state.elapsed_seconds)

    @property
    def is_expired(self) -> bool:
        """Check if timer has expired."""
        with self._lock:
            return self._state.is_expired

    @property
    def is_running(self) -> bool:
        """Check if timer is currently running."""
        with self._lock:
            return self._state.is_running

    def start(self) -> None:
        """Start the countdown timer."""
        with self._lock:
            if self._state.is_running:
                return
            self._state.is_running = True
            self._state.start_time = datetime.now()
            self._stop_event.clear()

    def stop(self) -> None:
        """Stop the countdown timer."""
        with self._lock:
            self._update_elapsed()
            self._state.is_running = False
            self._stop_event.set()

    def reset(self, new_time_limit: Optional[int] = None) -> None:
        """Reset the timer for a new question.

        Args:
            new_time_limit: Optional new time limit
        """
        with self._lock:
            if new_time_limit is not None:
                self.time_limit = max(1, new_time_limit)
            self._state = TimerState(total_seconds=self.time_limit)
            self._stop_event.clear()

    def _update_elapsed(self) -> None:
        """Update elapsed time from start time."""
        if self._state.start_time is not None and self._state.is_running:
            delta = datetime.now() - self._state.start_time
            self._state.elapsed_seconds = delta.total_seconds()

            # Check for expiry
            if self._state.elapsed_seconds >= self._state.total_seconds:
                self._state.is_expired = True
                self._state.is_running = False

    def tick(self) -> TimerState:
        """Update timer and return current state.

        Returns:
            Current timer state
        """
        with self._lock:
            self._update_elapsed()

            # Trigger callback on expiry
            if self._state.is_expired and self.on_expire is not None:
                try:
                    self.on_expire()
                except Exception as e:
                    from ingestforge.core.logging import get_logger

                    get_logger(__name__).debug(f"Expiry callback failed: {e}")

            return TimerState(
                total_seconds=self._state.total_seconds,
                elapsed_seconds=self._state.elapsed_seconds,
                is_running=self._state.is_running,
                is_expired=self._state.is_expired,
                start_time=self._state.start_time,
            )

    def get_display(self) -> Panel:
        """Get Rich Panel showing current timer state.

        Returns:
            Rich Panel with timer display
        """
        state = self.tick()
        remaining = max(0, state.total_seconds - state.elapsed_seconds)

        # Format time
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        # Determine color based on remaining time
        if state.is_expired:
            color = "red"
            status = "TIME'S UP!"
        elif remaining <= self.warning_threshold:
            color = "yellow"
            status = time_str
        else:
            color = "green"
            status = time_str

        text = Text()
        text.append("⏱  ", style="bold")
        text.append(status, style=f"bold {color}")

        return Panel(
            text,
            title="[bold]Time Remaining[/bold]",
            border_style=color,
            width=20,
        )

    def format_remaining(self) -> str:
        """Get formatted remaining time string.

        Returns:
            Time string like "01:30"
        """
        remaining = self.remaining_seconds
        minutes = int(remaining // 60)
        seconds = int(remaining % 60)
        return f"{minutes:02d}:{seconds:02d}"


def create_timer_display(
    timer: ExamTimer,
    question_text: str,
    question_number: int,
    total_questions: int,
) -> Panel:
    """Create combined timer and question display.

    Args:
        timer: ExamTimer instance
        question_text: Current question text
        question_number: Current question number
        total_questions: Total number of questions

    Returns:
        Rich Panel with combined display
    """
    state = timer.tick()
    remaining = max(0, state.total_seconds - state.elapsed_seconds)

    # Timer formatting
    minutes = int(remaining // 60)
    seconds = int(remaining % 60)
    time_str = f"{minutes:02d}:{seconds:02d}"

    # Color based on time
    if state.is_expired:
        timer_color = "red"
        time_str = "TIME'S UP!"
    elif remaining <= timer.warning_threshold:
        timer_color = "yellow"
    else:
        timer_color = "green"

    # Build display
    text = Text()
    text.append(f"Question {question_number}/{total_questions}\n\n", style="bold cyan")
    text.append(f"{question_text}\n\n", style="white")
    text.append("Time: ", style="dim")
    text.append(time_str, style=f"bold {timer_color}")

    return Panel(
        text,
        title="[bold]Exam Mode[/bold]",
        border_style="blue",
    )


class TimedSession:
    """Manages a timed exam session with multiple questions.

    Attributes:
        time_per_question: Seconds per question
        questions: List of question texts
        current_index: Index of current question
    """

    def __init__(
        self,
        time_per_question: int = ExamTimer.DEFAULT_TIME_LIMIT,
        warning_threshold: int = ExamTimer.DEFAULT_WARNING,
    ) -> None:
        """Initialize timed session.

        Args:
            time_per_question: Seconds allowed per question
            warning_threshold: Seconds to show warning
        """
        self.time_per_question = time_per_question
        self.warning_threshold = warning_threshold
        self.timer = ExamTimer(time_per_question, warning_threshold)
        self.questions: list[str] = []
        self.answers: list[Optional[str]] = []
        self.times_taken: list[float] = []
        self.current_index = 0
        self._start_time: Optional[datetime] = None

    def add_question(self, question: str) -> None:
        """Add a question to the session.

        Args:
            question: Question text
        """
        self.questions.append(question)
        self.answers.append(None)

    def start_question(self) -> bool:
        """Start timer for current question.

        Returns:
            True if question started, False if no more questions
        """
        if self.current_index >= len(self.questions):
            return False

        self.timer.reset()
        self.timer.start()
        self._start_time = datetime.now()
        return True

    def submit_answer(self, answer: str) -> float:
        """Submit answer for current question.

        Args:
            answer: User's answer

        Returns:
            Time taken in seconds
        """
        self.timer.stop()

        time_taken = 0.0
        if self._start_time is not None:
            time_taken = (datetime.now() - self._start_time).total_seconds()

        if self.current_index < len(self.answers):
            self.answers[self.current_index] = answer
            self.times_taken.append(time_taken)

        return time_taken

    def next_question(self) -> bool:
        """Move to next question.

        Returns:
            True if moved to next, False if at end
        """
        if self.current_index >= len(self.questions) - 1:
            return False

        self.current_index += 1
        return True

    @property
    def current_question(self) -> Optional[str]:
        """Get current question text."""
        if self.current_index < len(self.questions):
            return self.questions[self.current_index]
        return None

    @property
    def is_complete(self) -> bool:
        """Check if all questions answered."""
        return self.current_index >= len(self.questions) - 1 and len(
            self.times_taken
        ) >= len(self.questions)

    def get_session_summary(self) -> dict:
        """Get session summary statistics.

        Returns:
            Dictionary with session stats
        """
        total_time = sum(self.times_taken)
        avg_time = total_time / len(self.times_taken) if self.times_taken else 0

        return {
            "total_questions": len(self.questions),
            "questions_answered": len([a for a in self.answers if a is not None]),
            "total_time_seconds": round(total_time, 1),
            "average_time_seconds": round(avg_time, 1),
            "time_per_question": self.times_taken,
        }
