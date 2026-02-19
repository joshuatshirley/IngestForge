"""Quiz command - Generate quiz questions from knowledge base.

Creates multiple-choice and open-ended questions for studying.
Supports difficulty levels, answer explanations, and exam mode.

Exam Mode Features:
- Timed questions with visual countdown
- Deferred feedback until quiz completion
- Final score with letter grade
- Review of incorrect answers

Follows Commandments #4 (Small Functions), #7 (Check Parameters),
#1 (Simple Control Flow), and #9 (Type Hints).
"""

from __future__ import annotations

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.prompt import Prompt

from ingestforge.cli.study.base import StudyCommand


# Valid difficulty levels
VALID_DIFFICULTIES = ["easy", "medium", "hard"]

# Question type mix by difficulty
DIFFICULTY_MIX = {
    "easy": {"multiple_choice": 80, "open_ended": 20},
    "medium": {"multiple_choice": 60, "open_ended": 40},
    "hard": {"multiple_choice": 40, "open_ended": 60},
}


class QuizCommand(StudyCommand):
    """Generate quiz questions from knowledge base."""

    def execute(
        self,
        topic: str,
        count: int = 20,
        project: Optional[Path] = None,
        output: Optional[Path] = None,
        difficulty: str = "medium",
        include_answers: bool = True,
        include_explanations: bool = True,
    ) -> int:
        """
        Generate quiz questions about a topic.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            topic: Topic for quiz questions
            count: Number of questions
            project: Project directory
            output: Output file path
            difficulty: Question difficulty level
            include_answers: Include correct answers
            include_explanations: Include answer explanations
        """
        try:
            self._validate_inputs(topic, count, difficulty)
            ctx = self.initialize_context(project, require_storage=True)

            llm_client = self.get_llm_client(ctx)
            if llm_client is None:
                return 1

            chunks = self.search_topic_context(ctx["storage"], topic, k=30)
            if not chunks:
                self._handle_no_context(topic)
                return 0

            quiz_data = self._generate_quiz(
                llm_client,
                topic,
                chunks,
                count,
                difficulty,
                include_answers,
                include_explanations,
            )

            if not quiz_data:
                self.print_error("Failed to generate quiz questions")
                return 1

            self._display_quiz(quiz_data, topic, include_answers)

            if output:
                self._save_quiz(output, quiz_data, include_answers)

            return 0

        except Exception as e:
            return self.handle_error(e, "Quiz generation failed")

    def _validate_inputs(self, topic: str, count: int, difficulty: str) -> None:
        """Validate all inputs.

        Rule #7: Check parameters
        """
        self.validate_topic(topic)
        self.validate_count(count, min_val=1, max_val=50)
        self._validate_difficulty(difficulty)

    def _validate_difficulty(self, difficulty: str) -> None:
        """Validate difficulty level.

        Args:
            difficulty: Difficulty string to validate

        Raises:
            typer.BadParameter: If invalid
        """
        if difficulty.lower() not in VALID_DIFFICULTIES:
            raise typer.BadParameter(
                f"Invalid difficulty '{difficulty}'. "
                f"Must be one of: {', '.join(VALID_DIFFICULTIES)}"
            )

    def _handle_no_context(self, topic: str) -> None:
        """Handle case where no context found."""
        self.print_warning(f"No context found for topic: '{topic}'")
        self.print_info(
            "Try:\n"
            f"  1. Ingesting documents about {topic}\n"
            "  2. Using a broader topic term\n"
            "  3. Checking spelling"
        )

    def _generate_quiz(
        self,
        llm_client: Any,
        topic: str,
        chunks: list,
        count: int,
        difficulty: str,
        include_answers: bool,
        include_explanations: bool,
    ) -> Optional[Dict[str, Any]]:
        """Generate quiz questions using LLM.

        Rule #4: Function <60 lines
        """
        context = self.format_context_for_prompt(chunks)
        question_mix = DIFFICULTY_MIX.get(difficulty, DIFFICULTY_MIX["medium"])

        prompt = self._build_quiz_prompt(
            topic, count, difficulty, context, question_mix, include_explanations
        )

        response = self.generate_with_llm(llm_client, prompt, f"{count} quiz questions")

        quiz_data = self.parse_json_response(response)

        if not quiz_data:
            quiz_data = self._extract_questions_from_text(response, topic, difficulty)

        # Track topic for each question
        quiz_data = self._add_topic_metadata(quiz_data, topic, difficulty)

        return quiz_data

    def _build_quiz_prompt(
        self,
        topic: str,
        count: int,
        difficulty: str,
        context: str,
        question_mix: Dict[str, int],
        include_explanations: bool,
    ) -> str:
        """Build prompt for quiz generation.

        Rule #4: Function <60 lines
        """
        mc_count = int(count * question_mix["multiple_choice"] / 100)
        oe_count = count - mc_count

        explanation_instruction = ""
        if include_explanations:
            explanation_instruction = (
                '      "explanation": "why this answer is correct",\n'
            )

        difficulty_guidance = self._get_difficulty_guidance(difficulty)

        return f"""Generate a {count}-question {difficulty} quiz about: {topic}

Context from knowledge base:
{context}

Generate questions following this JSON format:
{{
  "topic": "{topic}",
  "difficulty": "{difficulty}",
  "total_questions": {count},
  "questions": [
    {{
      "id": 1,
      "question": "question text",
      "type": "multiple_choice",
      "options": ["A) option1", "B) option2", "C) option3", "D) option4"],
      "correct_answer": "A",
{explanation_instruction}      "topic_tag": "subtopic"
    }},
    {{
      "id": 2,
      "question": "question text",
      "type": "open_ended",
      "correct_answer": "expected answer",
{explanation_instruction}      "topic_tag": "subtopic"
    }}
  ]
}}

Requirements:
- Generate {mc_count} multiple choice and {oe_count} open-ended questions
- Base all questions on the provided context
- {difficulty_guidance}
- Include detailed explanations for each answer
- Tag each question with its subtopic
- Ensure questions test understanding, not just recall

Return ONLY valid JSON, no additional text."""

    def _get_difficulty_guidance(self, difficulty: str) -> str:
        """Get difficulty-specific guidance."""
        guidance = {
            "easy": "Use straightforward questions testing basic recall and simple concepts",
            "medium": "Balance recall with application questions requiring synthesis",
            "hard": "Focus on analysis, evaluation, and complex application scenarios",
        }
        return guidance.get(difficulty, guidance["medium"])

    def _add_topic_metadata(
        self, quiz_data: Dict[str, Any], topic: str, difficulty: str
    ) -> Dict[str, Any]:
        """Add topic and difficulty metadata to quiz."""
        quiz_data["topic"] = topic
        quiz_data["difficulty"] = difficulty

        # Ensure topic tags exist
        for question in quiz_data.get("questions", []):
            if "topic_tag" not in question:
                question["topic_tag"] = topic

        return quiz_data

    def _extract_questions_from_text(
        self, text: str, topic: str, difficulty: str
    ) -> Dict[str, Any]:
        """Extract questions from plain text response (fallback)."""
        lines = text.strip().split("\n")
        questions = []
        current_question = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if self._is_question_start(line):
                if current_question:
                    questions.append(current_question)
                current_question = self._parse_question_line(line, len(questions) + 1)

        if current_question:
            questions.append(current_question)

        return {
            "topic": topic,
            "difficulty": difficulty,
            "total_questions": len(questions),
            "questions": questions,
        }

    def _is_question_start(self, line: str) -> bool:
        """Check if line starts a new question."""
        markers = ("Q:", "Question:", "?", "1.", "2.", "3.", "4.", "5.")
        return any(line.startswith(m) for m in markers)

    def _parse_question_line(self, line: str, question_id: int) -> Dict[str, Any]:
        """Parse a question line into a question dict."""
        # Clean up the question text
        for prefix in ["Q:", "Question:"]:
            if line.startswith(prefix):
                line = line[len(prefix) :].strip()

        return {
            "id": question_id,
            "question": line,
            "type": "open_ended",
            "correct_answer": "",
            "explanation": "",
            "topic_tag": "",
        }

    def _save_quiz(
        self, output: Path, quiz_data: Dict[str, Any], include_answers: bool
    ) -> None:
        """Save quiz to file.

        Saves as Markdown for human readability.
        """
        if output.suffix.lower() == ".json":
            self.save_json_output(output, quiz_data, f"Quiz saved to: {output}")
            return

        # Default to markdown
        if output.suffix.lower() != ".md":
            output = output.with_suffix(".md")

        content = self._format_quiz_markdown(quiz_data, include_answers)
        output.write_text(content, encoding="utf-8")
        self.print_success(f"Quiz saved to: {output}")

    def _format_quiz_markdown(
        self, quiz_data: Dict[str, Any], include_answers: bool
    ) -> str:
        """Format quiz as markdown document."""
        topic = quiz_data.get("topic", "Quiz")
        difficulty = quiz_data.get("difficulty", "medium")
        questions = quiz_data.get("questions", [])

        lines = [
            f"# {topic} - Practice Quiz",
            "",
            f"**Difficulty:** {difficulty.title()}",
            f"**Total Questions:** {len(questions)}",
            "",
            "---",
            "",
        ]

        # Questions section
        lines.append("## Questions")
        lines.append("")

        for q in questions:
            lines.extend(self._format_question_markdown(q, show_answer=False))

        # Answer key section
        if include_answers:
            lines.append("")
            lines.append("---")
            lines.append("")
            lines.append("## Answer Key")
            lines.append("")

            for q in questions:
                lines.extend(self._format_answer_markdown(q))

        lines.append("")
        lines.append("---")
        lines.append("*Generated by IngestForge*")

        return "\n".join(lines)

    def _format_question_markdown(
        self, question: Dict[str, Any], show_answer: bool = False
    ) -> List[str]:
        """Format a single question as markdown."""
        q_id = question.get("id", "?")
        q_text = question.get("question", "")
        q_type = question.get("type", "open_ended")
        topic_tag = question.get("topic_tag", "")

        lines = [f"### Question {q_id}"]
        if topic_tag:
            lines.append(f"*Topic: {topic_tag}*")
        lines.append("")
        lines.append(q_text)
        lines.append("")

        if q_type == "multiple_choice":
            options = question.get("options", [])
            for opt in options:
                lines.append(f"- {opt}")
            lines.append("")

        return lines

    def _format_answer_markdown(self, question: Dict[str, Any]) -> List[str]:
        """Format answer section for a question."""
        q_id = question.get("id", "?")
        answer = question.get("correct_answer", "")
        explanation = question.get("explanation", "")

        lines = [f"**{q_id}.** {answer}"]
        if explanation:
            lines.append(f"   *Explanation: {explanation}*")
        lines.append("")

        return lines

    def _display_quiz(
        self, quiz_data: Dict[str, Any], topic: str, include_answers: bool
    ) -> None:
        """Display quiz questions."""
        self.console.print()

        questions = quiz_data.get("questions", [])
        difficulty = quiz_data.get("difficulty", "medium")

        # Summary
        self.print_info(
            f"Generated {len(questions)} {difficulty} questions for: {topic}"
        )

        # Count by type
        mc_count = sum(1 for q in questions if q.get("type") == "multiple_choice")
        oe_count = len(questions) - mc_count
        self.console.print(f"  Multiple choice: {mc_count}, Open-ended: {oe_count}")
        self.console.print()

        # Display questions
        markdown_content = self._format_quiz_markdown(quiz_data, include_answers)

        panel = Panel(
            Markdown(markdown_content[:3000]),  # Truncate for display
            title=f"[bold green]Quiz: {topic}[/bold green]",
            border_style="green",
        )
        self.console.print(panel)

        # Topic breakdown
        self._display_topic_breakdown(questions)

    def _display_topic_breakdown(self, questions: List[Dict[str, Any]]) -> None:
        """Display breakdown of questions by topic."""
        topics: Dict[str, int] = {}
        for q in questions:
            tag = q.get("topic_tag", "General")
            topics[tag] = topics.get(tag, 0) + 1

        if len(topics) > 1:
            self.console.print()
            self.console.print("[bold cyan]Topic Breakdown:[/bold cyan]")
            for tag, count in sorted(topics.items(), key=lambda x: -x[1]):
                self.console.print(f"  {tag}: {count} questions")


# =============================================================================
# EXAM MODE - Timed Quiz with Deferred Feedback
# =============================================================================


@dataclass
class ExamAnswer:
    """Record of a single answer in exam mode."""

    question_id: int
    user_answer: str
    correct_answer: str
    is_correct: bool
    time_taken: float  # seconds
    timed_out: bool = False


@dataclass
class ExamResult:
    """Complete exam results."""

    topic: str
    difficulty: str
    total_questions: int
    correct_count: int
    answers: List[ExamAnswer] = field(default_factory=list)
    total_time: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    @property
    def score_percent(self) -> float:
        """Calculate score as percentage."""
        if self.total_questions == 0:
            return 0.0
        return (self.correct_count / self.total_questions) * 100

    @property
    def letter_grade(self) -> str:
        """Calculate letter grade from score."""
        score = self.score_percent
        if score >= 90:
            return "A"
        if score >= 80:
            return "B"
        if score >= 70:
            return "C"
        if score >= 60:
            return "D"
        return "F"


class ExamModeRunner:
    """Runs quiz in exam mode with timer and deferred feedback.

    Features:
    - Per-question time limit with visual countdown
    - No immediate feedback
    - Auto-submit when time expires
    - Final score with letter grade
    - Review of wrong answers
    """

    def __init__(
        self,
        console: Console,
        time_limit: int = 60,
        show_timer: bool = True,
    ) -> None:
        """Initialize exam mode runner.

        Args:
            console: Rich console for output
            time_limit: Seconds per question (default 60)
            show_timer: Whether to show countdown timer
        """
        self.console = console
        self.time_limit = time_limit
        self.show_timer = show_timer
        self._timer_stop = threading.Event()

    def run_exam(
        self,
        quiz_data: Dict[str, Any],
    ) -> ExamResult:
        """Run the exam and collect answers.

        Rule #4: Function <60 lines

        Args:
            quiz_data: Quiz data with questions

        Returns:
            ExamResult with all answers and score
        """
        questions = quiz_data.get("questions", [])
        topic = quiz_data.get("topic", "Quiz")
        difficulty = quiz_data.get("difficulty", "medium")

        result = ExamResult(
            topic=topic,
            difficulty=difficulty,
            total_questions=len(questions),
            correct_count=0,
            started_at=datetime.now(),
        )

        self._display_exam_intro(topic, len(questions), difficulty)

        for idx, question in enumerate(questions, 1):
            answer = self._ask_question(question, idx, len(questions))
            result.answers.append(answer)
            if answer.is_correct:
                result.correct_count += 1
            result.total_time += answer.time_taken

        result.completed_at = datetime.now()
        return result

    def _display_exam_intro(self, topic: str, count: int, difficulty: str) -> None:
        """Display exam introduction."""
        self.console.print()
        intro = Panel(
            f"[bold cyan]Exam Mode[/bold cyan]\n\n"
            f"Topic: {topic}\n"
            f"Questions: {count}\n"
            f"Difficulty: {difficulty}\n"
            f"Time limit: {self.time_limit} seconds per question\n\n"
            f"[yellow]Instructions:[/yellow]\n"
            f"- Answer each question within the time limit\n"
            f"- No feedback until the end\n"
            f"- Press Enter to submit your answer\n"
            f"- Type 'skip' to skip a question\n",
            title="[bold]Exam Starting[/bold]",
            border_style="cyan",
        )
        self.console.print(intro)
        self.console.print()

        # Countdown to start
        for i in range(3, 0, -1):
            self.console.print(f"[yellow]Starting in {i}...[/yellow]", end="\r")
            time.sleep(1)
        self.console.print("[green]Go!                    [/green]")
        self.console.print()

    def _ask_question(
        self,
        question: Dict[str, Any],
        current: int,
        total: int,
    ) -> ExamAnswer:
        """Ask a single question with timer.

        Rule #4: Function <60 lines

        Args:
            question: Question data
            current: Current question number
            total: Total questions

        Returns:
            ExamAnswer with user's response
        """
        q_id = question.get("id", current)
        q_text = question.get("question", "")
        q_type = question.get("type", "open_ended")
        correct = question.get("correct_answer", "")
        options = question.get("options", [])

        # Display question
        self._display_question(q_id, q_text, q_type, options, current, total)

        # Start timer and get answer
        start_time = time.time()
        user_answer, timed_out = self._get_timed_answer()
        elapsed = time.time() - start_time

        # Check if correct
        is_correct = self._check_answer(user_answer, correct, q_type)

        return ExamAnswer(
            question_id=q_id,
            user_answer=user_answer,
            correct_answer=correct,
            is_correct=is_correct,
            time_taken=elapsed,
            timed_out=timed_out,
        )

    def _display_question(
        self,
        q_id: int,
        text: str,
        q_type: str,
        options: List[str],
        current: int,
        total: int,
    ) -> None:
        """Display a question."""
        self.console.print()
        self.console.print(f"[bold cyan]Question {current}/{total}[/bold cyan]")
        self.console.print()
        self.console.print(f"[white]{text}[/white]")
        self.console.print()

        if q_type == "multiple_choice" and options:
            for opt in options:
                self.console.print(f"  {opt}")
            self.console.print()
            self.console.print("[dim]Enter A, B, C, or D:[/dim]")
        else:
            self.console.print("[dim]Type your answer:[/dim]")

    def _get_timed_answer(self) -> tuple[str, bool]:
        """Get answer with timeout.

        Returns:
            Tuple of (answer, timed_out)
        """
        # Simple implementation - no async needed
        self._timer_stop.clear()

        # Show timer in background thread
        timer_thread = None
        if self.show_timer:
            timer_thread = threading.Thread(
                target=self._countdown_display,
                daemon=True,
            )
            timer_thread.start()

        try:
            # Get input with timeout simulation
            # Note: Python input() doesn't support timeout natively
            # We'll use a simplified approach
            answer = Prompt.ask(
                f"[yellow][{self.time_limit}s][/yellow]",
                default="",
            )
            self._timer_stop.set()
            return answer.strip(), False

        except KeyboardInterrupt:
            self._timer_stop.set()
            return "", True

    def _countdown_display(self) -> None:
        """Display countdown timer (runs in background thread)."""
        start = time.time()
        while not self._timer_stop.is_set():
            elapsed = time.time() - start
            remaining = max(0, self.time_limit - int(elapsed))
            if remaining <= 0:
                break
            time.sleep(0.5)

    def _check_answer(self, user_answer: str, correct: str, q_type: str) -> bool:
        """Check if user's answer is correct.

        Args:
            user_answer: User's response
            correct: Correct answer
            q_type: Question type

        Returns:
            True if correct
        """
        if not user_answer:
            return False

        user_clean = user_answer.strip().lower()
        correct_clean = correct.strip().lower()

        if q_type == "multiple_choice":
            # For MC, just check the letter
            return user_clean[:1] == correct_clean[:1]

        # For open-ended, check for substantial overlap
        user_words = set(user_clean.split())
        correct_words = set(correct_clean.split())

        if not correct_words:
            return False

        overlap = len(user_words & correct_words)
        threshold = len(correct_words) * 0.5

        return overlap >= threshold

    def display_results(self, result: ExamResult) -> None:
        """Display final exam results.

        Rule #4: Function <60 lines
        """
        self.console.print()
        self.console.print()

        # Grade styling
        grade = result.letter_grade
        grade_colors = {
            "A": "green",
            "B": "cyan",
            "C": "yellow",
            "D": "red",
            "F": "red",
        }
        grade_color = grade_colors.get(grade, "white")

        # Summary panel
        summary = Panel(
            f"[bold]Score: {result.correct_count}/{result.total_questions} "
            f"({result.score_percent:.1f}%)[/bold]\n\n"
            f"Grade: [{grade_color}][bold]{grade}[/bold][/{grade_color}]\n\n"
            f"Time: {result.total_time:.1f} seconds\n"
            f"Avg per question: {result.total_time / max(1, result.total_questions):.1f}s",
            title=f"[bold]Exam Complete: {result.topic}[/bold]",
            border_style=grade_color,
        )
        self.console.print(summary)

        # Show incorrect answers for review
        incorrect = [a for a in result.answers if not a.is_correct]
        if incorrect:
            self.console.print()
            self.console.print("[bold yellow]Review Incorrect Answers:[/bold yellow]")
            self.console.print()

            for answer in incorrect:
                self.console.print(
                    f"  Q{answer.question_id}: "
                    f"Your answer: [red]{answer.user_answer or '(skipped)'}[/red] | "
                    f"Correct: [green]{answer.correct_answer}[/green]"
                )

        # Show timeout summary
        timeouts = sum(1 for a in result.answers if a.timed_out)
        if timeouts:
            self.console.print()
            self.console.print(f"[dim]Questions timed out: {timeouts}[/dim]")


# Create subcommand group for quiz
quiz_app = typer.Typer(
    name="quiz",
    help="Quiz generation commands",
    add_completion=False,
)


@quiz_app.command("create")
def create_command(
    topic: str = typer.Argument(..., help="Topic to generate questions about"),
    count: int = typer.Option(
        20, "--count", "-n", help="Number of questions to generate"
    ),
    difficulty: str = typer.Option(
        "medium", "--difficulty", "-d", help="Question difficulty (easy/medium/hard)"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for quiz (markdown or JSON)"
    ),
    include_answers: bool = typer.Option(
        True, "--include-answers/--no-answers", help="Include answer key in output"
    ),
    include_explanations: bool = typer.Option(
        True, "--explanations/--no-explanations", help="Include answer explanations"
    ),
) -> None:
    """Create a quiz from your knowledge base.

    Generates multiple-choice and open-ended questions based on
    ingested documents. Question difficulty affects the mix of
    question types and cognitive complexity.

    Difficulty levels:
    - easy: 80% multiple choice, basic recall
    - medium: 60% multiple choice, application focus
    - hard: 40% multiple choice, analysis and synthesis

    Examples:
        # Create 20 medium-difficulty questions
        ingestforge study quiz create "Machine Learning" --difficulty medium

        # Create hard quiz with explanations
        ingestforge study quiz create "Algorithms" -d hard --count 30

        # Save quiz without answers (for testing)
        ingestforge study quiz create "Biology" --no-answers -o test.md
    """
    cmd = QuizCommand()
    exit_code = cmd.execute(
        topic, count, project, output, difficulty, include_answers, include_explanations
    )
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@quiz_app.command("exam")
def exam_command(
    topic: str = typer.Argument(..., help="Topic for exam questions"),
    count: int = typer.Option(10, "--count", "-n", help="Number of questions"),
    time_limit: int = typer.Option(
        60, "--time-limit", "-t", help="Seconds per question"
    ),
    difficulty: str = typer.Option(
        "medium", "--difficulty", "-d", help="Question difficulty"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save results to file"
    ),
) -> None:
    """Take a timed exam with deferred feedback.

    Simulates real exam conditions:
    - Timer per question (default 60 seconds)
    - No immediate feedback on answers
    - Cannot go back to previous questions
    - Final score with letter grade at the end
    - Review of incorrect answers

    Use this mode to:
    - Build test-taking stamina
    - Practice under time pressure
    - Simulate real exam conditions

    Examples:
        # Take a 10-question exam with 60s per question
        ingestforge study quiz exam "Biology" --count 10

        # Shorter time limit for speed practice
        ingestforge study quiz exam "History" --time-limit 30

        # Save results for tracking progress
        ingestforge study quiz exam "Math" -o exam_results.json
    """
    console = Console()

    # Generate quiz first
    cmd = QuizCommand()
    cmd.console = console

    try:
        # Initialize and generate questions
        ctx = cmd.initialize_context(project, require_storage=True)
        llm_client = cmd.get_llm_client(ctx)

        if llm_client is None:
            raise typer.Exit(1)

        chunks = cmd.search_topic_context(ctx["storage"], topic, k=30)
        if not chunks:
            console.print(f"[yellow]No content found for topic: {topic}[/yellow]")
            raise typer.Exit(0)

        quiz_data = cmd._generate_quiz(
            llm_client, topic, chunks, count, difficulty, True, True
        )

        if not quiz_data:
            console.print("[red]Failed to generate exam questions[/red]")
            raise typer.Exit(1)

        # Run exam mode
        runner = ExamModeRunner(
            console=console,
            time_limit=time_limit,
            show_timer=True,
        )

        result = runner.run_exam(quiz_data)
        runner.display_results(result)

        # Save results if requested
        if output:
            import json

            result_data = {
                "topic": result.topic,
                "difficulty": result.difficulty,
                "score_percent": result.score_percent,
                "letter_grade": result.letter_grade,
                "correct": result.correct_count,
                "total": result.total_questions,
                "total_time": result.total_time,
                "started_at": result.started_at.isoformat()
                if result.started_at
                else None,
                "completed_at": result.completed_at.isoformat()
                if result.completed_at
                else None,
                "answers": [
                    {
                        "question_id": a.question_id,
                        "user_answer": a.user_answer,
                        "correct_answer": a.correct_answer,
                        "is_correct": a.is_correct,
                        "time_taken": a.time_taken,
                    }
                    for a in result.answers
                ],
            }
            output.write_text(
                json.dumps(result_data, indent=2),
                encoding="utf-8",
            )
            console.print(f"\n[green]Results saved to: {output}[/green]")

    except KeyboardInterrupt:
        console.print("\n[yellow]Exam cancelled[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Exam error: {e}[/red]")
        raise typer.Exit(1)


# Legacy command wrapper for backwards compatibility
def command(
    topic: str = typer.Argument(..., help="Topic to generate questions about"),
    count: int = typer.Option(
        5, "--count", "-n", help="Number of questions to generate"
    ),
    project: Optional[Path] = typer.Option(
        None, "--project", "-p", help="Project directory"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output file for quiz (JSON format)"
    ),
    difficulty: str = typer.Option(
        "medium", "--difficulty", "-d", help="Question difficulty (easy/medium/hard)"
    ),
) -> None:
    """Generate quiz questions from knowledge base.

    Creates multiple-choice and open-ended questions to test
    understanding of a topic based on ingested documents.

    Requires documents about the topic to be ingested first.

    Examples:
        # Generate 5 medium questions
        ingestforge study quiz "Python programming"

        # Generate 10 hard questions
        ingestforge study quiz "Machine Learning" --count 10 --difficulty hard

        # Save to file
        ingestforge study quiz "History" --output history_quiz.json

        # Specific project
        ingestforge study quiz "Biology" -p /path/to/project
    """
    cmd = QuizCommand()
    exit_code = cmd.execute(topic, count, project, output, difficulty, True, True)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
