"""LLM-powered explanation generator for study materials (QUIZ-002.1).

This module provides "why" button functionality for quiz answers,
generating context-aware explanations using the configured LLM.

NASA JPL Commandments compliance:
- Rule #1: No deep nesting, early returns
- Rule #4: Functions <60 lines
- Rule #7: Input validation
- Rule #9: Full type hints

Usage:
    from ingestforge.study.explanations import ExplanationGenerator

    generator = ExplanationGenerator(config)
    explanation = generator.explain_answer(
        question="What is the capital of France?",
        correct_answer="Paris",
        user_answer="Lyon",  # Optional
    )
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, TYPE_CHECKING

from ingestforge.core.logging import get_logger

if TYPE_CHECKING:
    from ingestforge.core.config import Config
    from ingestforge.llm.base import LLMClient

logger = get_logger(__name__)

# Maximum explanation length to prevent verbose responses
MAX_EXPLANATION_TOKENS: int = 256

# Prompt templates
EXPLANATION_PROMPT = """Explain why the correct answer is what it is, in 2-3 concise sentences.

Question: {question}
Correct Answer: {correct_answer}
{user_context}
Explain briefly why this is correct:"""

WRONG_ANSWER_CONTEXT = "User's Wrong Answer: {user_answer}\nBriefly explain why this was incorrect and why the correct answer is right."

CONCEPT_PROMPT = """Given this question-answer pair, identify the key concept being tested and explain it in 2-3 sentences.

Question: {question}
Answer: {answer}

Key concept and explanation:"""


@dataclass
class Explanation:
    """An explanation for a study question.

    Attributes:
        text: The explanation text
        question: The original question
        correct_answer: The correct answer
        user_answer: The user's answer (if wrong)
        source: Where this explanation came from
    """

    text: str
    question: str
    correct_answer: str
    user_answer: Optional[str] = None
    source: str = "llm"

    @property
    def was_wrong(self) -> bool:
        """Whether the user got this wrong."""
        return self.user_answer is not None and self.user_answer != self.correct_answer

    @property
    def is_fallback(self) -> bool:
        """Whether this is a fallback explanation."""
        return self.source == "fallback"


@dataclass
class ExplanationRequest:
    """Request for an explanation.

    Attributes:
        question: The question text
        correct_answer: The correct answer
        user_answer: Optional user's answer (for wrong answer explanations)
        context: Optional additional context from source material
    """

    question: str
    correct_answer: str
    user_answer: Optional[str] = None
    context: Optional[str] = None

    def validate(self) -> List[str]:
        """Validate the request.

        Returns:
            List of validation error messages (empty if valid)
        """
        errors: List[str] = []

        if not self.question or not self.question.strip():
            errors.append("Question cannot be empty")

        if not self.correct_answer or not self.correct_answer.strip():
            errors.append("Correct answer cannot be empty")

        return errors


class ExplanationGenerator:
    """Generates LLM-powered explanations for study questions.

    This class provides the "Why?" button functionality for quizzes,
    generating concise, context-aware explanations for answers.

    Args:
        config: IngestForge configuration
        llm_client: Optional LLM client (auto-created from config if not provided)

    Example:
        generator = ExplanationGenerator(config)

        # Explain a correct answer
        explanation = generator.explain_answer(
            question="What year did WWII end?",
            correct_answer="1945",
        )

        # Explain why user's answer was wrong
        explanation = generator.explain_answer(
            question="What year did WWII end?",
            correct_answer="1945",
            user_answer="1944",
        )
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        llm_client: Optional[LLMClient] = None,
    ) -> None:
        """Initialize the explanation generator."""
        self._config = config
        self._llm_client = llm_client
        self._llm_available: Optional[bool] = None

    @property
    def llm_client(self) -> Optional[LLMClient]:
        """Lazy-initialize LLM client."""
        if self._llm_client is not None:
            return self._llm_client

        if self._config is None:
            logger.warning("No config provided, cannot create LLM client")
            return None

        try:
            from ingestforge.llm.factory import get_best_available_client

            self._llm_client = get_best_available_client(self._config)
            return self._llm_client
        except Exception as e:
            logger.warning(f"Failed to create LLM client: {e}")
            return None

    @property
    def is_available(self) -> bool:
        """Check if LLM explanations are available."""
        if self._llm_available is not None:
            return self._llm_available

        client = self.llm_client
        if client is None:
            self._llm_available = False
            return False

        try:
            self._llm_available = client.is_available()
        except Exception:
            self._llm_available = False

        return self._llm_available

    def explain_answer(
        self,
        question: str,
        correct_answer: str,
        user_answer: Optional[str] = None,
        context: Optional[str] = None,
    ) -> Explanation:
        """Generate an explanation for an answer.

        Args:
            question: The question text
            correct_answer: The correct answer
            user_answer: Optional user's answer (for wrong answer context)
            context: Optional source material context

        Returns:
            Explanation object with generated text
        """
        request = ExplanationRequest(
            question=question,
            correct_answer=correct_answer,
            user_answer=user_answer,
            context=context,
        )
        return self.explain(request)

    def explain(self, request: ExplanationRequest) -> Explanation:
        """Generate an explanation from a request.

        Args:
            request: ExplanationRequest with question details

        Returns:
            Explanation object
        """
        errors = request.validate()
        if errors:
            return self._create_fallback(
                request,
                f"Invalid request: {'; '.join(errors)}",
            )
        if not self.is_available:
            return self._create_fallback(
                request,
                "LLM not available for explanations",
            )

        # Generate explanation via LLM
        try:
            return self._generate_explanation(request)
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            return self._create_fallback(request, str(e))

    def _generate_explanation(self, request: ExplanationRequest) -> Explanation:
        """Generate explanation using LLM.

        Args:
            request: Validated explanation request

        Returns:
            Explanation with LLM-generated text
        """
        client = self.llm_client
        assert client is not None, "LLM client should be available"

        # Build prompt based on whether user got it wrong
        prompt = self._build_prompt(request)

        # Generate with appropriate config
        from ingestforge.llm.base import GenerationConfig

        gen_config = GenerationConfig(
            max_tokens=MAX_EXPLANATION_TOKENS,
            temperature=0.3,  # Factual explanations need lower temperature
        )

        response = client.generate(prompt, gen_config)

        return Explanation(
            text=response.strip(),
            question=request.question,
            correct_answer=request.correct_answer,
            user_answer=request.user_answer,
            source="llm",
        )

    def _build_prompt(self, request: ExplanationRequest) -> str:
        """Build the LLM prompt for explanation.

        Args:
            request: Explanation request

        Returns:
            Formatted prompt string
        """
        # Build user context section
        user_context = ""
        if request.user_answer and request.user_answer != request.correct_answer:
            user_context = WRONG_ANSWER_CONTEXT.format(user_answer=request.user_answer)

        # Add source context if provided
        if request.context:
            user_context = f"Source Context: {request.context}\n{user_context}"

        return EXPLANATION_PROMPT.format(
            question=request.question,
            correct_answer=request.correct_answer,
            user_context=user_context,
        )

    def _create_fallback(
        self,
        request: ExplanationRequest,
        reason: str,
    ) -> Explanation:
        """Create a fallback explanation when LLM is unavailable.

        Args:
            request: The explanation request
            reason: Why LLM was not used

        Returns:
            Fallback Explanation
        """
        logger.debug(f"Using fallback explanation: {reason}")

        # Simple fallback text
        if request.user_answer and request.user_answer != request.correct_answer:
            text = (
                f"The correct answer is '{request.correct_answer}', "
                f"not '{request.user_answer}'."
            )
        else:
            text = f"The answer is '{request.correct_answer}'."

        return Explanation(
            text=text,
            question=request.question,
            correct_answer=request.correct_answer,
            user_answer=request.user_answer,
            source="fallback",
        )

    def explain_concept(
        self,
        question: str,
        answer: str,
    ) -> Explanation:
        """Explain the key concept behind a question.

        This provides a deeper understanding of the underlying concept,
        not just why the answer is correct.

        Args:
            question: The question text
            answer: The answer

        Returns:
            Explanation of the key concept
        """
        if not self.is_available:
            return Explanation(
                text=f"This question tests your knowledge of concepts related to: {answer}",
                question=question,
                correct_answer=answer,
                source="fallback",
            )

        try:
            client = self.llm_client
            assert client is not None

            prompt = CONCEPT_PROMPT.format(
                question=question,
                answer=answer,
            )

            from ingestforge.llm.base import GenerationConfig

            gen_config = GenerationConfig(
                max_tokens=MAX_EXPLANATION_TOKENS,
                temperature=0.3,
            )

            response = client.generate(prompt, gen_config)

            return Explanation(
                text=response.strip(),
                question=question,
                correct_answer=answer,
                source="llm",
            )
        except Exception as e:
            logger.error(f"Failed to explain concept: {e}")
            return Explanation(
                text=f"This question tests your knowledge of concepts related to: {answer}",
                question=question,
                correct_answer=answer,
                source="fallback",
            )


def explain_answer(
    question: str,
    correct_answer: str,
    user_answer: Optional[str] = None,
    config: Optional[Config] = None,
) -> str:
    """Convenience function to generate an explanation.

    Args:
        question: The question text
        correct_answer: The correct answer
        user_answer: Optional user's wrong answer
        config: Optional IngestForge config

    Returns:
        Explanation text
    """
    generator = ExplanationGenerator(config=config)
    explanation = generator.explain_answer(
        question=question,
        correct_answer=correct_answer,
        user_answer=user_answer,
    )
    return explanation.text
