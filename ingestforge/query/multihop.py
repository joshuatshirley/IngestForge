"""Multi-hop reasoning for complex queries.

Enables answering questions that require chaining multiple retrieval steps.
Example: "How did X influence Y which led to Z?"
"""

from __future__ import annotations

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class MultiHopReasoner:
    """Multi-hop reasoning for complex queries."""

    def __init__(self, storage: Any, llm_client: Any, max_hops: int = 3) -> None:
        """Initialize multi-hop reasoner.

        Args:
            storage: ChunkRepository instance
            llm_client: LLM client for decomposition and synthesis
            max_hops: Maximum reasoning hops
        """
        self.storage = storage
        self.llm_client = llm_client
        self.max_hops = max_hops

    def reason(self, query: str, k: int = 5) -> Dict[str, Any]:
        """Perform multi-hop reasoning.

        Args:
            query: Complex query requiring multiple steps
            k: Number of chunks per hop

        Returns:
            Reasoning result with steps and final answer
        """
        # Decompose query into sub-questions
        sub_questions = self._decompose_query(query)

        # Execute reasoning chain
        reasoning_chain = self._execute_chain(sub_questions, k)

        # Synthesize final answer
        final_answer = self._synthesize_answer(query, reasoning_chain)

        return {
            "query": query,
            "sub_questions": sub_questions,
            "reasoning_chain": reasoning_chain,
            "answer": final_answer,
            "hops": len(reasoning_chain),
        }

    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex query into sub-questions.

        Args:
            query: Complex query

        Returns:
            List of sub-questions
        """
        # Use LLM to decompose query
        prompt = f"""Decompose this complex query into 2-{self.max_hops} simpler sub-questions
that need to be answered sequentially:

Query: {query}

Return a JSON array of sub-questions in order:
["sub-question 1", "sub-question 2", ...]

Only return the JSON array, no other text."""

        try:
            response = self._generate_llm(prompt)

            # Parse JSON array
            import json

            sub_questions = json.loads(response)

            if isinstance(sub_questions, list):
                return sub_questions[: self.max_hops]

        except Exception as e:
            logger.warning(
                f"Failed to decompose query using LLM: {e}. "
                "Falling back to simple decomposition."
            )

        # Fallback: simple splitting
        return self._simple_decompose(query)

    def _simple_decompose(self, query: str) -> List[str]:
        """Simple query decomposition fallback.

        Args:
            query: Query to decompose

        Returns:
            List of sub-questions
        """
        # Look for connecting words that indicate multi-step reasoning
        if "which led to" in query.lower():
            parts = query.split("which led to")
            return [part.strip() for part in parts]

        if "and then" in query.lower():
            parts = query.split("and then")
            return [part.strip() for part in parts]

        # Default: single question
        return [query]

    def _execute_chain(self, sub_questions: List[str], k: int) -> List[Dict[str, Any]]:
        """Execute reasoning chain.

        Args:
            sub_questions: List of sub-questions
            k: Chunks per question

        Returns:
            Reasoning chain with intermediate results
        """
        chain: list[dict[str, Any]] = []
        context_accumulator: list[str] = []

        for idx, sub_q in enumerate(sub_questions):
            # Retrieve relevant chunks
            chunks = self.storage.search(sub_q, k=k)

            # Build context including previous steps
            chunk_texts = [getattr(c, "text", str(c))[:500] for c in chunks]
            current_context = "\n\n".join(chunk_texts)

            # Generate intermediate answer
            answer = self._answer_subquestion(
                sub_q, current_context, context_accumulator
            )

            step = {
                "step": idx + 1,
                "question": sub_q,
                "chunks_retrieved": len(chunks),
                "answer": answer,
            }

            chain.append(step)

            # Accumulate context for next step
            context_accumulator.append(f"Q{idx+1}: {sub_q}\nA{idx+1}: {answer}")

        return chain

    def _answer_subquestion(
        self,
        sub_question: str,
        current_context: str,
        previous_steps: List[str],
    ) -> str:
        """Answer a sub-question.

        Args:
            sub_question: Current sub-question
            current_context: Context from retrieval
            previous_steps: Previous reasoning steps

        Returns:
            Answer to sub-question
        """
        previous_context = "\n\n".join(previous_steps) if previous_steps else ""

        prompt = f"""Answer this question based on the provided context.

Question: {sub_question}

Previous reasoning steps:
{previous_context if previous_context else 'None'}

Current context:
{current_context}

Provide a concise answer (2-3 sentences)."""

        return self._generate_llm(prompt)

    def _synthesize_answer(
        self, original_query: str, reasoning_chain: List[Dict[str, Any]]
    ) -> str:
        """Synthesize final answer from reasoning chain.

        Args:
            original_query: Original complex query
            reasoning_chain: Chain of intermediate results

        Returns:
            Final synthesized answer
        """
        # Build summary of reasoning steps
        steps_summary = []
        for step in reasoning_chain:
            steps_summary.append(
                f"Step {step['step']}: {step['question']}\n" f"Answer: {step['answer']}"
            )

        steps_text = "\n\n".join(steps_summary)

        prompt = f"""Synthesize a final answer to the original query based on the
reasoning chain below.

Original query: {original_query}

Reasoning chain:
{steps_text}

Provide a comprehensive final answer that integrates all steps."""

        return self._generate_llm(prompt)

    def _generate_llm(self, prompt: str) -> str:
        """Generate text using LLM.

        Args:
            prompt: Prompt text

        Returns:
            Generated response
        """
        if hasattr(self.llm_client, "generate"):
            generated: str = self.llm_client.generate(prompt)
            return generated
        elif hasattr(self.llm_client, "complete"):
            completed: str = self.llm_client.complete(prompt)
            return completed
        elif callable(self.llm_client):
            called: str = self.llm_client(prompt)
            return called
        else:
            return "LLM client not available"


def multihop_query(
    storage: Any,
    llm_client: Any,
    query: str,
    max_hops: int = 3,
    k: int = 5,
) -> Dict[str, Any]:
    """Perform multi-hop reasoning query.

    Args:
        storage: ChunkRepository instance
        llm_client: LLM client
        query: Complex query
        max_hops: Maximum reasoning hops
        k: Chunks per hop

    Returns:
        Multi-hop reasoning result
    """
    reasoner = MultiHopReasoner(storage, llm_client, max_hops)
    return reasoner.reason(query, k)
