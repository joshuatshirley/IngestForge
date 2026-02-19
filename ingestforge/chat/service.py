"""
Chat Service for Conversational RAG.

Conversational Query Mode
Handles multi-turn conversation logic, query contextualization, and RAG execution.

JPL Compliance:
- Rule #1: No recursion
- Rule #2: Bounded loops
- Rule #9: Complete type hints
"""

from dataclasses import dataclass
from typing import List, Optional

from ingestforge.core.logging import get_logger
from ingestforge.core.config import Config
from ingestforge.core.config_loaders import load_config
from ingestforge.llm.factory import get_llm_client
from ingestforge.llm.base import GenerationConfig, LLMClient
from ingestforge.retrieval.hybrid import HybridRetriever
from ingestforge.storage.base import SearchResult

logger = get_logger(__name__)

# JPL Rule #2: Fixed bounds
MAX_HISTORY_MESSAGES = 10
MAX_CONTEXT_CHUNKS = 5
MAX_REWRITE_TOKENS = 200
MAX_ANSWER_TOKENS = 1024


@dataclass
class ChatMessage:
    """Single message in conversation history."""

    role: str  # 'user' or 'ai'
    content: str


@dataclass
class ChatResponse:
    """Response from chat service."""

    answer: str
    sources: List[SearchResult]
    context_query: str  # The rewritten query used for search


class ChatService:
    """
    Orchestrates conversational RAG flow.

    Follow-up questions use context.
    Query clarification and contextualization.
    Integration with ReAct engine components.

    Flow:
    1. Contextualize (rewrite) user query based on history.
    2. Retrieve relevant chunks using rewritten query.
    3. Generate answer using chunks + history.
    """

    def __init__(self, config: Optional[Config] = None):
        """Initialize chat service."""
        self.config = config or load_config()
        self._llm: Optional[LLMClient] = None
        self._retriever: Optional[HybridRetriever] = None

    @property
    def llm(self) -> LLMClient:
        """Lazy load LLM client."""
        if self._llm is None:
            self._llm = get_llm_client(self.config)
        return self._llm

    @property
    def retriever(self) -> HybridRetriever:
        """Lazy load retriever."""
        if self._retriever is None:
            # Import here to avoid circular deps if any
            from ingestforge.core.pipeline.pipeline import Pipeline

            pipeline = Pipeline(self.config)
            self._retriever = HybridRetriever(self.config, pipeline.storage)
        return self._retriever

    def chat(self, history: List[ChatMessage], user_query: str) -> ChatResponse:
        """
        Execute chat turn.

        Implements multi-turn context usage.

        Args:
            history: Previous conversation messages.
            user_query: Current user question.

        Returns:
            ChatResponse with answer and sources.
        """
        # 1. Contextualize Query
        context_query = self._rewrite_query(history, user_query)
        logger.info(f"Rewrote query: '{user_query}' -> '{context_query}'")

        # 2. Retrieve
        chunks = self.retriever.search(query=context_query, top_k=MAX_CONTEXT_CHUNKS)

        # 3. Generate Answer
        answer = self._generate_answer(history, user_query, chunks)

        return ChatResponse(answer=answer, sources=chunks, context_query=context_query)

    def _rewrite_query(self, history: List[ChatMessage], current_query: str) -> str:
        """
        Rewrite query to be standalone based on history.

        Query contextualization logic.
        Rule #4: <60 lines
        """
        if not history:
            return current_query

        # Take last N messages for context
        recent_history = history[-MAX_HISTORY_MESSAGES:]

        history_text = "\n".join(
            f"{msg.role.upper()}: {msg.content}" for msg in recent_history
        )

        prompt = f"""Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone query.
If the follow-up question is already standalone, return it unchanged.
Do NOT answer the question, just rewrite it.

Chat History:
{history_text}

Follow Up Input: {current_query}
Standalone Question:"""

        try:
            config = GenerationConfig(max_tokens=MAX_REWRITE_TOKENS, temperature=0.0)
            response = self.llm.generate(prompt, config)
            # Clean up response (remove quotes etc)
            cleaned = response.text.strip().strip('"')
            return cleaned
        except Exception as e:
            logger.error(f"Query rewrite failed: {e}")
            return current_query

    def _generate_answer(
        self, history: List[ChatMessage], query: str, chunks: List[SearchResult]
    ) -> str:
        """
        Generate final answer using context.

        Sources + History for answer generation.
        """
        context_text = "\n\n".join(
            f"Source {i+1}:\n{chunk.content}" for i, chunk in enumerate(chunks)
        )

        prompt = f"""You are a helpful research assistant. Answer the user's question based ONLY on the provided sources.
If the answer is not in the sources, say you don't know.

Sources:
{context_text}

Question: {query}

Answer:"""

        try:
            config = GenerationConfig(max_tokens=MAX_ANSWER_TOKENS, temperature=0.3)
            response = self.llm.generate(prompt, config)
            return response.text
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "I encountered an error generating the response."
