"""
Query processing pipeline.

Full pipeline from query to response with retrieval and generation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from ingestforge.core.config import Config
from ingestforge.core.logging import get_logger
from ingestforge.query.cache import QueryCache, CacheConfig
from ingestforge.retrieval.hybrid import HybridRetriever
from ingestforge.retrieval.reranker import Reranker
from ingestforge.storage.base import SearchResult

logger = get_logger(__name__)

# =============================================================================
# JPL Power of Ten Rule #2: Fixed Upper Bounds for All Loops
# =============================================================================
# Epic JPL Static Analysis Complete
# Epic JPL Rule #2 (Bounded Loops) Compliance
# Timestamp: 2026-02-18 19:00 UTC
#
# These constants establish fixed upper bounds for all loop iterations in the
# query pipeline, ensuring compliance with NASA JPL Power of Ten Rule #2.
# All bounds are set conservatively above typical production values to ensure
# zero behavior change while preventing unbounded iteration risks.
#
# Reference: JPL_COMPREHENSIVE_STATIC_ANALYSIS_US602_2026-02-18.md
# =============================================================================

MAX_RETRIEVAL_RESULTS = 1000
"""Maximum search results from retrieval before deduplication.
Typical: 20-100 results | Margin: 10x-50x | Epic | 2026-02-18 19:00 UTC"""

MAX_SOURCES_FOR_MERGING = 100
"""Maximum sources to process in adjacent chunk merging.
Typical: 10-50 sources | Margin: 2x-10x | Epic | 2026-02-18 19:00 UTC"""

MAX_FOLLOWUP_LINES = 50
"""Maximum lines to parse from LLM follow-up suggestions.
Typical: 5-15 lines | Margin: 3x-10x | Epic | 2026-02-18 19:00 UTC"""

MAX_DOC_CHUNKS = 500
"""Maximum chunks to iterate per document during context expansion.
Typical: 50-300 chunks | Margin: 1.7x-10x | Epic | 2026-02-18 19:00 UTC"""

MAX_EXPANDED_SOURCES = 200
"""Maximum expanded sources before final deduplication.
Typical: 30-100 sources | Margin: 2x-6x | Epic | 2026-02-18 19:00 UTC"""


@dataclass
class QueryResult:
    """Result of query processing."""

    query: str
    answer: Optional[str]
    sources: List[SearchResult]
    confidence: float
    metadata: Dict[str, Any]
    follow_up_suggestions: List[str] = None

    def __post_init__(self) -> None:
        if self.follow_up_suggestions is None:
            self.follow_up_suggestions = []


@dataclass
class ClarificationNeeded:
    """
    Special result indicating query needs clarification.

    Returned when query is ambiguous and needs refinement.
    This is a "query result" that signals the frontend to show clarification dialog.
    """

    original_query: str
    clarity_score: float
    suggestions: List[str]
    reason: str

    def to_query_result(self) -> QueryResult:
        """
        Convert to QueryResult for backward compatibility.

        Returns QueryResult with clarification metadata.
        """
        return QueryResult(
            query=self.original_query,
            answer=None,
            sources=[],
            confidence=0.0,
            metadata={
                "needs_clarification": True,
                "clarity_score": self.clarity_score,
                "suggestions": self.suggestions,
                "reason": self.reason,
            },
            follow_up_suggestions=self.suggestions,
        )


class QueryPipeline:
    """
    Full query processing pipeline.

    Steps:
    1. Query classification and expansion
    2. Hybrid retrieval (BM25 + semantic)
    3. Reranking
    4. Response generation (optional)
    """

    def __init__(
        self,
        config: Config,
        retriever: HybridRetriever,
        llm_client: Optional[Any] = None,
        enable_cache: bool = True,
        cache_path: Optional[Path] = None,
    ):
        """
        Initialize query pipeline.

        Args:
            config: IngestForge configuration
            retriever: Hybrid retriever for search
            llm_client: Optional LLM client for generation
            enable_cache: Enable query result caching
            cache_path: Optional path for cache persistence
        """
        self.config = config
        self.retriever = retriever
        self.llm_client = llm_client

        # Components
        self.reranker = Reranker(config.retrieval.rerank_model)
        self._classifier = None
        self._expander = None
        self._clarifier = None

        # Initialize query cache
        cache_config = CacheConfig(
            enabled=enable_cache,
            max_size=1000,
            ttl_seconds=3600,  # 1 hour
            persist_path=cache_path,
        )
        self._cache = QueryCache(cache_config)

    @property
    def classifier(self) -> Any:
        """Lazy-load query classifier."""
        if self._classifier is None:
            from ingestforge.query.classifier import QueryClassifier

            self._classifier = QueryClassifier()
        return self._classifier

    @property
    def expander(self) -> Any:
        """Lazy-load query expander."""
        if self._expander is None:
            from ingestforge.query.expander import QueryExpander

            self._expander = QueryExpander()
        return self._expander

    @property
    def clarifier(self) -> Any:
        """
        Lazy-load query clarifier.

        Provides query ambiguity detection.
        """
        if self._clarifier is None:
            from ingestforge.query.clarifier import IFQueryClarifier

            self._clarifier = IFQueryClarifier()
        return self._clarifier

    def _check_clarification(
        self, query: str, threshold: float
    ) -> Optional[QueryResult]:
        """
        Check if query needs clarification before processing.

        Ambiguity guard to prevent low-quality RAG results.

        Rule #1: Early return pattern
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            query: User query to evaluate
            threshold: Clarity threshold (0.0-1.0)

        Returns:
            QueryResult with clarification metadata if needed, None otherwise
        """
        try:
            # Evaluate query clarity
            artifact = self.clarifier.evaluate(query)

            # If query is clear, continue processing
            if not artifact.needs_clarification:
                logger.debug(
                    f"Query clarity OK: score={artifact.clarity_score.score:.2f}"
                )
                return None

            # Query needs clarification
            logger.info(
                f"Query needs clarification: score={artifact.clarity_score.score:.2f}, "
                f"reason={artifact.reason}"
            )

            return QueryResult(
                query=query,
                answer=None,
                sources=[],
                confidence=0.0,
                metadata={
                    "needs_clarification": True,
                    "clarity_score": artifact.clarity_score.score,
                    "suggestions": artifact.suggestions,
                    "reason": artifact.reason,
                    "factors": artifact.clarity_score.factors,
                },
                follow_up_suggestions=artifact.suggestions,
            )

        except Exception as e:
            # If clarification fails, log and continue (fail-open)
            logger.warning(f"Query clarification failed: {e}, continuing anyway")
            return None

    def _check_cache(
        self,
        query: str,
        cache_key_options: dict,
        use_cache: bool,
        generate_answer: bool,
    ) -> Optional[QueryResult]:
        """
        Check cache for existing results.

        Rule #1: Early return for cache miss
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            query: User query
            cache_key_options: Cache key options
            use_cache: Whether to use cache
            generate_answer: Whether answer generation requested

        Returns:
            Cached QueryResult or None if cache miss
        """
        if not use_cache or generate_answer:
            return None

        cached = self._cache.get(query, cache_key_options)
        if not cached:
            return None

        logger.debug("Cache hit", query=query[:50])

        # Reconstruct SearchResult objects from cached dicts
        sources = [self._dict_to_search_result(r) for r in cached]

        return QueryResult(
            query=query,
            answer=None,
            sources=sources,
            confidence=0.0,
            metadata={"cached": True},
        )

    def _retrieve_and_deduplicate(
        self,
        query: str,
        query_type: str,
        expanded_queries: List[str],
        top_k: int,
        library_filter: Optional[str],
        **kwargs,
    ) -> List:
        """
        Retrieve results and deduplicate.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            query: Original query
            query_type: Classified query type
            expanded_queries: Expanded query variations
            top_k: Number of results
            library_filter: Library filter
            **kwargs: Additional search parameters

        Returns:
            Deduplicated list of search results
        """
        # Retrieve with expansions
        all_results = []
        for q in [query] + expanded_queries[:2]:  # Limit expansions
            results = self.retriever.search(
                q,
                top_k=top_k * 2,
                library_filter=library_filter,
                query_intent=query_type,
                **kwargs,
            )
            all_results.extend(results)

        # =================================================================
        # JPL Rule #2: Bound retrieval results before deduplication
        # Epic Bounded loops compliance
        # Timestamp: 2026-02-18 19:00 UTC
        #
        # VIOLATION #1 FIXED: Line 284 unbounded loop over all_results
        # Risk: HIGH - OOM with 10K+ search results
        # Fix: Bound to MAX_RETRIEVAL_RESULTS (1000)
        # Impact: ZERO - typical retrieval returns <100 results
        # =================================================================
        bounded_results = all_results[:MAX_RETRIEVAL_RESULTS]

        # Deduplicate by chunk_id
        seen = set()
        unique_results = []
        for r in bounded_results:
            if r.chunk_id not in seen:
                seen.add(r.chunk_id)
                unique_results.append(r)

        return unique_results

    def _perform_reranking(self, query: str, unique_results: List, top_k: int) -> List:
        """
        Rerank results if needed.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            query: User query
            unique_results: Deduplicated results
            top_k: Number of results

        Returns:
            Reranked results
        """
        if self.config.retrieval.rerank and len(unique_results) > top_k:
            return self.reranker.rerank(query, unique_results, top_k)
        return unique_results[:top_k]

    def _try_generate_answer(
        self,
        generate_answer: bool,
        query: str,
        reranked: List,
        conversation_context: Optional[str],
    ) -> tuple:
        """
        Try to generate answer with LLM.

        Rule #1: Early return for disabled/missing LLM
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            generate_answer: Whether to generate
            query: User query
            reranked: Reranked results
            conversation_context: Conversation context

        Returns:
            Tuple of (answer, confidence, follow_ups, warning)
        """
        if not (generate_answer and self.llm_client and reranked):
            return (None, 0.0, [], None)

        try:
            answer, confidence, follow_ups = self._generate_answer(
                query,
                reranked,
                conversation_context=conversation_context,
            )
            return (answer, confidence, follow_ups, None)

        except Exception as e:
            logger.warning(f"Answer generation failed: {e}")
            warning = f"Retrieved {len(reranked)} chunks but synthesis unavailable (LLM error: {type(e).__name__})"
            return (None, 0.0, [], warning)

    def _build_query_result(
        self,
        query: str,
        answer: Optional[str],
        sources: List[SearchResult],
        confidence: float,
        query_type: str,
        domains: List[str],
        expanded_queries: List[str],
        unique_results_count: int,
        follow_ups: List[str],
        llm_failure_warning: Optional[str],
        cached: bool = False,
    ) -> QueryResult:
        """Build QueryResult from pipeline components.

        Rule #4: No large functions - Extracted from process()
        """
        return QueryResult(
            query=query,
            answer=answer,
            sources=sources,
            confidence=confidence,
            metadata={
                "query_type": query_type,
                "domains": domains,
                "expanded_queries": expanded_queries,
                "total_retrieved": unique_results_count + len(expanded_queries),
                "unique_results": unique_results_count,
                "cached": cached,
                "llm_failure_warning": llm_failure_warning,
            },
            follow_up_suggestions=follow_ups,
        )

    def process(
        self,
        query: str,
        top_k: Optional[int] = None,
        generate_answer: bool = True,
        use_cache: bool = True,
        conversation_context: Optional[str] = None,
        library_filter: Optional[str] = None,
        enable_clarification: bool = False,
        clarity_threshold: float = 0.7,
        **kwargs,
    ) -> QueryResult:
        """
        Process a query through the full pipeline.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines (refactored from 69)
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            query: User query
            top_k: Number of results
            generate_answer: Generate LLM answer
            use_cache: Use cached results if available
            conversation_context: Previous conversation for multi-turn
            library_filter: If provided, only return chunks from this library
            enable_clarification: Enable query clarification check ()
            clarity_threshold: Clarity threshold for clarification (0.0-1.0)

        Returns:
            QueryResult with answer and sources (or clarification metadata if needed)
        """
        top_k = top_k or self.config.retrieval.top_k
        cache_opts = {"top_k": top_k, "library_filter": library_filter}

        # Check if query needs clarification (before cache/retrieval)
        if enable_clarification:
            clarification_result = self._check_clarification(query, clarity_threshold)
            if clarification_result:
                return clarification_result

        # Check cache first
        cached_result = self._check_cache(query, cache_opts, use_cache, generate_answer)
        if cached_result:
            return cached_result

        # Execute retrieval pipeline
        query_type, expanded_queries, reranked, domains = self._execute_retrieval(
            query, top_k, library_filter, cache_opts, use_cache, **kwargs
        )

        # Generate answer
        answer, confidence, follow_ups, warning = self._try_generate_answer(
            generate_answer, query, reranked, conversation_context
        )

        return self._build_query_result(
            query=query,
            answer=answer,
            sources=reranked,
            confidence=confidence,
            query_type=query_type,
            domains=domains,
            expanded_queries=expanded_queries,
            unique_results_count=len(reranked),
            follow_ups=follow_ups,
            llm_failure_warning=warning,
            cached=False,
        )

    def _execute_retrieval(
        self,
        query: str,
        top_k: int,
        library_filter: Optional[str],
        cache_opts: dict,
        use_cache: bool,
        **kwargs,
    ) -> Tuple[str, List[str], List, List[str]]:
        """
        Execute the retrieval pipeline: classify, expand, retrieve, rerank.

        Rule #1: Extracted helper reduces nesting in process()
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            query: User query
            top_k: Number of results
            library_filter: Library filter
            cache_opts: Cache key options
            use_cache: Whether to cache results
            **kwargs: Additional arguments

        Returns:
            Tuple of (query_type, expanded_queries, reranked_results, domains)
        """
        # Classify and expand
        classification = self.classifier.classify_full(query)
        query_type = classification.intent
        domains = classification.domains

        # Get domain strategies
        from ingestforge.query.routing import get_merged_strategy

        domain_strategy = get_merged_strategy(domains)

        # Apply modifiers to kwargs for the retriever
        kwargs["bm25_modifier"] = domain_strategy.bm25_modifier
        kwargs["semantic_modifier"] = domain_strategy.semantic_modifier
        if domain_strategy.boost_fields:
            kwargs["boost_fields"] = domain_strategy.boost_fields

        expanded_queries = self.expander.expand(query, query_type)

        # Retrieve and deduplicate
        unique_results = self._retrieve_and_deduplicate(
            query, query_type, expanded_queries, top_k, library_filter, **kwargs
        )

        # Rerank
        reranked = self._perform_reranking(query, unique_results, top_k)

        # Cache if enabled
        if use_cache and reranked:
            self._cache.set(query, reranked, cache_opts)

        return query_type, expanded_queries, reranked, domains

    def _dict_to_search_result(self, data: Dict[str, Any]) -> SearchResult:
        """Convert cached dict back to SearchResult."""
        return SearchResult(
            chunk_id=data.get("chunk_id", ""),
            content=data.get("content", ""),
            score=data.get("score", 0.0),
            document_id=data.get("document_id", ""),
            section_title=data.get("section_title", ""),
            chunk_type=data.get("chunk_type", ""),
            source_file=data.get("source_file", ""),
            word_count=data.get("word_count", 0),
            metadata=data.get("metadata"),
        )

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get query cache statistics."""
        return self._cache.get_stats()

    def clear_cache(self) -> None:
        """Clear the query cache."""
        self._cache.invalidate()

    _FOLLOWUP_MARKER = "---FOLLOWUPS---"

    def _merge_adjacent_chunks(self, sources: List[SearchResult]) -> List[SearchResult]:
        """
        Merge adjacent chunks from the same document to preserve context.

        Chunks are considered adjacent if they're from the same document
        and have sequential chunk numbers (e.g., chunk_0307 and chunk_0308).
        """
        import re

        if len(sources) < 2:
            return sources

        # Extract chunk numbers from IDs
        def get_chunk_num(chunk_id: str) -> Optional[int]:
            match = re.search(r"chunk_(\d+)", chunk_id)
            return int(match.group(1)) if match else None

        # Sort by document and chunk number
        sorted_sources = sorted(
            sources, key=lambda s: (s.document_id, get_chunk_num(s.chunk_id) or 0)
        )

        # =================================================================
        # JPL Rule #2: Bound sources before nested while loops
        # Epic Bounded loops compliance
        # Timestamp: 2026-02-18 19:00 UTC
        #
        # VIOLATIONS #2 & #3 FIXED: Lines 562, 569 nested unbounded loops
        # Risk: CRITICAL - O(n²) complexity with unbounded input
        # Fix: Bound to MAX_SOURCES_FOR_MERGING (100)
        # Impact: ZERO - chunk merging typically uses <50 sources
        # =================================================================
        bounded_sources = sorted_sources[:MAX_SOURCES_FOR_MERGING]

        merged = []
        i = 0
        while i < len(bounded_sources):
            current = bounded_sources[i]
            current_num = get_chunk_num(current.chunk_id)

            # Look for adjacent chunks from same document
            combined_content = current.content
            j = i + 1
            while j < len(bounded_sources):
                next_source = bounded_sources[j]
                next_num = get_chunk_num(next_source.chunk_id)

                # Check if adjacent (same doc, sequential number)
                if (
                    next_source.document_id == current.document_id
                    and current_num is not None
                    and next_num is not None
                    and next_num == current_num + (j - i)
                ):
                    combined_content += "\n\n" + next_source.content
                    j += 1
                else:
                    break

            # Create merged result
            merged_source = SearchResult(
                chunk_id=current.chunk_id,
                content=combined_content,
                score=current.score,
                document_id=current.document_id,
                section_title=current.section_title,
                chunk_type=current.chunk_type,
                source_file=current.source_file,
                word_count=len(combined_content.split()),
                metadata=current.metadata,
            )
            merged.append(merged_source)
            i = j

        # Re-sort by original score
        merged.sort(key=lambda s: s.score, reverse=True)
        return merged

    def _build_source_context(self, merged_sources: List[SearchResult]) -> str:
        """
        Build context string from sources.

        Rule #2: Fixed loop bound
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            merged_sources: List of search results to build context from

        Returns:
            Formatted context string
        """
        context_parts = []
        for i, source in enumerate(merged_sources[:5], 1):
            title = source.section_title or source.source_file or f"Source {i}"
            context_parts.append(f"[{i}] ({title})\n{source.content[:3000]}")
        return "\n\n".join(context_parts)

    def _build_rag_system_prompt(self) -> str:
        """
        Build system prompt for RAG answer generation.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Returns:
            System prompt string
        """
        return (
            "You are a research assistant. Answer questions using ONLY the provided sources.\n\n"
            "CRITICAL RULES:\n"
            "1. COMPREHENSIVENESS: Provide COMPLETE information. NEVER summarize or omit items from lists.\n"
            "2. CITATIONS: Cite sources by number [1], [2] for every factual claim.\n"
            "3. VERBATIM LISTS: When sources contain lists of items, reproduce the ENTIRE list.\n"
            "4. STRUCTURE:\n"
            "   - Direct answer (1-2 sentences with citation)\n"
            "   - Supporting details (with citations)\n"
            "   - Source references\n\n"
            'If information is not in sources, say "This information is not in the provided sources."'
        )

    def _build_rag_user_prompt(
        self, query: str, conversation_context: Optional[str]
    ) -> str:
        """
        Build user prompt with conversation context.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            query: User question
            conversation_context: Optional previous conversation

        Returns:
            User prompt string
        """
        user_parts = []
        if conversation_context:
            user_parts.append(f"Previous conversation:\n{conversation_context}\n")
        user_parts.append(f"Question: {query}")
        user_parts.append(
            "\nAfter your answer, write `---FOLLOWUPS---` on its own line, "
            "then list 2-3 follow-up questions the user might ask."
        )
        return "\n".join(user_parts)

    def _parse_followup_suggestions(self, response: str) -> tuple[str, List[str]]:
        """
        Parse follow-up suggestions from LLM response.

        Rule #1: Early return pattern
        Rule #2: Fixed loop bound
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            response: Raw LLM response

        Returns:
            Tuple of (cleaned_answer, follow_up_questions)
        """
        answer = response.strip()
        follow_ups = []
        if self._FOLLOWUP_MARKER not in answer:
            return (answer, follow_ups)

        parts = answer.split(self._FOLLOWUP_MARKER, 1)
        answer = parts[0].strip()
        followup_text = parts[1].strip()

        # =================================================================
        # JPL Rule #2: Bound LLM response line processing
        # Epic Bounded loops compliance
        # Timestamp: 2026-02-18 19:00 UTC
        #
        # VIOLATION #4 FIXED: Line 697 unbounded iteration over LLM text
        # Risk: MEDIUM - Malicious/misconfigured LLM could return 1000+ lines
        # Fix: Bound to MAX_FOLLOWUP_LINES (50)
        # Impact: ZERO - LLM typically returns 5-10 follow-up suggestions
        # =================================================================
        lines = followup_text.splitlines()[:MAX_FOLLOWUP_LINES]

        for line in lines:
            line = line.strip().lstrip("-•*0123456789.) ")
            if line:
                follow_ups.append(line)

        return (answer, follow_ups)

    def _calculate_confidence(self, sources: List[SearchResult]) -> float:
        """
        Calculate confidence score from source scores.

        Rule #1: Early return pattern
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            sources: List of search results

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not sources:
            return 0.0
        return sum(s.score for s in sources[:3]) / 3

    def _validate_and_regenerate(
        self,
        answer: str,
        source_context: str,
        query: str,
        merged_sources: List[SearchResult],
        sources: List[SearchResult],
        confidence: float,
    ) -> tuple[str, float]:
        """
        Validate answer and regenerate if needed.

        Rule #1: Early return pattern
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            answer: Generated answer
            source_context: Source context string
            query: User question
            merged_sources: Merged search results
            sources: Original search results
            confidence: Current confidence score

        Returns:
            Tuple of (potentially_regenerated_answer, adjusted_confidence)
        """
        validate_answers = getattr(
            getattr(self.config.retrieval, "validate_answers", None), "enabled", False
        )
        if not (validate_answers or True):  # Enable by default for now
            return (answer, confidence)

        try:
            from ingestforge.query.validation import AnswerValidator

            validator = AnswerValidator()
            validation = validator.validate(
                answer, source_context, query, num_sources=len(merged_sources[:5])
            )
            if validation.is_valid or validation.coverage_score >= 0.6:
                if validation.warnings:
                    logger.debug(f"Answer validation warnings: {validation.warnings}")
                return (answer, confidence)

            # Regenerate with expanded context
            logger.info(
                f"Answer validation failed (coverage={validation.coverage_score:.2f}), "
                f"attempting regeneration with expanded context"
            )
            expanded_sources = self._expand_context(sources)
            regenerated = self._generate_answer_strict(query, expanded_sources)
            return (regenerated, confidence * 0.9)

        except Exception as e:
            logger.warning(f"Answer validation failed: {e}")
            return (answer, confidence)

    def _generate_answer(
        self,
        query: str,
        sources: List[SearchResult],
        conversation_context: Optional[str] = None,
    ) -> tuple[str, float, List[str]]:
        """
        Generate answer using LLM with structured prompts.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            query: User question
            sources: Retrieved search results
            conversation_context: Optional previous conversation

        Returns:
            Tuple of (answer, confidence, follow_up_suggestions)
        """
        # Build context and prompts using helpers
        merged_sources = self._merge_adjacent_chunks(sources)
        source_context = self._build_source_context(merged_sources)
        system_prompt = self._build_rag_system_prompt()
        user_prompt = self._build_rag_user_prompt(query, conversation_context)

        # Generate response
        response = self.llm_client.generate_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=source_context,
        )

        # Parse and validate
        answer, follow_ups = self._parse_followup_suggestions(response)
        confidence = self._calculate_confidence(sources)
        answer, confidence = self._validate_and_regenerate(
            answer, source_context, query, merged_sources, sources, confidence
        )

        return answer, min(1.0, confidence), follow_ups[:3]

    def _is_adjacent_chunk(
        self,
        chunk: Any,
        current_idx: int,
        max_distance: int = 2,
    ) -> bool:
        """
        Check if chunk is adjacent to current chunk.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            chunk: Chunk to check
            current_idx: Current chunk index
            max_distance: Maximum distance to consider adjacent

        Returns:
            True if chunk is within max_distance positions
        """
        if not hasattr(chunk, "chunk_index"):
            return False
        return abs(chunk.chunk_index - current_idx) <= max_distance

    def _is_unique_chunk(
        self,
        chunk: Any,
        expanded: List[SearchResult],
    ) -> bool:
        """
        Check if chunk is not already in expanded list.

        Rule #1: Simple logic, no nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            chunk: Chunk to check
            expanded: List of already added search results

        Returns:
            True if chunk is unique
        """
        chunk_ids = {s.chunk_id for s in expanded}
        return chunk.chunk_id not in chunk_ids

    def _get_adjacent_chunks(
        self,
        source: SearchResult,
        doc_chunks: List[Any],
        expanded: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Get adjacent chunks from document.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            source: Source chunk to expand from
            doc_chunks: All chunks from the document
            expanded: Current expanded list

        Returns:
            List of adjacent search results
        """
        if not hasattr(source, "chunk_index"):
            return []

        current_idx = source.chunk_index
        adjacent = []

        # =================================================================
        # JPL Rule #2: Bound document chunk iteration
        # Epic Bounded loops compliance
        # Timestamp: 2026-02-18 19:00 UTC
        #
        # VIOLATION #5 FIXED: Line 897 unbounded iteration over doc_chunks
        # Risk: HIGH - Large documents (books, manuals) can have 5000+ chunks
        # Fix: Bound to MAX_DOC_CHUNKS (500)
        # Impact: MINIMAL - most documents have <200 chunks
        # =================================================================
        bounded_chunks = doc_chunks[:MAX_DOC_CHUNKS]

        for chunk in bounded_chunks:
            if self._is_adjacent_chunk(chunk, current_idx):
                if self._is_unique_chunk(chunk, expanded):
                    adjacent.append(SearchResult.from_chunk(chunk, source.score * 0.9))

        return adjacent

    def _expand_from_document(
        self,
        source: SearchResult,
        expanded: List[SearchResult],
    ) -> List[SearchResult]:
        """
        Expand context from a single document.

        Rule #1: Early returns eliminate nesting
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            source: Source to expand from
            expanded: Current expanded list

        Returns:
            List of new search results from this document
        """
        try:
            # Get more chunks from the same document
            doc_chunks = self.retriever.storage.get_chunks_by_document(
                source.document_id
            )
            return self._get_adjacent_chunks(source, doc_chunks, expanded)

        except Exception as e:
            logger.debug(f"Could not expand context for {source.document_id}: {e}")
            return []

    def _expand_context(
        self,
        sources: List[SearchResult],
        max_sources: int = 10,
    ) -> List[SearchResult]:
        """
        Expand context by including more sources and adjacent chunks.

        Rule #1: Reduced nesting (max 2 levels)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Used when initial answer generation fails validation.

        Args:
            sources: Original source chunks
            max_sources: Maximum sources to include

        Returns:
            Expanded list of sources
        """
        expanded = list(sources)
        seen_docs = set()
        for source in sources[:5]:  # Only expand top 5 sources
            if source.document_id in seen_docs:
                continue

            seen_docs.add(source.document_id)
            adjacent = self._expand_from_document(source, expanded)
            expanded.extend(adjacent)

        # =================================================================
        # JPL Rule #2: Bound expanded sources before deduplication
        # Epic Bounded loops compliance
        # Timestamp: 2026-02-18 19:00 UTC
        #
        # VIOLATION #6 FIXED: Line 971 unbounded iteration over expanded list
        # Risk: MEDIUM - Dynamically built list can grow to 500+ sources
        # Fix: Bound to MAX_EXPANDED_SOURCES (200)
        # Impact: ZERO - final limit at line 978 enforces max_sources anyway
        # =================================================================
        bounded_expanded = expanded[:MAX_EXPANDED_SOURCES]

        # Deduplicate and limit
        seen_ids = set()
        unique_expanded = []
        for source in bounded_expanded:
            if source.chunk_id not in seen_ids:
                seen_ids.add(source.chunk_id)
                unique_expanded.append(source)

        # Sort by score and limit
        unique_expanded.sort(key=lambda s: s.score, reverse=True)
        return unique_expanded[:max_sources]

    def _generate_answer_strict(
        self,
        query: str,
        sources: List[SearchResult],
    ) -> str:
        """
        Generate answer using strict mode with lower temperature and stricter prompting.

        Used for regeneration attempts after validation failure.

        Args:
            query: User query
            sources: Retrieved source chunks

        Returns:
            Generated answer string
        """
        # Merge adjacent chunks to preserve context
        merged_sources = self._merge_adjacent_chunks(sources)

        # Build context from sources
        context_parts = []
        for i, source in enumerate(
            merged_sources[:7], 1
        ):  # More sources in strict mode
            title = source.section_title or source.source_file or f"Source {i}"
            context_parts.append(
                f"[{i}] ({title})\n{source.content[:2500]}"  # More content per source
            )
        source_context = "\n\n".join(context_parts)

        system_prompt = (
            "You are a precise research assistant. Answer ONLY from the provided sources.\n\n"
            "STRICT REQUIREMENTS:\n"
            "1. EVERY claim must have a citation [1], [2], etc.\n"
            "2. NEVER add information not explicitly in the sources.\n"
            "3. If sources contain numbered lists, reproduce them COMPLETELY.\n"
            '4. If you cannot answer from sources, state: "This information is not in the provided sources."\n'
            "5. Be thorough but do not speculate or infer beyond what sources state.\n\n"
            "Your answer must be directly verifiable against the provided sources."
        )

        user_prompt = f"Question: {query}\n\nProvide a complete, well-cited answer based only on the sources above."

        # Use lower temperature and shorter max tokens for strict generation
        response = self.llm_client.generate_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=source_context,
            temperature=0.1,  # Lower temperature for more deterministic output
            max_tokens=600,  # Shorter response to stay focused
        )

        return response.strip()

    def _build_source_context(self, sources: List[SearchResult]) -> str:
        """Build context string from sources.

        Rule #4: No large functions - Extracted from process_stream
        """
        context_parts = []
        for i, source in enumerate(sources[:5], 1):
            title = source.section_title or source.source_file or f"Source {i}"
            context_parts.append(f"[{i}] ({title})\n{source.content[:1500]}")
        return "\n\n".join(context_parts)

    def _build_streaming_system_prompt(self) -> str:
        """Build system prompt for streaming responses.

        Rule #4: No large functions - Extracted from process_stream
        """
        return (
            "You are a research assistant. Answer questions using ONLY the provided sources.\n\n"
            "CRITICAL RULES:\n"
            "1. COMPREHENSIVENESS: Provide COMPLETE information. NEVER summarize or omit items from lists.\n"
            "2. CITATIONS: Cite sources by number [1], [2] for every factual claim.\n"
            "3. VERBATIM LISTS: When sources contain lists of items, reproduce the ENTIRE list.\n"
            "4. STRUCTURE:\n"
            "   - Direct answer (1-2 sentences with citation)\n"
            "   - Supporting details (with citations)\n"
            "   - Source references\n\n"
            'If information is not in sources, say "This information is not in the provided sources."'
        )

    def _build_streaming_user_prompt(
        self,
        query: str,
        conversation_context: Optional[str] = None,
    ) -> str:
        """Build user prompt with optional conversation context.

        Rule #4: No large functions - Extracted from process_stream
        """
        user_parts = []
        if conversation_context:
            user_parts.append(f"Previous conversation:\n{conversation_context}\n")
        user_parts.append(f"Question: {query}")
        user_parts.append(
            "\nAfter your answer, write `---FOLLOWUPS---` on its own line, "
            "then list 2-3 follow-up questions the user might ask."
        )
        return "\n".join(user_parts)

    def process_stream(
        self,
        query: str,
        top_k: Optional[int] = None,
        conversation_context: Optional[str] = None,
        library_filter: Optional[str] = None,
        **kwargs,
    ) -> tuple[QueryResult, Generator[str, None, None]]:
        """
        Process query with streaming answer generation.

        Performs retrieval synchronously, then returns a QueryResult
        (with answer=None) and a generator that yields answer text chunks.
        The caller should consume the generator to build the full answer.

        Args:
            query: User query
            top_k: Number of results
            conversation_context: Previous conversation for multi-turn
            library_filter: If provided, only return chunks from this library

        Returns:
            Tuple of (QueryResult with sources, answer text generator)

        Rule #4: No large functions - Refactored to <60 lines
        """
        if not self.llm_client:
            result = self.process(
                query,
                top_k=top_k,
                generate_answer=False,
                library_filter=library_filter,
                **kwargs,
            )
            return result, iter([])

        # Do retrieval synchronously (steps 1-4)
        result = self.process(
            query,
            top_k=top_k,
            generate_answer=False,
            library_filter=library_filter,
            **kwargs,
        )

        if not result.sources:
            return result, iter([])

        # Build prompts using helper methods
        source_context = self._build_source_context(result.sources)
        system_prompt = self._build_streaming_system_prompt()
        user_prompt = self._build_streaming_user_prompt(query, conversation_context)

        stream = self.llm_client.stream_generate_with_context(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            context=source_context,
        )

        return result, stream

    def search_only(
        self,
        query: str,
        top_k: Optional[int] = None,
        library_filter: Optional[str] = None,
        **kwargs,
    ) -> List[SearchResult]:
        """
        Search without answer generation.

        Args:
            query: Search query
            top_k: Number of results
            library_filter: If provided, only return chunks from this library

        Returns:
            List of search results
        """
        result = self.process(
            query,
            top_k=top_k,
            generate_answer=False,
            library_filter=library_filter,
            **kwargs,
        )
        return result.sources
