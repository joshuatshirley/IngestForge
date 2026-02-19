"""
Agent Domain Router for IngestForge.

Agent-Vertical Awareness.
Bridges domain detection to agent tool activation, enabling the ReAct engine
to automatically use domain-specific tools based on content context.

NASA JPL Power of Ten compliant.
"""

from typing import List, Tuple

from ingestforge.agent.tool_registry import ToolRegistry
from ingestforge.core.logging import get_logger
from ingestforge.enrichment.router import DomainRouter

logger = get_logger(__name__)

# JPL Rule #2: Fixed upper bounds
MAX_DOMAINS = 5
MAX_CONTENT_LENGTH = 32000
MIN_DETECTION_SCORE = 2


class AgentDomainRouter:
    """
    Bridges domain detection to agent tool activation.

    GWT-1: Detects domain and activates appropriate tools.
    GWT-2: Supports multi-domain content.
    GWT-3: Universal tools always remain available.

    Rule #4: Methods < 60 lines.
    Rule #9: Complete type hints.
    """

    def __init__(
        self,
        min_score: int = MIN_DETECTION_SCORE,
        max_domains: int = MAX_DOMAINS,
    ) -> None:
        """
        Initialize the domain router.

        Args:
            min_score: Minimum score for domain detection.
            max_domains: Maximum domains to return.
        """
        assert min_score >= 0, "min_score must be non-negative"
        assert max_domains > 0, "max_domains must be positive"

        self._domain_router = DomainRouter()
        self._min_score = min_score
        self._max_domains = min(max_domains, MAX_DOMAINS)

    def detect_domains(self, content: str) -> List[str]:
        """
        Detect domains from content.

        GWT-1: Detects domain from content.
        GWT-2: Returns multiple domains if detected.

        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.

        Args:
            content: Text content to analyze.

        Returns:
            List of detected domain names (e.g., ["legal", "cyber"]).
        """
        if not content or not content.strip():
            return []

        # JPL Rule #2: Bound content length
        content = content[:MAX_CONTENT_LENGTH]

        # Use existing DomainRouter for classification
        ranked = self._domain_router.classify_chunk(content)

        if not ranked:
            return []

        # Filter by minimum score
        detected: List[str] = []
        top_score = ranked[0][1]

        if top_score < self._min_score:
            logger.debug(
                f"No domains detected (top score {top_score} < {self._min_score})"
            )
            return []

        # Take primary domain
        detected.append(ranked[0][0])

        # Add secondary domains if score is close to primary (>= 80%)
        for domain, score in ranked[1:]:
            if len(detected) >= self._max_domains:
                break
            if score >= (top_score * 0.8) and score >= self._min_score:
                detected.append(domain)

        logger.debug(f"Detected domains: {detected}")
        return detected

    def activate_tools(
        self,
        registry: ToolRegistry,
        content: str,
    ) -> Tuple[int, List[str]]:
        """
        Detect domains and activate appropriate tools.

        GWT-1: Activates domain-specific tools.
        GWT-3: Universal tools remain available.

        Rule #4: Function < 60 lines.
        Rule #5: Assert preconditions.
        Rule #7: Check return values.

        Args:
            registry: Tool registry to activate tools in.
            content: Content to detect domains from.

        Returns:
            Tuple of (enabled_tool_count, detected_domains).
        """
        assert registry is not None, "registry cannot be None"

        # Detect domains from content
        domains = self.detect_domains(content)

        if not domains:
            # No domains detected - enable only universal tools
            enabled = registry.activate_for_domains([])
            logger.info(f"No domains detected, activated {enabled} universal tools")
            return (enabled, [])

        # Activate tools for detected domains
        enabled = registry.activate_for_domains(domains)
        logger.info(f"Activated {enabled} tools for domains: {domains}")

        return (enabled, domains)

    def get_active_tool_summary(
        self,
        registry: ToolRegistry,
        content: str,
    ) -> str:
        """
        Get summary of active tools after domain detection.

        Useful for including in LLM prompts.

        Rule #4: Function < 60 lines.

        Args:
            registry: Tool registry.
            content: Content to detect domains from.

        Returns:
            Formatted string describing active tools.
        """
        enabled_count, domains = self.activate_tools(registry, content)

        lines = []

        if domains:
            lines.append(f"Detected domains: {', '.join(domains)}")
        else:
            lines.append("No specific domain detected.")

        lines.append(f"Active tools: {enabled_count}")
        lines.append("")
        lines.append("Available tools:")
        lines.append(registry.get_prompt_description())

        return "\n".join(lines)

    def get_domains_with_scores(
        self,
        content: str,
    ) -> List[Tuple[str, int]]:
        """
        Get detected domains with their scores.

        Useful for debugging and transparency.

        Args:
            content: Content to analyze.

        Returns:
            List of (domain, score) tuples.
        """
        if not content or not content.strip():
            return []

        content = content[:MAX_CONTENT_LENGTH]
        ranked = self._domain_router.classify_chunk(content)

        if not ranked:
            return []

        # Return top domains above threshold
        result: List[Tuple[str, int]] = []
        for domain, score in ranked[: self._max_domains]:
            if score >= self._min_score:
                result.append((domain, score))

        return result


# ---------------------------------------------------------------------------
# Convenience Functions
# ---------------------------------------------------------------------------


def create_agent_domain_router(
    min_score: int = MIN_DETECTION_SCORE,
) -> AgentDomainRouter:
    """
    Create an AgentDomainRouter instance.

    Args:
        min_score: Minimum score for domain detection.

    Returns:
        Configured AgentDomainRouter.
    """
    return AgentDomainRouter(min_score=min_score)


def activate_domain_tools(
    registry: ToolRegistry,
    content: str,
    min_score: int = MIN_DETECTION_SCORE,
) -> Tuple[int, List[str]]:
    """
    Convenience function to detect domains and activate tools.

    Args:
        registry: Tool registry.
        content: Content to analyze.
        min_score: Minimum detection score.

    Returns:
        Tuple of (enabled_count, detected_domains).
    """
    router = AgentDomainRouter(min_score=min_score)
    return router.activate_tools(registry, content)
