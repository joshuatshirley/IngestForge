"""
Domain Registry and Routing Logic.

Task 109: YAML-based domain registry & rule loader.
Task 110: Zero-shot document classifier (Base Engine).
"""

import yaml
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from pydantic import BaseModel

from ingestforge.core.logging import get_logger

logger = get_logger(__name__)


class DomainSignature(BaseModel):
    pattern: str
    weight: int = 1


class DomainDefinition(BaseModel):
    name: str
    id: str
    description: str = ""
    signatures: List[DomainSignature] = []
    extraction_rules: List[Dict[str, Any]] = []


# JPL Rule #2: Fixed upper bounds
MAX_DOMAINS = 100


class DomainRegistry:
    """
    Registry for document domains loaded from YAML.
    Rule #4: Small scope, manageable logic.
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DomainRegistry, cls).__new__(cls)
            cls._instance._domains = {}
            cls._instance._load_all()
        return cls._instance

    def _load_all(self) -> None:
        """Load all domains from the domains directory."""
        domain_dir = Path(__file__).parent.parent.parent / "domains"
        if not domain_dir.exists():
            logger.warning(f"Domain directory not found: {domain_dir}")
            return

        # JPL Rule #2: Bounded iteration over file system
        files = list(domain_dir.glob("*.yaml"))[:MAX_DOMAINS]
        for file in files:
            try:
                with open(file, "r") as f:
                    data = yaml.safe_load(f)
                    domain = DomainDefinition(**data)
                    self._domains[domain.id] = domain
                    logger.debug(f"Loaded domain: {domain.id}")
            except Exception as e:
                logger.error(f"Failed to load domain from {file}: {e}")

    def list_domains(self) -> List[str]:
        return list(self._domains.keys())

    def get_domain(self, domain_id: str) -> Optional[DomainDefinition]:
        return self._domains.get(domain_id)

    def get_all_signatures(self) -> List[Tuple[str, re.Pattern, int]]:
        """Flatten signatures for the router."""
        sigs = []
        for domain_id, domain in self._domains.items():
            for sig in domain.signatures:
                try:
                    pattern = re.compile(sig.pattern, re.IGNORECASE)
                    sigs.append((domain_id, pattern, sig.weight))
                except re.error:
                    logger.error(f"Invalid regex in domain {domain_id}: {sig.pattern}")
        return sigs


class DomainRouter:
    """
    Heuristic and Zero-shot router.
    Rule #4: Methods < 60 lines.
    """

    def __init__(self, use_llm: bool = False):
        self.registry = DomainRegistry()
        self.use_llm = use_llm

    def classify(self, text: str) -> List[Tuple[str, float]]:
        """
        Classify text into domains.
        Returns a list of (domain_id, score) sorted by score.
        """
        if not text:
            return []

        # 1. Heuristic signatures (Fast)
        scores = self._get_heuristic_scores(text)

        # 2. Zero-shot LLM (Deep) if enabled and heuristic is weak
        if self.use_llm and (not scores or max(scores.values()) < 5):
            llm_scores = self._classify_zero_shot(text)
            for d, s in llm_scores.items():
                scores[d] = scores.get(d, 0) + s

        # Normalize results
        total = sum(scores.values()) or 1
        results = [(d, s / total) for d, s in scores.items()]

        return sorted(results, key=lambda x: x[1], reverse=True)

    def _get_heuristic_scores(self, text: str) -> Dict[str, int]:
        scores = {}
        sigs = self.registry.get_all_signatures()
        for domain_id, pattern, weight in sigs:
            matches = len(pattern.findall(text))
            if matches > 0:
                scores[domain_id] = scores.get(domain_id, 0) + (matches * weight)
        return scores

    def _classify_zero_shot(self, text: str) -> Dict[str, float]:
        """
        Placeholder for LLM-based zero-shot classification.
        Task 110: Implementation of the engine interface.
        """
        # In a real implementation, this would call an LLM with a list of domains
        # and ask it to pick the most likely one based on text summary.
        return {"generic": 0.5}
