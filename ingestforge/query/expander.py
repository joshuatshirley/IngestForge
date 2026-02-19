"""
Query expansion for improved recall.

Expand queries with synonyms and related terms.
"""

import re
from typing import Dict, List, Optional, Set


class QueryExpander:
    """
    Expand queries for improved retrieval recall.

    Techniques:
    - Synonym expansion
    - Acronym expansion
    - Question paraphrasing
    """

    def __init__(self) -> None:
        # Common synonym mappings
        self.synonyms: Dict[str, List[str]] = {
            "requirements": [
                "requirements",
                "prerequisites",
                "qualifications",
                "criteria",
            ],
            "process": ["process", "procedure", "steps", "workflow"],
            "how to": ["how to", "how do I", "steps to", "guide to"],
            "maximum": ["maximum", "max", "upper limit", "highest"],
            "minimum": ["minimum", "min", "lower limit", "lowest"],
            "obtain": ["obtain", "get", "acquire", "receive"],
            "submit": ["submit", "send", "file", "provide"],
            "document": ["document", "documentation", "paperwork", "form"],
            "eligible": ["eligible", "qualify", "meet requirements"],
            "deadline": ["deadline", "due date", "cutoff", "time limit"],
        }

        # Common acronyms (domain-agnostic)
        self.acronyms: Dict[str, str] = {
            "FAQ": "frequently asked questions",
            "ETA": "estimated time of arrival",
            "ASAP": "as soon as possible",
            "TBD": "to be determined",
            "N/A": "not applicable",
            "FYI": "for your information",
        }

    def expand(
        self,
        query: str,
        query_type: Optional[str] = None,
        max_expansions: int = 3,
    ) -> List[str]:
        """
        Expand query into alternative formulations.

        Args:
            query: Original query
            query_type: Optional query classification
            max_expansions: Maximum number of expansions

        Returns:
            List of expanded queries
        """
        expansions: Set[str] = set()

        # Synonym expansion
        syn_expanded = self._expand_synonyms(query)
        if syn_expanded != query:
            expansions.add(syn_expanded)

        # Acronym expansion
        acr_expanded = self._expand_acronyms(query)
        if acr_expanded != query:
            expansions.add(acr_expanded)

        # Question paraphrasing
        paraphrases = self._generate_paraphrases(query, query_type)
        expansions.update(paraphrases)

        # Remove original query and limit
        expansions.discard(query.lower())
        return list(expansions)[:max_expansions]

    def _expand_synonyms(self, query: str) -> str:
        """Replace terms with synonyms."""
        result = query.lower()

        for term, syns in self.synonyms.items():
            if term in result and len(syns) > 1:
                # Replace with second synonym (first is original)
                result = result.replace(term, syns[1])
                break  # Only one replacement per query

        return result

    def _expand_acronyms(self, query: str) -> str:
        """Expand acronyms in query."""
        result = query

        for acronym, expansion in self.acronyms.items():
            pattern = r"\b" + re.escape(acronym) + r"\b"
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
                break

        return result

    def _paraphrase_procedural(self, base: str) -> List[str]:
        """
        Generate paraphrases for procedural queries.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            base: Query without question mark

        Returns:
            List of paraphrases
        """
        assert base is not None, "Base query cannot be None"
        assert isinstance(base, str), "Base query must be string"

        paraphrases: List[str] = []
        if base.startswith("how to "):
            paraphrases.append("steps to " + base[7:])
            paraphrases.append("process for " + base[7:])
            return paraphrases

        if base.startswith("how do i "):
            paraphrases.append("how to " + base[9:])
            paraphrases.append("steps to " + base[9:])
            return paraphrases

        return paraphrases

    def _paraphrase_factual(self, base: str) -> List[str]:
        """
        Generate paraphrases for factual queries.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            base: Query without question mark

        Returns:
            List of paraphrases
        """
        assert base is not None, "Base query cannot be None"
        assert isinstance(base, str), "Base query must be string"

        paraphrases: List[str] = []
        if base.startswith("what is "):
            paraphrases.append("define " + base[8:])
            paraphrases.append(base[8:] + " definition")
            return paraphrases

        if base.startswith("what are "):
            paraphrases.append("list of " + base[9:])
            return paraphrases

        return paraphrases

    def _paraphrase_conceptual(self, base: str) -> List[str]:
        """
        Generate paraphrases for conceptual queries.

        Rule #1: Extracted helper reduces nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            base: Query without question mark

        Returns:
            List of paraphrases
        """
        assert base is not None, "Base query cannot be None"
        assert isinstance(base, str), "Base query must be string"

        paraphrases: List[str] = []
        if base.startswith("explain "):
            paraphrases.append("what is " + base[8:])
            paraphrases.append(base[8:] + " overview")
            return paraphrases

        return paraphrases

    def _add_question_marks(
        self,
        paraphrases: List[str],
        has_question_mark: bool,
    ) -> List[str]:
        """
        Add question marks back to paraphrases if original had one.

        Rule #1: Extracted helper eliminates nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            paraphrases: List of paraphrases
            has_question_mark: Whether original query had question mark

        Returns:
            Paraphrases with question marks added if needed
        """
        assert paraphrases is not None, "Paraphrases cannot be None"
        assert isinstance(paraphrases, list), "Paraphrases must be list"
        if not has_question_mark:
            return paraphrases
        result = [p + "?" for p in paraphrases]
        assert all(p.endswith("?") for p in result), "All paraphrases must end with '?'"
        return result

    def _generate_paraphrases(
        self,
        query: str,
        query_type: Optional[str],
    ) -> List[str]:
        """
        Generate query paraphrases based on type.

        Rule #1: Zero nesting - dictionary dispatch with helpers
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            query: Original query
            query_type: Query classification (procedural, factual, conceptual)

        Returns:
            List of paraphrased queries
        """
        assert query is not None, "Query cannot be None"
        assert isinstance(query, str), "Query must be string"
        query_lower: str = query.lower().strip()
        base: str = query_lower.rstrip("?")
        has_question_mark: bool = query.endswith("?")
        paraphrase_generators = {
            "procedural": self._paraphrase_procedural,
            "factual": self._paraphrase_factual,
            "conceptual": self._paraphrase_conceptual,
        }
        if query_type not in paraphrase_generators:
            return []
        generator = paraphrase_generators[query_type]
        paraphrases = generator(base)
        result = self._add_question_marks(paraphrases, has_question_mark)
        assert isinstance(result, list), "Result must be a list"

        return result
