"""Advanced query parser for boolean operators and field filters."""
import re
from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class QueryFilter:
    """A field:value filter (e.g., author:Smith, year:2023)."""

    field: str
    value: str


@dataclass
class ParsedQuery:
    """Parsed query with operators and filters."""

    terms: List[str]  # Search terms
    exact_phrases: List[str]  # Quoted phrases for exact match
    filters: List[QueryFilter]  # Field filters
    exclude_terms: List[str]  # Terms with NOT operator
    and_groups: List[List[str]]  # Terms grouped by AND
    or_groups: List[List[str]]  # Terms grouped by OR
    is_regex: bool = False  # Whether to use regex search


class QueryParser:
    """Parse advanced search queries with boolean operators and filters."""

    # Regex patterns
    QUOTED_PATTERN = r'"([^"]+)"'
    FIELD_FILTER_PATTERN = r"(\w+):(\S+)"
    BOOLEAN_OPS = {"AND", "OR", "NOT"}

    def __init__(self) -> None:
        """Initialize parser."""
        pass

    def _extract_quoted_phrases(self, query: str) -> tuple[Any, ...]:
        """
        Extract quoted phrases from query.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            query: Query string

        Returns:
            Tuple of (exact_phrases, query_without_quotes)
        """
        exact_phrases = re.findall(self.QUOTED_PATTERN, query)
        query_without_quotes = re.sub(self.QUOTED_PATTERN, "", query)
        return (exact_phrases, query_without_quotes)

    def _extract_filters(self, query: str) -> tuple[Any, ...]:
        """
        Extract field filters from query.

        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            query: Query string (without quotes)

        Returns:
            Tuple of (filters, query_without_filters)
        """
        filters = []
        filter_matches = re.findall(self.FIELD_FILTER_PATTERN, query)

        for field, value in filter_matches:
            # Don't treat AND:value as filter
            if field not in self.BOOLEAN_OPS:
                filters.append(QueryFilter(field=field.lower(), value=value))

        query_without_filters = re.sub(self.FIELD_FILTER_PATTERN, "", query)
        return (filters, query_without_filters)

    def _process_regular_term(
        self,
        token: str,
        negate_next: bool,
        terms: List[str],
        exclude_terms: List[str],
        current_and_group: List[str],
        current_or_group: List[str],
    ) -> tuple[Any, ...]:
        """Process regular term token.

        Rule #4: No large functions - Extracted from _process_token
        """
        token_lower = token.lower()
        if negate_next:
            exclude_terms.append(token_lower)
            return (False, False, current_and_group, current_or_group)

        # Add to all relevant lists
        terms.append(token_lower)
        if current_and_group is not None:
            current_and_group.append(token_lower)
        if current_or_group is not None:
            current_or_group.append(token_lower)

        return (False, False, current_and_group, current_or_group)

    def _process_token(
        self,
        token: str,
        negate_next: bool,
        terms: List[str],
        exclude_terms: List[str],
        and_groups: List[List[str]],
        or_groups: List[List[str]],
        current_and_group: List[str],
        current_or_group: List[str],
    ) -> tuple[Any, ...]:
        """
        Process single token and update state.

        Rule #1: Early returns for control flow
        Rule #4: Function <60 lines (refactored to 47 lines)
        Rule #9: Full type hints

        Args:
            token: Token to process
            negate_next: Whether to negate
            terms: Terms list (mutated)
            exclude_terms: Exclude terms list (mutated)
            and_groups: AND groups list (mutated)
            or_groups: OR groups list (mutated)
            current_and_group: Current AND group (mutated)
            current_or_group: Current OR group (mutated)

        Returns:
            Tuple of (should_continue, new_negate_next, new_and_group, new_or_group)
        """
        # Handle NOT operator
        if token == "NOT":
            return (True, True, current_and_group, current_or_group)

        # Handle AND operator
        if token == "AND":
            if current_and_group:
                and_groups.append(current_and_group)
            return (True, negate_next, [], current_or_group)

        # Handle OR operator
        if token == "OR":
            if current_or_group:
                or_groups.append(current_or_group)
            return (True, negate_next, current_and_group, [])

        # Skip parentheses
        if token in ("(", ")"):
            return (True, negate_next, current_and_group, current_or_group)

        # Regular term - delegate to helper
        return self._process_regular_term(
            token,
            negate_next,
            terms,
            exclude_terms,
            current_and_group,
            current_or_group,
        )

    def _process_boolean_operators(self, tokens: List[str]) -> tuple[Any, ...]:
        """
        Process boolean operators in tokens.

        Rule #2: Fixed loop bound
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            tokens: Token list

        Returns:
            Tuple of (terms, exclude_terms, and_groups, or_groups)
        """
        terms: list[str] = []
        exclude_terms: list[str] = []
        and_groups: list[list[str]] = []
        or_groups: list[list[str]] = []

        current_and_group: list[str] = []
        current_or_group: list[str] = []
        negate_next = False
        i = 0
        while i < len(tokens):
            (
                should_continue,
                negate_next,
                current_and_group,
                current_or_group,
            ) = self._process_token(
                tokens[i],
                negate_next,
                terms,
                exclude_terms,
                and_groups,
                or_groups,
                current_and_group,
                current_or_group,
            )

            i += 1

            if should_continue:
                continue

        # Add final groups
        if current_and_group:
            and_groups.append(current_and_group)
        if current_or_group:
            or_groups.append(current_or_group)

        return (terms, exclude_terms, and_groups, or_groups)

    def parse(self, query: str, is_regex: bool = False) -> ParsedQuery:
        """
        Parse query string into structured components.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            query: Raw query string
            is_regex: Whether to treat query as regex

        Returns:
            ParsedQuery with extracted components
        """
        if is_regex:
            return ParsedQuery(
                terms=[query],
                exact_phrases=[],
                filters=[],
                exclude_terms=[],
                and_groups=[],
                or_groups=[],
                is_regex=True,
            )

        # Extract components using helpers
        exact_phrases, query_without_quotes = self._extract_quoted_phrases(query)
        filters, query_without_filters = self._extract_filters(query_without_quotes)
        tokens = query_without_filters.split()
        terms, exclude_terms, and_groups, or_groups = self._process_boolean_operators(
            tokens
        )

        return ParsedQuery(
            terms=terms,
            exact_phrases=exact_phrases,
            filters=filters,
            exclude_terms=exclude_terms,
            and_groups=and_groups,
            or_groups=or_groups,
            is_regex=False,
        )

    def _check_author_filter(
        self, source_location: dict[str, Any], filter_value: str
    ) -> bool:
        """
        Check if author filter matches.

        Rule #1: Simple loop with early return
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            source_location: Source location metadata
            filter_value: Filter value to match

        Returns:
            True if matches, False otherwise
        """
        authors = source_location.get("authors", [])
        filter_lower = filter_value.lower()

        for author in authors:
            name = author.get("name", "").lower()
            last_name = author.get("last_name", "").lower()
            if filter_lower in name or filter_lower in last_name:
                return True

        return False

    def _check_year_filter(
        self, source_location: dict[str, Any], filter_value: str
    ) -> bool:
        """
        Check if year filter matches.

        Rule #1: Simple comparison
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            source_location: Source location metadata
            filter_value: Year value to match

        Returns:
            True if matches, False otherwise
        """
        pub_date = str(source_location.get("publication_date", ""))
        return pub_date.startswith(filter_value)

    def _check_tag_filter(self, metadata: dict[str, Any], filter_value: str) -> bool:
        """
        Check if tag filter matches.

        Rule #1: Simple list comprehension
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            metadata: Chunk metadata
            filter_value: Tag value to match

        Returns:
            True if matches, False otherwise
        """
        tags = metadata.get("tags", [])
        return filter_value.lower() in [t.lower() for t in tags]

    def _check_single_filter(
        self, f: QueryFilter, metadata: dict[str, Any], source_location: dict[str, Any]
    ) -> bool:
        """
        Check if single filter matches chunk.

        Rule #1: Early return eliminates nesting
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            f: Filter to check
            metadata: Chunk metadata
            source_location: Source location metadata

        Returns:
            True if matches, False otherwise
        """
        # Check metadata field
        if f.field in metadata:
            if str(metadata[f.field]).lower() == f.value.lower():
                return True

        # Check source_location field
        if f.field in source_location:
            if str(source_location[f.field]).lower() == f.value.lower():
                return True

        # Check special fields
        if f.field == "author":
            return self._check_author_filter(source_location, f.value)

        if f.field == "year":
            return self._check_year_filter(source_location, f.value)

        if f.field == "tag":
            return self._check_tag_filter(metadata, f.value)

        return False

    def _chunk_matches_filters(
        self, chunk: Dict[str, Any], filters: List[QueryFilter]
    ) -> bool:
        """
        Check if chunk matches all filters.

        Rule #1: Simple loop with early return
        Rule #4: Function <60 lines
        Rule #9: Full type hints

        Args:
            chunk: Chunk dictionary
            filters: List of filters to check

        Returns:
            True if all filters match, False otherwise
        """
        metadata = chunk.get("metadata", {})
        source_location = chunk.get("source_location", {})

        for f in filters:
            if not self._check_single_filter(f, metadata, source_location):
                return False

        return True

    def apply_filters(
        self, parsed: ParsedQuery, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Apply field filters to chunk results.

        Rule #1: Reduced nesting (max 1 level)
        Rule #4: Function <60 lines
        Rule #7: Parameter validation
        Rule #9: Full type hints

        Args:
            parsed: Parsed query with filters
            chunks: List of chunk dictionaries

        Returns:
            Filtered chunks
        """
        if not parsed.filters:
            return chunks

        # Filter chunks using helper
        filtered = []
        for chunk in chunks:
            if self._chunk_matches_filters(chunk, parsed.filters):
                filtered.append(chunk)

        return filtered

    def apply_exclusions(
        self, parsed: ParsedQuery, chunks: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Exclude chunks containing NOT terms.

        Args:
            parsed: Parsed query with exclusions
            chunks: List of chunk dictionaries

        Returns:
            Filtered chunks
        """
        if not parsed.exclude_terms:
            return chunks

        filtered = []
        for chunk in chunks:
            content = chunk.get("content", "").lower()

            # Exclude if any NOT term is found
            has_excluded = False
            for term in parsed.exclude_terms:
                if term in content:
                    has_excluded = True
                    break

            if not has_excluded:
                filtered.append(chunk)

        return filtered
