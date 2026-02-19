"""
Metadata Serialization Utilities.

This module provides centralized JSON serialization for complex metadata types
that can't be directly serialized. By handling serialization here, storage
backends don't need to duplicate this logic.

Architecture Context
--------------------
Serialization is used when persisting chunks to storage:

    ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
    │  ChunkRecord    │────→│  Serialization  │────→│    Storage      │
    │ + SourceLocation│     │  serialize_*()  │     │  (JSONL/Chroma) │
    │ + datetime      │     └─────────────────┘     └─────────────────┘
    │ + Path          │
    └─────────────────┘

Types Requiring Serialization
-----------------------------
1. **SourceLocation**: Dataclass with nested Author objects
2. **Path**: Needs conversion to string
3. **datetime**: Needs ISO format string
4. **Enum**: Needs .value extraction
5. **Nested dicts**: Need recursive handling

Functions
---------
**serialize_source_location(source_location)**
    Convert SourceLocation dataclass to JSON-compatible dict.

**deserialize_source_location(data, location_class)**
    Reconstruct SourceLocation from dict.

**serialize_metadata(metadata)**
    Generic serializer handling Path, datetime, dataclasses.

**clean_metadata_for_storage(metadata)**
    Remove None values and empty collections to save space.

**merge_metadata(base, updates)**
    Deep merge dictionaries with update priority.

Usage Pattern
-------------
Storage backends use these utilities:

    class JSONLStorage:
        def save_chunk(self, chunk: Any) -> None:
            metadata = serialize_metadata({
                "source_location": chunk.source_location,
                "processed_at": datetime.now(),
                "file_path": chunk.source_path,
            })
            # metadata is now JSON-serializable

    class ChromaDBStorage:
        def load_chunk(self, doc: Any) -> None:
            source_location = deserialize_source_location(
                doc["metadata"]["source_location"],
                SourceLocation
            )

Design Decisions
----------------
1. **Centralized**: All serialization logic in one place.
2. **Type-aware**: Handles dataclasses, Path, datetime, etc.
3. **Defensive**: clean_metadata removes nulls to save storage space.
4. **Round-trip safe**: serialize → deserialize produces equivalent objects.
"""

from typing import Any, Callable, Dict, List
from dataclasses import asdict, is_dataclass
from pathlib import Path


def serialize_source_location(source_location: Any) -> Dict[str, Any]:
    """Serialize a SourceLocation dataclass to a JSON-compatible dict.

    Args:
        source_location: SourceLocation dataclass instance

    Returns:
        Dictionary representation suitable for JSON serialization

    Examples:
        >>> from ingestforge.core.provenance import SourceLocation
        >>> loc = SourceLocation(
        ...     file_path="document.pdf",
        ...     page_number=1,
        ...     chunk_index=0
        ... )
        >>> serialize_source_location(loc)
        {'file_path': 'document.pdf', 'page_number': 1, 'chunk_index': 0, ...}
    """
    if source_location is None:
        return {}

    if is_dataclass(source_location):
        result_dict: dict[str, Any] = asdict(source_location)
        return result_dict

    # Fallback for dict-like objects
    if hasattr(source_location, "__dict__"):
        result_vars: dict[str, Any] = vars(source_location)
        return result_vars

    return {}


def deserialize_source_location(data: Dict[str, Any], location_class: type) -> Any:
    """Deserialize a dict to a SourceLocation dataclass.

    Args:
        data: Dictionary with source location data
        location_class: The dataclass type to instantiate (e.g., SourceLocation)

    Returns:
        Instance of location_class

    Examples:
        >>> from ingestforge.core.provenance import SourceLocation
        >>> data = {'file_path': 'document.pdf', 'page_number': 1, 'chunk_index': 0}
        >>> loc = deserialize_source_location(data, SourceLocation)
        >>> loc.file_path
        'document.pdf'
    """
    if not data:
        return None

    # Filter only fields that exist in the dataclass
    if is_dataclass(location_class):
        import inspect

        sig = inspect.signature(location_class)
        valid_fields = {k: v for k, v in data.items() if k in sig.parameters}
        return location_class(**valid_fields)

    return location_class(**data)


def _serialize_primitive(value: Any) -> Any:
    """
    Serialize primitive types (str, int, float, bool, None).

    Rule #1: Extract type handler to reduce nesting
    Rule #4: Function <60 lines
    """
    return value


def _serialize_path(value: Path) -> str:
    """
    Serialize Path to string.

    Rule #1: Extract type handler
    Rule #9: Type hints
    """
    return str(value)


def _serialize_datetime(value: Any) -> str:
    """
    Serialize datetime to ISO format.

    Rule #1: Extract type handler
    Rule #9: Type hints
    """
    result: str = value.isoformat()
    return result


def _serialize_dataclass(value: Any) -> Dict[str, Any]:
    """
    Serialize dataclass to dict.

    Rule #1: Extract type handler
    Rule #9: Type hints
    """
    return asdict(value)


def _serialize_sequence(value: Any) -> List[Any]:
    """
    Serialize list/tuple recursively.

    Rule #1: Extract type handler
    Rule #9: Type hints
    """
    return [serialize_metadata({"item": item})["item"] for item in value]


def _serialize_dict(value: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize dict recursively.

    Rule #1: Extract type handler
    Rule #9: Type hints
    """
    return serialize_metadata(value)


def _get_serializer_for_value(value: Any) -> Callable[[Any], Any]:
    """
    Get appropriate serializer function for a value.

    Rule #1: Dictionary dispatch eliminates nesting
    Rule #4: Function <60 lines
    Rule #9: Full type hints
    """
    from datetime import datetime
    from pathlib import Path

    if value is None or isinstance(value, (str, int, float, bool)):
        return _serialize_primitive

    # Type-based dispatch
    if isinstance(value, Path):
        return _serialize_path
    if isinstance(value, datetime):
        return _serialize_datetime
    if is_dataclass(value):
        return _serialize_dataclass
    if isinstance(value, (list, tuple)):
        return _serialize_sequence
    if isinstance(value, dict):
        return _serialize_dict

    # Fallback to string conversion
    return str


def serialize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Serialize metadata dict to JSON-compatible format.

    Rule #1: Simple control flow (max 1 nesting level)
    Rule #4: Function <60 lines
    Rule #8: Clear abstraction via type handlers
    Rule #9: Full type hints

    Handles common types like SourceLocation, datetime, Path, etc.

    Args:
        metadata: Dictionary with metadata

    Returns:
        JSON-compatible dictionary

    Examples:
        >>> from pathlib import Path
        >>> from datetime import datetime
        >>> metadata = {
        ...     'file_path': Path('document.pdf'),
        ...     'timestamp': datetime.now(),
        ...     'count': 42
        ... }
        >>> serialize_metadata(metadata)
        {'file_path': 'document.pdf', 'timestamp': '2024-...', 'count': 42}
    """
    result: dict[str, Any] = {}
    for key, value in metadata.items():
        serializer = _get_serializer_for_value(value)
        result[key] = serializer(value)

    return result


def clean_metadata_for_storage(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Clean metadata for storage by removing None values and empty dicts.

    Args:
        metadata: Dictionary with metadata

    Returns:
        Cleaned metadata dictionary

    Examples:
        >>> clean_metadata_for_storage({'a': 1, 'b': None, 'c': {}, 'd': []})
        {'a': 1}
    """
    cleaned = {}

    for key, value in metadata.items():
        # Skip None values
        if value is None:
            continue

        # Skip empty collections
        if isinstance(value, (dict, list, tuple, set)) and not value:
            continue

        # Recursively clean nested dicts
        if isinstance(value, dict):
            cleaned_nested = clean_metadata_for_storage(value)
            if cleaned_nested:
                cleaned[key] = cleaned_nested
        else:
            cleaned[key] = value

    return cleaned


def merge_metadata(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Merge metadata dictionaries, with updates taking precedence.

    Args:
        base: Base metadata dictionary
        updates: Updates to apply

    Returns:
        Merged metadata dictionary

    Examples:
        >>> base = {'a': 1, 'b': 2, 'nested': {'x': 10}}
        >>> updates = {'b': 3, 'c': 4, 'nested': {'y': 20}}
        >>> merge_metadata(base, updates)
        {'a': 1, 'b': 3, 'c': 4, 'nested': {'x': 10, 'y': 20}}
    """
    result = base.copy()

    for key, value in updates.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = merge_metadata(result[key], value)
        else:
            result[key] = value

    return result
