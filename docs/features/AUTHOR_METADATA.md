# Author & Identity Metadata (TICKET-301)

**Status:** ✅ Implemented
**Version:** IngestForge v1.2+

## Overview

IngestForge now tracks author/contributor identity for every chunk in the knowledge base, enabling multi-user collaboration scenarios where knowing who contributed each piece of content is important.

## Quick Start

### Set author on a chunk:
```python
from ingestforge.core.provenance import set_author
from ingestforge.chunking.semantic_chunker import ChunkRecord

chunk = ChunkRecord(chunk_id="c1", document_id="d1", content="text")
set_author(chunk, "alice@example.com", "Alice Johnson")
```

### Get author info from a chunk:
```python
from ingestforge.core.provenance import get_author_info

author_info = get_author_info(chunk)
print(author_info.format_attribution)  # "Contributed by: Alice Johnson"
```

## Features

1. **Author Fields**: ChunkRecord includes `author_id` and `author_name`
2. **ContributorIdentity**: Structured class for author metadata
3. **Helper Functions**: `set_author` and `get_author_info`
4. **Query Display**: Automatic "Contributed by" in search results
5. **Export Preservation**: Author fields preserved in JSON/JSONL exports

## Testing

All tests passing:
- `tests/unit/provenance/test_contributor_identity.py`
- `tests/unit/provenance/test_chunk_author.py`
- `tests/unit/provenance/test_author_helpers.py`
- `tests/integration/test_author_metadata.py`

## Backward Compatibility

Fully backward compatible. Author fields are optional and default to None.
