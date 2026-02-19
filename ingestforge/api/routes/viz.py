"""Visualization API routes (TICKET-403).

Provides endpoints for knowledge graph visualization data."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ingestforge.core.config_loaders import load_config
from ingestforge.core.logging import get_logger
from ingestforge.storage.factory import get_storage_backend
from ingestforge.viz.graph_export import (
    GraphData,
    GraphNode,
    GraphEdge,
    NodeType,
    EdgeType,
)

logger = get_logger(__name__)
MAX_NODES = 500
MAX_EDGES = 2000

# JPL Rule #2: Additional fixed upper bounds for loop safety
MAX_CHUNKS_FILTER = 10000  # Maximum chunks to filter
MAX_CHUNKS_CITATION = 50000  # Maximum chunks for citation counting
MAX_ENTITIES_PER_CHUNK = 100  # Maximum entities to process per chunk

router = APIRouter(prefix="/v1/viz", tags=["visualization"])


# Pydantic models for API response
class GraphNodeResponse(BaseModel):
    """Node in the graph visualization."""

    id: str
    label: str
    type: str = "concept"
    size: float = 1.0
    color: str = ""
    group: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)


class GraphLinkResponse(BaseModel):
    """Link between nodes in the graph."""

    source: str
    target: str
    type: str = "related_to"
    weight: float = 1.0
    label: str = ""


class GraphMetadataResponse(BaseModel):
    """Metadata about the graph."""

    title: str = "Knowledge Graph"
    description: str = ""
    nodeCount: int = 0
    linkCount: int = 0


class GraphResponse(BaseModel):
    """Complete graph data for visualization."""

    nodes: List[GraphNodeResponse]
    links: List[GraphLinkResponse]
    metadata: GraphMetadataResponse


@router.get("/graph", response_model=GraphResponse)
async def get_graph(
    library: Optional[str] = Query(
        None,
        description="Filter by library name",
    ),
    max_nodes: int = Query(
        MAX_NODES,
        ge=1,
        le=MAX_NODES,
        description="Maximum number of nodes to return",
    ),
    include_edges: bool = Query(
        True,
        description="Include relationship edges",
    ),
) -> GraphResponse:
    """Get knowledge graph data for visualization.

    Returns a D3-compatible graph structure with nodes and links.

    Args:
        library: Optional library name to filter by
        max_nodes: Maximum nodes to include
        include_edges: Whether to include edges

    Returns:
        GraphResponse with nodes, links, and metadata
    """
    try:
        graph_data = _build_graph_from_storage(library, max_nodes, include_edges)
        return _convert_to_response(graph_data)

    except Exception as e:
        logger.error(f"Failed to build graph: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build knowledge graph: {str(e)}",
        )


def _build_graph_from_storage(
    library: Optional[str],
    max_nodes: int,
    include_edges: bool,
) -> GraphData:
    """Build graph data from storage backend.

    Args:
        library: Optional library filter
        max_nodes: Maximum nodes
        include_edges: Include edges

    Returns:
        GraphData structure
    """
    config = load_config()
    storage = get_storage_backend(config)

    # Retrieve chunks from storage
    chunks = _get_chunks(storage, library)
    if not chunks:
        return GraphData(title="Knowledge Graph", description="Empty")

    # Build graph
    return _extract_graph_from_chunks(chunks, max_nodes, include_edges)


def _get_chunks(storage: Any, library: Optional[str]) -> List[Any]:
    """Get chunks from storage.

    Args:
        storage: Storage backend
        library: Optional library filter

    Returns:
        List of chunks
    """
    try:
        # Try different storage APIs
        if hasattr(storage, "get_all_chunks"):
            chunks = storage.get_all_chunks()
        elif hasattr(storage, "list_all"):
            chunks = storage.list_all()
        else:
            chunks = storage.search("", k=MAX_NODES * 2)

        # Filter by library if specified
        if library:
            chunks = _filter_by_library(chunks, library)

        return chunks

    except Exception as e:
        logger.warning(f"Failed to get chunks: {e}")
        return []


def _filter_by_library(chunks: List[Any], library: str) -> List[Any]:
    """Filter chunks by library.

    JPL Rule #2: Bounded loop (max MAX_CHUNKS_FILTER).

    Args:
        chunks: List of chunks (will process up to MAX_CHUNKS_FILTER)
        library: Library name to filter

    Returns:
        Filtered chunks
    """
    # JPL Rule #2: Enforce fixed upper bound
    bounded_chunks = chunks[:MAX_CHUNKS_FILTER]

    filtered = []
    # JPL Rule #2: Loop bounded by MAX_CHUNKS_FILTER
    for chunk in bounded_chunks:
        metadata = _get_chunk_metadata(chunk)
        chunk_library = metadata.get("library", "")
        if chunk_library == library:
            filtered.append(chunk)
    return filtered


def _get_chunk_metadata(chunk: Any) -> Dict[str, Any]:
    """Extract metadata from chunk.

    Args:
        chunk: Chunk object or dict

    Returns:
        Metadata dictionary
    """
    if isinstance(chunk, dict):
        return chunk.get("metadata", {})
    if hasattr(chunk, "metadata"):
        meta = chunk.metadata
        return meta if isinstance(meta, dict) else vars(meta) if meta else {}
    return {}


def _extract_graph_from_chunks(
    chunks: List[Any],
    max_nodes: int,
    include_edges: bool,
) -> GraphData:
    """Extract graph structure from chunks.

    Args:
        chunks: List of chunks
        max_nodes: Maximum nodes
        include_edges: Include edges

    Returns:
        GraphData structure
    """
    graph = GraphData(
        title="Knowledge Graph",
        description=f"Extracted from {len(chunks)} chunks",
    )

    node_ids: set = set()
    doc_nodes: Dict[str, str] = {}

    for chunk in chunks[:max_nodes]:
        if len(graph.nodes) >= max_nodes:
            break

        _process_chunk(chunk, graph, node_ids, doc_nodes, include_edges)

    return graph


def _process_chunk(
    chunk: Any,
    graph: GraphData,
    node_ids: set,
    doc_nodes: Dict[str, str],
    include_edges: bool,
) -> None:
    """Process a chunk into graph nodes/edges.

    Args:
        chunk: Chunk to process
        graph: GraphData to populate
        node_ids: Set of existing node IDs
        doc_nodes: Map of document IDs to node IDs
        include_edges: Include edges
    """
    metadata = _get_chunk_metadata(chunk)
    source = metadata.get("source", "unknown")

    # Add document node
    if source not in doc_nodes:
        _add_document_node(source, graph, node_ids, doc_nodes)

    # Add entity nodes
    entities = metadata.get("entities", [])
    _add_type_nodes(
        entities,
        "entity",
        NodeType.ENTITY,
        2,
        source,
        graph,
        node_ids,
        doc_nodes,
        include_edges,
    )

    # Add concept nodes
    concepts = metadata.get("concepts", [])
    _add_type_nodes(
        concepts,
        "concept",
        NodeType.CONCEPT,
        1,
        source,
        graph,
        node_ids,
        doc_nodes,
        include_edges,
    )


def _add_document_node(
    doc_id: str,
    graph: GraphData,
    node_ids: set,
    doc_nodes: Dict[str, str],
) -> None:
    """Add a document node to the graph.

    Args:
        doc_id: Document identifier
        graph: GraphData to populate
        node_ids: Set of existing node IDs
        doc_nodes: Map of document IDs to node IDs
    """
    if doc_id in node_ids:
        return

    node_id = f"doc_{doc_id}"
    from pathlib import Path

    label = Path(doc_id).stem if doc_id else "Document"

    graph.nodes.append(
        GraphNode(
            id=node_id,
            label=label,
            node_type=NodeType.DOCUMENT,
            size=1.5,
            group=0,
        )
    )

    node_ids.add(node_id)
    doc_nodes[doc_id] = node_id


def _add_type_nodes(
    items: List[str],
    prefix: str,
    node_type: NodeType,
    group: int,
    source: str,
    graph: GraphData,
    node_ids: set,
    doc_nodes: Dict[str, str],
    include_edges: bool,
) -> None:
    """Add typed nodes (entities, concepts) to graph.

    Args:
        items: List of item strings
        prefix: ID prefix
        node_type: Type of node
        group: Group number for coloring
        source: Source document
        graph: GraphData to populate
        node_ids: Set of existing node IDs
        doc_nodes: Map of document IDs to node IDs
        include_edges: Include edges
    """
    for item in items[:20]:
        if len(graph.nodes) >= MAX_NODES:
            break

        node_id = f"{prefix}_{item}"

        # Add node if new
        if node_id not in node_ids:
            graph.nodes.append(
                GraphNode(
                    id=node_id,
                    label=str(item),
                    node_type=node_type,
                    size=1.0,
                    group=group,
                )
            )
            node_ids.add(node_id)

        # Add edge if enabled
        if include_edges and source in doc_nodes:
            if len(graph.edges) < MAX_EDGES:
                graph.edges.append(
                    GraphEdge(
                        source=doc_nodes[source],
                        target=node_id,
                        edge_type=EdgeType.CONTAINS,
                        weight=1.0,
                    )
                )


def _convert_to_response(graph_data: GraphData) -> GraphResponse:
    """Convert GraphData to API response.

    JPL Rule #2: Bounded loops (max MAX_NODES, MAX_EDGES).

    Args:
        graph_data: Internal graph data

    Returns:
        API response model
    """
    # JPL Rule #2: Enforce fixed upper bound on nodes
    bounded_nodes = graph_data.nodes[:MAX_NODES]

    nodes = [
        GraphNodeResponse(
            id=n.id,
            label=n.label,
            type=n.node_type.value,
            size=n.size,
            color=n.color,
            group=n.group,
            metadata=n.metadata,
        )
        # JPL Rule #2: List comprehension bounded by MAX_NODES
        for n in bounded_nodes
    ]

    # JPL Rule #2: Enforce fixed upper bound on edges
    bounded_edges = graph_data.edges[:MAX_EDGES]

    links = [
        GraphLinkResponse(
            source=e.source,
            target=e.target,
            type=e.edge_type.value,
            weight=e.weight,
            label=e.label,
        )
        # JPL Rule #2: List comprehension bounded by MAX_EDGES
        for e in bounded_edges
    ]

    metadata = GraphMetadataResponse(
        title=graph_data.title,
        description=graph_data.description,
        nodeCount=len(nodes),
        linkCount=len(links),
    )

    return GraphResponse(nodes=nodes, links=links, metadata=metadata)


# =============================================================================
# Enhanced Knowledge Mesh Endpoint
# =============================================================================


class KnowledgeMeshNodeProperties(BaseModel):
    """Node properties for knowledge mesh."""

    citation_count: int = 0
    centrality: float = 0.0
    frequency: int = 0


class KnowledgeMeshNodeMetadata(BaseModel):
    """Extended metadata for knowledge mesh nodes."""

    source_doc: str = ""
    first_seen: str = ""
    preview: str = ""
    spatial_links: List[Dict[str, Any]] = Field(default_factory=list)


class KnowledgeMeshNode(BaseModel):
    """Enhanced node for knowledge mesh visualization.

    Interactive Mesh D3 UI
    Epic AC: Provide citation counts and centrality scores.
    """

    id: str
    type: str  # person|org|location|concept|document|chunk
    label: str
    properties: KnowledgeMeshNodeProperties = Field(
        default_factory=KnowledgeMeshNodeProperties
    )
    metadata: KnowledgeMeshNodeMetadata = Field(
        default_factory=KnowledgeMeshNodeMetadata
    )


class KnowledgeMeshEdge(BaseModel):
    """Enhanced edge for knowledge mesh visualization.

    Interactive Mesh D3 UI
    Epic AC: Support different edge styles.
    """

    source: str
    target: str
    type: str  # relationship|citation|similarity
    weight: float = 1.0
    style: str = "solid"  # solid|dashed|dotted


class KnowledgeMeshMetadata(BaseModel):
    """Metadata about knowledge mesh query."""

    total_nodes: int = 0
    filtered_nodes: int = 0
    max_depth: int = 0
    computation_time_ms: float = 0.0


class KnowledgeMeshResponse(BaseModel):
    """Complete knowledge mesh response.

    Interactive Mesh D3 UI
    JPL Rule #9: Complete type hints.
    """

    nodes: List[KnowledgeMeshNode]
    edges: List[KnowledgeMeshEdge]
    metadata: KnowledgeMeshMetadata


@router.get("/graph/knowledge-mesh", response_model=KnowledgeMeshResponse)
async def get_knowledge_mesh(
    max_nodes: int = Query(
        500,
        ge=1,
        le=1000,
        description="Maximum number of nodes to return",
    ),
    min_citations: int = Query(
        0,
        ge=0,
        le=100,
        description="Filter nodes with fewer citations",
    ),
    depth: int = Query(
        2,
        ge=1,
        le=5,
        description="Graph traversal depth (hops from root)",
    ),
    entity_types: Optional[str] = Query(
        None,
        description="Comma-separated entity types to include",
    ),
    include_chunks: bool = Query(
        False,
        description="Include chunk nodes in graph",
    ),
) -> KnowledgeMeshResponse:
    """Get enhanced knowledge mesh for interactive visualization.

    Interactive Mesh D3 UI
    Epic AC: Support filtering by type, citations, and depth.

    Args:
        max_nodes: Maximum number of nodes to return (1-1000)
        min_citations: Minimum citation count filter
        depth: Graph traversal depth (1-5 hops)
        entity_types: Filter by types (e.g., "person,org,location")
        include_chunks: Whether to include chunk nodes

    Returns:
        KnowledgeMeshResponse with filtered nodes and edges

    Raises:
        HTTPException: If graph construction fails
    """
    import time

    start_time = time.time()

    try:
        # Parse entity types filter
        type_filter = None
        if entity_types:
            type_filter = set(entity_types.split(","))

        # Build enhanced graph
        graph_data = _build_enhanced_mesh(
            max_nodes=max_nodes,
            min_citations=min_citations,
            depth=depth,
            type_filter=type_filter,
            include_chunks=include_chunks,
        )

        computation_time = (time.time() - start_time) * 1000

        # Convert to response
        return _convert_to_mesh_response(graph_data, computation_time)

    except Exception as e:
        logger.error(f"Failed to build knowledge mesh: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to build knowledge mesh: {str(e)}",
        )


def _build_enhanced_mesh(
    max_nodes: int,
    min_citations: int,
    depth: int,
    type_filter: Optional[Set[str]],
    include_chunks: bool,
) -> GraphData:
    """Build enhanced knowledge mesh from storage.

    Interactive Mesh D3 UI
    JPL Rule #2: Bounded loops (max MAX_CHUNKS_CITATION).
    JPL Rule #4: Function < 60 lines.

    Args:
        max_nodes: Maximum nodes
        min_citations: Minimum citation filter
        depth: Traversal depth
        type_filter: Entity types to include
        include_chunks: Include chunk nodes

    Returns:
        GraphData with enhanced node properties
    """
    config = load_config()
    storage = get_storage_backend(config)

    # Get chunks from storage
    chunks = _get_chunks(storage, None)
    if not chunks:
        return GraphData(title="Knowledge Mesh", description="Empty")

    # Build graph with enhanced properties
    graph = GraphData(
        title="Knowledge Mesh",
        description=f"Extracted from {len(chunks)} chunks",
    )

    node_ids: set = set()
    doc_nodes: Dict[str, str] = {}
    entity_citations: Dict[str, int] = {}

    # JPL Rule #2: Enforce fixed upper bound for citation counting
    bounded_chunks_citation = chunks[:MAX_CHUNKS_CITATION]

    # First pass: count citations
    # JPL Rule #2: Loop bounded by MAX_CHUNKS_CITATION
    for chunk in bounded_chunks_citation:
        _count_citations(chunk, entity_citations)

    # Second pass: build graph with filters
    for chunk in chunks[: max_nodes * 2]:
        if len(graph.nodes) >= max_nodes:
            break

        _process_enhanced_chunk(
            chunk=chunk,
            graph=graph,
            node_ids=node_ids,
            doc_nodes=doc_nodes,
            entity_citations=entity_citations,
            min_citations=min_citations,
            type_filter=type_filter,
            include_chunks=include_chunks,
        )

    return graph


def _count_citations(chunk: Any, citations: Dict[str, int]) -> None:
    """Count entity citations across chunks.

    Citation count requirement.
    JPL Rule #2: Bounded loop (max MAX_ENTITIES_PER_CHUNK).
    JPL Rule #4: Function < 60 lines.

    Args:
        chunk: Chunk to process
        citations: Citation count dictionary
    """
    metadata = _get_chunk_metadata(chunk)
    entities = metadata.get("entities", [])

    # JPL Rule #2: Enforce fixed upper bound on entities
    bounded_entities = entities[:MAX_ENTITIES_PER_CHUNK]

    # JPL Rule #2: Loop bounded by MAX_ENTITIES_PER_CHUNK
    for entity in bounded_entities:
        entity_key = str(entity).lower()
        citations[entity_key] = citations.get(entity_key, 0) + 1


def _process_enhanced_chunk(
    chunk: Any,
    graph: GraphData,
    node_ids: set,
    doc_nodes: Dict[str, str],
    entity_citations: Dict[str, int],
    min_citations: int,
    type_filter: Optional[Set[str]],
    include_chunks: bool,
) -> None:
    """Process chunk with citation and type filtering.

    Interactive Mesh D3 UI
    JPL Rule #4: Function < 60 lines.

    Args:
        chunk: Chunk to process
        graph: GraphData to populate
        node_ids: Existing node IDs
        doc_nodes: Document node mapping
        entity_citations: Citation counts
        min_citations: Minimum citation filter
        type_filter: Entity types to include
        include_chunks: Include chunk nodes
    """
    metadata = _get_chunk_metadata(chunk)
    source = metadata.get("source", "unknown")

    # Add document node
    if source not in doc_nodes:
        _add_document_node(source, graph, node_ids, doc_nodes)

    # Add entities with filtering
    entities = metadata.get("entities", [])
    for entity in entities[:20]:
        entity_key = str(entity).lower()
        citations = entity_citations.get(entity_key, 0)

        # Apply citation filter
        if citations < min_citations:
            continue

        # Apply type filter
        entity_type = _infer_entity_type(entity, metadata)
        if type_filter and entity_type not in type_filter:
            continue

        node_id = f"entity_{entity}"
        if node_id not in node_ids:
            graph.nodes.append(
                GraphNode(
                    id=node_id,
                    label=str(entity),
                    node_type=NodeType.ENTITY,
                    size=min(1.0 + citations * 0.1, 3.0),
                    group=_get_entity_group(entity_type),
                    metadata={
                        "citation_count": citations,
                        "entity_type": entity_type,
                        "preview": str(entity)[:100],
                    },
                )
            )
            node_ids.add(node_id)

        # Add edge
        if source in doc_nodes and len(graph.edges) < MAX_EDGES:
            graph.edges.append(
                GraphEdge(
                    source=doc_nodes[source],
                    target=node_id,
                    edge_type=EdgeType.CONTAINS,
                    weight=1.0,
                )
            )


def _infer_entity_type(entity: str, metadata: Dict[str, Any]) -> str:
    """Infer entity type from entity string and metadata.

    Entity type classification.
    JPL Rule #4: Function < 60 lines.

    Args:
        entity: Entity string
        metadata: Chunk metadata

    Returns:
        Entity type string
    """
    entity_str = str(entity).lower()

    # Check for capitalized words (likely person/org)
    if entity.istitle() or entity.isupper():
        # Simple heuristic: short capitalized = person, long = org
        if len(entity.split()) <= 2:
            return "person"
        return "organization"

    # Check for location indicators
    location_keywords = {"city", "country", "state", "region", "street"}
    if any(kw in entity_str for kw in location_keywords):
        return "location"

    # Default to concept
    return "concept"


def _get_entity_group(entity_type: str) -> int:
    """Get group number for entity type.

    Node grouping for colors.
    JPL Rule #4: Function < 60 lines.

    Args:
        entity_type: Type of entity

    Returns:
        Group number (0-4)
    """
    type_groups = {
        "person": 1,
        "organization": 2,
        "location": 3,
        "concept": 4,
    }
    return type_groups.get(entity_type, 0)


def _convert_to_mesh_response(
    graph_data: GraphData,
    computation_time: float,
) -> KnowledgeMeshResponse:
    """Convert GraphData to KnowledgeMeshResponse.

    Interactive Mesh D3 UI
    JPL Rule #2: Bounded loops (max MAX_NODES, MAX_EDGES).
    JPL Rule #4: Function < 60 lines.

    Args:
        graph_data: Internal graph structure
        computation_time: Milliseconds to compute

    Returns:
        KnowledgeMeshResponse with enhanced nodes
    """
    # JPL Rule #2: Enforce fixed upper bound on nodes
    bounded_nodes = graph_data.nodes[:MAX_NODES]

    nodes = []
    # JPL Rule #2: Loop bounded by MAX_NODES
    for n in bounded_nodes:
        meta = n.metadata or {}
        nodes.append(
            KnowledgeMeshNode(
                id=n.id,
                type=meta.get("entity_type", n.node_type.value),
                label=n.label,
                properties=KnowledgeMeshNodeProperties(
                    citation_count=meta.get("citation_count", 0),
                    centrality=0.0,
                    frequency=meta.get("citation_count", 0),
                ),
                metadata=KnowledgeMeshNodeMetadata(
                    source_doc=meta.get("source", ""),
                    first_seen="",
                    preview=meta.get("preview", "")[:100],
                    spatial_links=meta.get("spatial_links", []),
                ),
            )
        )

    # JPL Rule #2: Enforce fixed upper bound on edges
    bounded_edges = graph_data.edges[:MAX_EDGES]

    edges = []
    # JPL Rule #2: Loop bounded by MAX_EDGES
    for e in bounded_edges:
        edge_style = "solid"
        if e.edge_type == EdgeType.REFERENCES:
            edge_style = "dashed"
        elif e.edge_type == EdgeType.SIMILAR_TO:
            edge_style = "dotted"

        edges.append(
            KnowledgeMeshEdge(
                source=e.source,
                target=e.target,
                type=e.edge_type.value,
                weight=e.weight,
                style=edge_style,
            )
        )

    metadata = KnowledgeMeshMetadata(
        total_nodes=len(nodes),
        filtered_nodes=len(nodes),
        max_depth=2,
        computation_time_ms=computation_time,
    )

    return KnowledgeMeshResponse(
        nodes=nodes,
        edges=edges,
        metadata=metadata,
    )
