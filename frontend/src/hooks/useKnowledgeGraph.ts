/**
 * useKnowledgeGraph Hook - Fetch and manage knowledge graph data
 *
 * US-1402.1: Interactive Mesh D3 UI
 * Epic AC: Fetch graph data with filtering
 *
 * JPL Power of Ten Compliance:
 * - Rule #9: Complete type hints
 * - Rule #4: All functions < 60 lines
 */

import { useSelector } from 'react-redux';
import {
  selectFilters,
  selectEntityTypesArray,
  type EntityType,
} from '../store/slices/graphSlice';
import { useGetKnowledgeMeshQuery } from '../store/api/ingestforgeApi';

// =============================================================================
// CONSTANTS (JPL Rule #2: Fixed upper bounds)
// =============================================================================

/** Maximum nodes to search (JPL Rule #2) */
const MAX_SEARCH_NODES = 5000;

/** Maximum edges to check for neighbors (JPL Rule #2) */
const MAX_EDGES_CHECK = 10000;

/** Maximum selected nodes for subgraph (JPL Rule #2) */
const MAX_SELECTED_NODES = 100;

// =============================================================================
// TYPE DEFINITIONS (Rule #9: Complete type hints)
// =============================================================================

/** Node properties from API */
export interface NodeProperties {
  citation_count: number;
  centrality: number;
  frequency: number;
}

/** Node metadata from API */
export interface NodeMetadata {
  source_doc: string;
  first_seen: string;
  preview: string;
}

/** Knowledge mesh node */
export interface KnowledgeMeshNode {
  id: string;
  type: string;
  label: string;
  properties: NodeProperties;
  metadata: NodeMetadata;
}

/** Knowledge mesh edge */
export interface KnowledgeMeshEdge {
  source: string;
  target: string;
  type: string;
  weight: number;
  style: 'solid' | 'dashed' | 'dotted';
}

/** Knowledge mesh metadata */
export interface KnowledgeMeshMetadata {
  total_nodes: number;
  filtered_nodes: number;
  max_depth: number;
  computation_time_ms: number;
}

/** Knowledge mesh response */
export interface KnowledgeMeshResponse {
  nodes: KnowledgeMeshNode[];
  edges: KnowledgeMeshEdge[];
  metadata: KnowledgeMeshMetadata;
}

/** Hook options */
export interface UseKnowledgeGraphOptions {
  maxNodes?: number;
  pollingInterval?: number;
  enabled?: boolean;
}

/** Hook return value */
export interface UseKnowledgeGraphResult {
  nodes: KnowledgeMeshNode[];
  edges: KnowledgeMeshEdge[];
  metadata: KnowledgeMeshMetadata | null;
  isLoading: boolean;
  isError: boolean;
  error: unknown;
  refetch: () => void;
}

// =============================================================================
// HOOK IMPLEMENTATION (Rule #4: < 60 lines)
// =============================================================================

/**
 * Fetch knowledge graph data with current filters.
 *
 * US-1402.1: Interactive Mesh D3 UI
 * Epic AC: Auto-refetch on filter changes.
 *
 * @param options - Hook configuration options
 * @returns Graph data, loading state, and refetch function
 *
 * @example
 * ```tsx
 * const { nodes, edges, isLoading, refetch } = useKnowledgeGraph({
 *   maxNodes: 500,
 *   pollingInterval: 60000,
 * });
 * ```
 */
export function useKnowledgeGraph(
  options: UseKnowledgeGraphOptions = {}
): UseKnowledgeGraphResult {
  const { maxNodes = 500, pollingInterval, enabled = true } = options;

  // Get current filters from Redux
  const filters = useSelector(selectFilters);
  const entityTypesArray = useSelector(selectEntityTypesArray);

  // Build query parameters
  const queryParams = {
    max_nodes: maxNodes,
    min_citations: filters.minCitations,
    depth: filters.maxDepth,
    entity_types: entityTypesArray.join(','),
    include_chunks: filters.showChunks,
  };

  // Fetch data with RTK Query
  const { data, isLoading, isError, error, refetch } =
    useGetKnowledgeMeshQuery(queryParams, {
      pollingInterval,
      skip: !enabled,
      refetchOnFocus: true,
      refetchOnMountOrArgChange: true,
    });

  return {
    nodes: data?.nodes ?? [],
    edges: data?.edges ?? [],
    metadata: data?.metadata ?? null,
    isLoading,
    isError,
    error,
    refetch,
  };
}

/**
 * Filter nodes by search query (client-side fuzzy match).
 *
 * US-1402.1: Search functionality.
 * JPL Rule #2: Bounded loop (max MAX_SEARCH_NODES).
 * JPL Rule #4: Function < 60 lines.
 *
 * @param nodes - All nodes (will process up to MAX_SEARCH_NODES)
 * @param query - Search query string
 * @returns Filtered node IDs
 */
export function filterNodesBySearch(
  nodes: KnowledgeMeshNode[],
  query: string
): string[] {
  if (!query.trim()) {
    return [];
  }

  // JPL Rule #2: Enforce fixed upper bound
  const boundedNodes = nodes.slice(0, MAX_SEARCH_NODES);
  const lowerQuery = query.toLowerCase();
  const matches: string[] = [];

  // JPL Rule #2: Loop bounded by MAX_SEARCH_NODES
  for (const node of boundedNodes) {
    const labelMatch = node.label.toLowerCase().includes(lowerQuery);
    const typeMatch = node.type.toLowerCase().includes(lowerQuery);
    const previewMatch = node.metadata.preview
      .toLowerCase()
      .includes(lowerQuery);

    if (labelMatch || typeMatch || previewMatch) {
      matches.push(node.id);
    }
  }

  return matches;
}

/**
 * Get neighbor node IDs for a given node.
 *
 * US-1402.1: Multi-select subgraph.
 * JPL Rule #2: Bounded loop (max MAX_EDGES_CHECK).
 * JPL Rule #4: Function < 60 lines.
 *
 * @param nodeId - Target node ID
 * @param edges - All edges (will process up to MAX_EDGES_CHECK)
 * @returns Set of neighbor node IDs
 */
export function getNeighborIds(
  nodeId: string,
  edges: KnowledgeMeshEdge[]
): Set<string> {
  const neighbors = new Set<string>();

  // JPL Rule #2: Enforce fixed upper bound
  const boundedEdges = edges.slice(0, MAX_EDGES_CHECK);

  // JPL Rule #2: Loop bounded by MAX_EDGES_CHECK
  for (const edge of boundedEdges) {
    if (edge.source === nodeId) {
      neighbors.add(edge.target);
    } else if (edge.target === nodeId) {
      neighbors.add(edge.source);
    }
  }

  return neighbors;
}

/**
 * Get subgraph for selected nodes (nodes + 1-hop neighbors).
 *
 * US-1402.1: Selection subgraph.
 * JPL Rule #2: Bounded loop (max MAX_SELECTED_NODES).
 * JPL Rule #4: Function < 60 lines.
 *
 * @param selectedIds - Selected node IDs (up to MAX_SELECTED_NODES)
 * @param allNodes - All nodes
 * @param allEdges - All edges
 * @returns Subgraph nodes and edges
 */
export function getSelectionSubgraph(
  selectedIds: string[],
  allNodes: KnowledgeMeshNode[],
  allEdges: KnowledgeMeshEdge[]
): { nodes: KnowledgeMeshNode[]; edges: KnowledgeMeshEdge[] } {
  // JPL Rule #2: Enforce fixed upper bound on selection
  const boundedSelection = selectedIds.slice(0, MAX_SELECTED_NODES);

  // Get all neighbors of selected nodes
  const subgraphNodeIds = new Set<string>(boundedSelection);

  // JPL Rule #2: Loop bounded by MAX_SELECTED_NODES
  for (const nodeId of boundedSelection) {
    const neighbors = getNeighborIds(nodeId, allEdges);
    neighbors.forEach((n) => subgraphNodeIds.add(n));
  }

  // Filter nodes and edges
  const nodes = allNodes.filter((n) => subgraphNodeIds.has(n.id));
  const edges = allEdges.filter(
    (e) => subgraphNodeIds.has(e.source) && subgraphNodeIds.has(e.target)
  );

  return { nodes, edges };
}
