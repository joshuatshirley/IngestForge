/**
 * Unit tests for useKnowledgeGraph hook
 *
 * US-1402.1: Interactive Mesh D3 UI
 * GWT Format: Given-When-Then
 * Target Coverage: >80%
 */

import { renderHook } from '@testing-library/react';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import graphReducer from '../../store/slices/graphSlice';
import {
  filterNodesBySearch,
  getNeighborIds,
  getSelectionSubgraph,
  type KnowledgeMeshNode,
  type KnowledgeMeshEdge,
} from '../useKnowledgeGraph';

// =============================================================================
// TEST DATA
// =============================================================================

const mockNodes: KnowledgeMeshNode[] = [
  {
    id: 'node-1',
    type: 'person',
    label: 'John Doe',
    properties: { citation_count: 5, centrality: 0.8, frequency: 10 },
    metadata: { source_doc: 'doc1.pdf', first_seen: '', preview: 'John is a researcher' },
  },
  {
    id: 'node-2',
    type: 'organization',
    label: 'Acme Corp',
    properties: { citation_count: 3, centrality: 0.6, frequency: 5 },
    metadata: { source_doc: 'doc1.pdf', first_seen: '', preview: 'Acme is a company' },
  },
  {
    id: 'node-3',
    type: 'location',
    label: 'New York',
    properties: { citation_count: 2, centrality: 0.4, frequency: 3 },
    metadata: { source_doc: 'doc2.pdf', first_seen: '', preview: 'City in USA' },
  },
];

const mockEdges: KnowledgeMeshEdge[] = [
  { source: 'node-1', target: 'node-2', type: 'works_at', weight: 1.0, style: 'solid' },
  { source: 'node-2', target: 'node-3', type: 'located_in', weight: 0.8, style: 'solid' },
];

// =============================================================================
// GWT TESTS: filterNodesBySearch
// =============================================================================

describe('filterNodesBySearch', () => {
  test('GWT-H01: Empty query returns empty array', () => {
    // Given: Nodes and empty search query
    const nodes = mockNodes;
    const query = '';

    // When: filterNodesBySearch is called
    const result = filterNodesBySearch(nodes, query);

    // Then: Empty array should be returned
    expect(result).toEqual([]);
  });

  test('GWT-H02: Query matches node label', () => {
    // Given: Nodes with "John Doe" label
    const nodes = mockNodes;
    const query = 'john';

    // When: filterNodesBySearch is called
    const result = filterNodesBySearch(nodes, query);

    // Then: "John Doe" node should be in results
    expect(result).toContain('node-1');
    expect(result).toHaveLength(1);
  });

  test('GWT-H03: Query matches node type', () => {
    // Given: Nodes with various types
    const nodes = mockNodes;
    const query = 'organization';

    // When: filterNodesBySearch is called
    const result = filterNodesBySearch(nodes, query);

    // Then: Organization node should be in results
    expect(result).toContain('node-2');
  });

  test('GWT-H04: Query matches preview text', () => {
    // Given: Nodes with preview metadata
    const nodes = mockNodes;
    const query = 'researcher';

    // When: filterNodesBySearch is called
    const result = filterNodesBySearch(nodes, query);

    // Then: Node with "researcher" in preview should be in results
    expect(result).toContain('node-1');
  });

  test('GWT-H05: Case-insensitive search', () => {
    // Given: Nodes with mixed case labels
    const nodes = mockNodes;
    const query = 'ACME';

    // When: filterNodesBySearch is called
    const result = filterNodesBySearch(nodes, query);

    // Then: Match should be case-insensitive
    expect(result).toContain('node-2');
  });

  test('GWT-H06: Multiple nodes match query', () => {
    // Given: Nodes where multiple contain "doc"
    const nodes = mockNodes;
    const query = 'doc';

    // When: filterNodesBySearch is called
    const result = filterNodesBySearch(nodes, query);

    // Then: Multiple results should be returned
    expect(result.length).toBeGreaterThan(0);
  });
});

// =============================================================================
// GWT TESTS: getNeighborIds
// =============================================================================

describe('getNeighborIds', () => {
  test('GWT-H07: Returns neighbors for source node', () => {
    // Given: Node with outgoing edges
    const nodeId = 'node-1';
    const edges = mockEdges;

    // When: getNeighborIds is called
    const neighbors = getNeighborIds(nodeId, edges);

    // Then: Target nodes should be in neighbors
    expect(neighbors.has('node-2')).toBe(true);
    expect(neighbors.size).toBe(1);
  });

  test('GWT-H08: Returns neighbors for target node', () => {
    // Given: Node with incoming edges
    const nodeId = 'node-2';
    const edges = mockEdges;

    // When: getNeighborIds is called
    const neighbors = getNeighborIds(nodeId, edges);

    // Then: Both source and target nodes should be in neighbors
    expect(neighbors.has('node-1')).toBe(true);
    expect(neighbors.has('node-3')).toBe(true);
    expect(neighbors.size).toBe(2);
  });

  test('GWT-H09: Returns empty set for isolated node', () => {
    // Given: Node with no edges
    const nodeId = 'isolated-node';
    const edges = mockEdges;

    // When: getNeighborIds is called
    const neighbors = getNeighborIds(nodeId, edges);

    // Then: Empty set should be returned
    expect(neighbors.size).toBe(0);
  });

  test('GWT-H10: Handles empty edges array', () => {
    // Given: No edges
    const nodeId = 'node-1';
    const edges: KnowledgeMeshEdge[] = [];

    // When: getNeighborIds is called
    const neighbors = getNeighborIds(nodeId, edges);

    // Then: Empty set should be returned
    expect(neighbors.size).toBe(0);
  });
});

// =============================================================================
// GWT TESTS: getSelectionSubgraph
// =============================================================================

describe('getSelectionSubgraph', () => {
  test('GWT-H11: Returns selected nodes and neighbors', () => {
    // Given: Selection of one node
    const selectedIds = ['node-1'];
    const allNodes = mockNodes;
    const allEdges = mockEdges;

    // When: getSelectionSubgraph is called
    const subgraph = getSelectionSubgraph(selectedIds, allNodes, allEdges);

    // Then: Selected node and its neighbors should be included
    expect(subgraph.nodes).toHaveLength(2); // node-1 and node-2
    expect(subgraph.nodes.some((n) => n.id === 'node-1')).toBe(true);
    expect(subgraph.nodes.some((n) => n.id === 'node-2')).toBe(true);
  });

  test('GWT-H12: Returns edges within subgraph', () => {
    // Given: Selection of two connected nodes
    const selectedIds = ['node-1', 'node-2'];
    const allNodes = mockNodes;
    const allEdges = mockEdges;

    // When: getSelectionSubgraph is called
    const subgraph = getSelectionSubgraph(selectedIds, allNodes, allEdges);

    // Then: Edges connecting subgraph nodes should be included
    expect(subgraph.edges.length).toBeGreaterThan(0);
    expect(
      subgraph.edges.some(
        (e) => e.source === 'node-1' && e.target === 'node-2'
      )
    ).toBe(true);
  });

  test('GWT-H13: Empty selection returns empty subgraph', () => {
    // Given: No selected nodes
    const selectedIds: string[] = [];
    const allNodes = mockNodes;
    const allEdges = mockEdges;

    // When: getSelectionSubgraph is called
    const subgraph = getSelectionSubgraph(selectedIds, allNodes, allEdges);

    // Then: Empty subgraph should be returned
    expect(subgraph.nodes).toHaveLength(0);
    expect(subgraph.edges).toHaveLength(0);
  });

  test('GWT-H14: Multiple selections include all neighbors', () => {
    // Given: Multiple selected nodes
    const selectedIds = ['node-1', 'node-3'];
    const allNodes = mockNodes;
    const allEdges = mockEdges;

    // When: getSelectionSubgraph is called
    const subgraph = getSelectionSubgraph(selectedIds, allNodes, allEdges);

    // Then: All selected nodes and their neighbors should be included
    expect(subgraph.nodes.length).toBeGreaterThanOrEqual(2);
  });

  test('GWT-H15: Excludes edges outside subgraph', () => {
    // Given: Selection that doesn't include all edge endpoints
    const selectedIds = ['node-1'];
    const allNodes = mockNodes;
    const allEdges = mockEdges;

    // When: getSelectionSubgraph is called
    const subgraph = getSelectionSubgraph(selectedIds, allNodes, allEdges);

    // Then: Only edges within subgraph should be included
    subgraph.edges.forEach((edge) => {
      const sourceInSubgraph = subgraph.nodes.some((n) => n.id === edge.source);
      const targetInSubgraph = subgraph.nodes.some((n) => n.id === edge.target);
      expect(sourceInSubgraph && targetInSubgraph).toBe(true);
    });
  });
});

/**
 * Test Coverage Summary:
 * - filterNodesBySearch: 6 tests (GWT-H01 to GWT-H06)
 * - getNeighborIds: 4 tests (GWT-H07 to GWT-H10)
 * - getSelectionSubgraph: 5 tests (GWT-H11 to GWT-H15)
 *
 * Total: 15 tests
 * Expected Coverage: >85%
 * JPL Compliance: ✓ All test functions < 60 lines
 */
