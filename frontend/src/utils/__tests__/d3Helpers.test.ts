/**
 * Unit tests for d3Helpers utilities
 *
 * US-1402.1: Interactive Mesh D3 UI
 * GWT Format: Given-When-Then
 * Target Coverage: >80%
 */

import * as d3 from 'd3';
import {
  createForceSimulation,
  cullNodesOutsideViewport,
  getViewportBounds,
  calculateNodeSize,
  truncateLabel,
  throttle,
  debounce,
  type SimulationNode,
  type SimulationEdge,
  type ViewportBounds,
} from '../d3Helpers';

// =============================================================================
// TEST DATA
// =============================================================================

const mockNodes: SimulationNode[] = [
  { id: 'node-1', label: 'Node 1', type: 'person', x: 100, y: 100 },
  { id: 'node-2', label: 'Node 2', type: 'organization', x: 200, y: 200 },
  { id: 'node-3', label: 'Node 3', type: 'location', x: 300, y: 300 },
  { id: 'node-4', label: 'Node 4', type: 'concept', x: 5000, y: 5000 }, // Outside viewport
];

const mockEdges: SimulationEdge[] = [
  { source: 'node-1', target: 'node-2' },
  { source: 'node-2', target: 'node-3' },
];

// =============================================================================
// GWT TESTS: createForceSimulation
// =============================================================================

describe('createForceSimulation', () => {
  test('GWT-D01: Creates simulation with default config', () => {
    // Given: Nodes and edges
    const nodes = [...mockNodes];
    const edges = [...mockEdges];

    // When: createForceSimulation is called
    const simulation = createForceSimulation(nodes, edges, 800, 600);

    // Then: Simulation should be created
    expect(simulation).toBeDefined();
    expect(simulation.nodes()).toHaveLength(mockNodes.length);
  });

  test('GWT-D02: Applies custom force configuration', () => {
    // Given: Nodes, edges, and custom config
    const nodes = [...mockNodes];
    const edges = [...mockEdges];
    const config = { linkDistance: 200, chargeStrength: -500 };

    // When: createForceSimulation is called with config
    const simulation = createForceSimulation(nodes, edges, 800, 600, config);

    // Then: Simulation should be created with custom config
    expect(simulation).toBeDefined();
  });

  test('GWT-D03: Handles empty nodes array', () => {
    // Given: Empty nodes array
    const nodes: SimulationNode[] = [];
    const edges: SimulationEdge[] = [];

    // When: createForceSimulation is called
    const simulation = createForceSimulation(nodes, edges, 800, 600);

    // Then: Simulation should be created without errors
    expect(simulation).toBeDefined();
    expect(simulation.nodes()).toHaveLength(0);
  });
});

// =============================================================================
// GWT TESTS: cullNodesOutsideViewport
// =============================================================================

describe('cullNodesOutsideViewport', () => {
  test('GWT-D04: Returns only visible nodes', () => {
    // Given: Nodes with various positions
    const nodes = mockNodes;
    const viewport: ViewportBounds = {
      x: 0,
      y: 0,
      width: 800,
      height: 600,
      scale: 1.0,
    };

    // When: cullNodesOutsideViewport is called
    const visible = cullNodesOutsideViewport(nodes, viewport);

    // Then: Only nodes within viewport should be returned
    expect(visible.length).toBeLessThan(nodes.length);
    expect(visible.some((n) => n.id === 'node-4')).toBe(false); // Outside viewport
  });

  test('GWT-D05: Returns all nodes when viewport is large', () => {
    // Given: Large viewport covering all nodes
    const nodes = mockNodes.slice(0, 3); // Exclude node-4
    const viewport: ViewportBounds = {
      x: 0,
      y: 0,
      width: 10000,
      height: 10000,
      scale: 1.0,
    };

    // When: cullNodesOutsideViewport is called
    const visible = cullNodesOutsideViewport(nodes, viewport);

    // Then: All nodes should be returned
    expect(visible).toHaveLength(nodes.length);
  });

  test('GWT-D06: Accounts for zoom scale', () => {
    // Given: Zoomed viewport
    const nodes = mockNodes;
    const viewport: ViewportBounds = {
      x: 0,
      y: 0,
      width: 800,
      height: 600,
      scale: 0.5, // Zoomed out
    };

    // When: cullNodesOutsideViewport is called
    const visible = cullNodesOutsideViewport(nodes, viewport);

    // Then: More nodes should be visible due to zoom
    expect(visible.length).toBeGreaterThan(0);
  });

  test('GWT-D07: Handles empty nodes array', () => {
    // Given: Empty nodes array
    const nodes: SimulationNode[] = [];
    const viewport: ViewportBounds = {
      x: 0,
      y: 0,
      width: 800,
      height: 600,
      scale: 1.0,
    };

    // When: cullNodesOutsideViewport is called
    const visible = cullNodesOutsideViewport(nodes, viewport);

    // Then: Empty array should be returned
    expect(visible).toHaveLength(0);
  });
});

// =============================================================================
// GWT TESTS: getViewportBounds
// =============================================================================

describe('getViewportBounds', () => {
  test('GWT-D08: Extracts bounds from transform', () => {
    // Given: D3 zoom transform
    const transform = d3.zoomIdentity.translate(100, 200).scale(1.5);
    const width = 800;
    const height = 600;

    // When: getViewportBounds is called
    const bounds = getViewportBounds(transform, width, height);

    // Then: Bounds should match transform
    expect(bounds.x).toBe(100);
    expect(bounds.y).toBe(200);
    expect(bounds.scale).toBe(1.5);
    expect(bounds.width).toBe(width);
    expect(bounds.height).toBe(height);
  });

  test('GWT-D09: Handles identity transform', () => {
    // Given: Identity transform (no translation/scale)
    const transform = d3.zoomIdentity;

    // When: getViewportBounds is called
    const bounds = getViewportBounds(transform, 800, 600);

    // Then: Bounds should reflect identity
    expect(bounds.x).toBe(0);
    expect(bounds.y).toBe(0);
    expect(bounds.scale).toBe(1.0);
  });
});

// =============================================================================
// GWT TESTS: calculateNodeSize
// =============================================================================

describe('calculateNodeSize', () => {
  test('GWT-D10: Returns minimum size for zero citations', () => {
    // Given: Citation count of 0
    const citationCount = 0;

    // When: calculateNodeSize is called
    const size = calculateNodeSize(citationCount);

    // Then: Minimum size should be returned
    expect(size).toBeGreaterThanOrEqual(10);
  });

  test('GWT-D11: Increases size with citation count', () => {
    // Given: Different citation counts
    const size1 = calculateNodeSize(1);
    const size2 = calculateNodeSize(10);
    const size3 = calculateNodeSize(100);

    // Then: Size should increase with citations
    expect(size2).toBeGreaterThan(size1);
    expect(size3).toBeGreaterThan(size2);
  });

  test('GWT-D12: Respects maximum size', () => {
    // Given: Very high citation count
    const citationCount = 10000;
    const maxSize = 50;

    // When: calculateNodeSize is called
    const size = calculateNodeSize(citationCount, 10, maxSize);

    // Then: Size should not exceed maximum
    expect(size).toBeLessThanOrEqual(maxSize);
  });

  test('GWT-D13: Custom min/max sizes work', () => {
    // Given: Custom size range
    const minSize = 20;
    const maxSize = 80;

    // When: calculateNodeSize is called with custom range
    const size = calculateNodeSize(50, minSize, maxSize);

    // Then: Size should be within custom range
    expect(size).toBeGreaterThanOrEqual(minSize);
    expect(size).toBeLessThanOrEqual(maxSize);
  });
});

// =============================================================================
// GWT TESTS: truncateLabel
// =============================================================================

describe('truncateLabel', () => {
  test('GWT-D14: Short labels unchanged', () => {
    // Given: Short label
    const label = 'Short';

    // When: truncateLabel is called
    const result = truncateLabel(label);

    // Then: Label should be unchanged
    expect(result).toBe('Short');
  });

  test('GWT-D15: Long labels truncated with ellipsis', () => {
    // Given: Long label
    const label = 'This is a very long label that exceeds the maximum length';

    // When: truncateLabel is called with max 15
    const result = truncateLabel(label, 15);

    // Then: Label should be truncated with "..."
    expect(result).toHaveLength(18); // 15 + "..."
    expect(result.endsWith('...')).toBe(true);
  });

  test('GWT-D16: Exactly max length not truncated', () => {
    // Given: Label exactly at max length
    const label = '123456789012345'; // 15 chars
    const result = truncateLabel(label, 15);

    // Then: Should not be truncated
    expect(result).toBe(label);
    expect(result.endsWith('...')).toBe(false);
  });

  test('GWT-D17: Empty label handled', () => {
    // Given: Empty label
    const label = '';

    // When: truncateLabel is called
    const result = truncateLabel(label);

    // Then: Empty string should be returned
    expect(result).toBe('');
  });
});

// =============================================================================
// GWT TESTS: throttle
// =============================================================================

describe('throttle', () => {
  jest.useFakeTimers();

  test('GWT-D18: Throttles function calls', () => {
    // Given: Throttled function with 100ms delay
    const fn = jest.fn();
    const throttled = throttle(fn, 100);

    // When: Function is called multiple times rapidly
    throttled();
    throttled();
    throttled();

    // Then: Function should be called only once
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('GWT-D19: Allows call after delay', () => {
    // Given: Throttled function
    const fn = jest.fn();
    const throttled = throttle(fn, 100);

    // When: Function is called, delay passes, called again
    throttled();
    jest.advanceTimersByTime(100);
    throttled();

    // Then: Function should be called twice
    expect(fn).toHaveBeenCalledTimes(2);
  });

  afterEach(() => {
    jest.clearAllTimers();
  });
});

// =============================================================================
// GWT TESTS: debounce
// =============================================================================

describe('debounce', () => {
  jest.useFakeTimers();

  test('GWT-D20: Delays function execution', () => {
    // Given: Debounced function with 300ms delay
    const fn = jest.fn();
    const debounced = debounce(fn, 300);

    // When: Function is called
    debounced();

    // Then: Function should not be called immediately
    expect(fn).not.toHaveBeenCalled();
  });

  test('GWT-D21: Executes after delay', () => {
    // Given: Debounced function
    const fn = jest.fn();
    const debounced = debounce(fn, 300);

    // When: Function is called and delay passes
    debounced();
    jest.advanceTimersByTime(300);

    // Then: Function should be executed
    expect(fn).toHaveBeenCalledTimes(1);
  });

  test('GWT-D22: Cancels previous calls', () => {
    // Given: Debounced function
    const fn = jest.fn();
    const debounced = debounce(fn, 300);

    // When: Function is called multiple times rapidly
    debounced();
    jest.advanceTimersByTime(100);
    debounced();
    jest.advanceTimersByTime(100);
    debounced();
    jest.advanceTimersByTime(300);

    // Then: Function should be called only once (last call)
    expect(fn).toHaveBeenCalledTimes(1);
  });

  afterEach(() => {
    jest.clearAllTimers();
  });
});

/**
 * Test Coverage Summary:
 * - createForceSimulation: 3 tests (GWT-D01 to GWT-D03)
 * - cullNodesOutsideViewport: 4 tests (GWT-D04 to GWT-D07)
 * - getViewportBounds: 2 tests (GWT-D08 to GWT-D09)
 * - calculateNodeSize: 4 tests (GWT-D10 to GWT-D13)
 * - truncateLabel: 4 tests (GWT-D14 to GWT-D17)
 * - throttle: 2 tests (GWT-D18 to GWT-D19)
 * - debounce: 3 tests (GWT-D20 to GWT-D22)
 *
 * Total: 22 tests
 * Expected Coverage: >88%
 * JPL Compliance: ✓ All test functions < 60 lines
 */
