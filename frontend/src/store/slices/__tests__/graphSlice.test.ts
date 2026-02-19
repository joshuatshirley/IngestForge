/**
 * Unit tests for graphSlice Redux slice
 *
 * US-1402.1: Interactive Mesh D3 UI
 * GWT Format: Given-When-Then
 * Target Coverage: >80%
 * JPL Compliance: All test functions < 60 lines
 */

import graphReducer, {
  setMinCitations,
  setMaxDepth,
  setShowChunks,
  setSearchQuery,
  toggleEntityType,
  setEntityTypes,
  resetFilters,
  selectNode,
  deselectNode,
  clearSelection,
  setHighlightedNodes,
  clearHighlights,
  setViewport,
  resetViewport,
  selectFilters,
  selectSelection,
  selectViewport,
  selectSelectedNodeIds,
  selectHighlightedNodeIds,
  selectEntityTypesArray,
  selectSelectionCount,
  type GraphState,
  type EntityType,
} from '../graphSlice';

// =============================================================================
// TEST DATA
// =============================================================================

const initialState: GraphState = {
  filters: {
    entityTypes: new Set<EntityType>([
      'person',
      'organization',
      'location',
      'concept',
      'document',
    ]),
    minCitations: 0,
    maxDepth: 2,
    showChunks: false,
    searchQuery: '',
  },
  selection: {
    selectedNodeIds: new Set<string>(),
    highlightedNodeIds: new Set<string>(),
  },
  viewport: {
    scale: 1.0,
    x: 0,
    y: 0,
  },
};

// =============================================================================
// GWT TESTS: Filter Actions
// =============================================================================

describe('graphSlice - Filter Actions', () => {
  test('GWT-F01: setMinCitations updates min citations filter', () => {
    // Given: Initial state with minCitations = 0
    const state = initialState;

    // When: setMinCitations action is dispatched with value 5
    const newState = graphReducer(state, setMinCitations(5));

    // Then: minCitations should be updated to 5
    expect(newState.filters.minCitations).toBe(5);
  });

  test('GWT-F02: setMaxDepth updates depth filter', () => {
    // Given: Initial state with maxDepth = 2
    const state = initialState;

    // When: setMaxDepth action is dispatched with value 4
    const newState = graphReducer(state, setMaxDepth(4));

    // Then: maxDepth should be updated to 4
    expect(newState.filters.maxDepth).toBe(4);
  });

  test('GWT-F03: setShowChunks updates chunks visibility', () => {
    // Given: Initial state with showChunks = false
    const state = initialState;

    // When: setShowChunks action is dispatched with value true
    const newState = graphReducer(state, setShowChunks(true));

    // Then: showChunks should be updated to true
    expect(newState.filters.showChunks).toBe(true);
  });

  test('GWT-F04: setSearchQuery updates search query', () => {
    // Given: Initial state with empty search query
    const state = initialState;

    // When: setSearchQuery action is dispatched with "test"
    const newState = graphReducer(state, setSearchQuery('test'));

    // Then: searchQuery should be updated to "test"
    expect(newState.filters.searchQuery).toBe('test');
  });

  test('GWT-F05: toggleEntityType adds type when not present', () => {
    // Given: Initial state without "event" type
    const state = {
      ...initialState,
      filters: {
        ...initialState.filters,
        entityTypes: new Set<EntityType>(['person']),
      },
    };

    // When: toggleEntityType is dispatched with "organization"
    const newState = graphReducer(state, toggleEntityType('organization'));

    // Then: "organization" should be added to entityTypes
    expect(newState.filters.entityTypes.has('organization')).toBe(true);
    expect(newState.filters.entityTypes.size).toBe(2);
  });

  test('GWT-F06: toggleEntityType removes type when present', () => {
    // Given: Initial state with "person" type
    const state = {
      ...initialState,
      filters: {
        ...initialState.filters,
        entityTypes: new Set<EntityType>(['person', 'organization']),
      },
    };

    // When: toggleEntityType is dispatched with "person"
    const newState = graphReducer(state, toggleEntityType('person'));

    // Then: "person" should be removed from entityTypes
    expect(newState.filters.entityTypes.has('person')).toBe(false);
    expect(newState.filters.entityTypes.size).toBe(1);
  });

  test('GWT-F07: setEntityTypes replaces all types', () => {
    // Given: Initial state with default entity types
    const state = initialState;

    // When: setEntityTypes is dispatched with new types
    const newTypes: EntityType[] = ['person', 'location'];
    const newState = graphReducer(state, setEntityTypes(newTypes));

    // Then: entityTypes should contain only the new types
    expect(newState.filters.entityTypes.size).toBe(2);
    expect(newState.filters.entityTypes.has('person')).toBe(true);
    expect(newState.filters.entityTypes.has('location')).toBe(true);
    expect(newState.filters.entityTypes.has('organization')).toBe(false);
  });

  test('GWT-F08: resetFilters restores default filters', () => {
    // Given: State with modified filters
    const state = {
      ...initialState,
      filters: {
        entityTypes: new Set<EntityType>(['person']),
        minCitations: 10,
        maxDepth: 5,
        showChunks: true,
        searchQuery: 'test query',
      },
    };

    // When: resetFilters action is dispatched
    const newState = graphReducer(state, resetFilters());

    // Then: All filters should be reset to defaults
    expect(newState.filters.minCitations).toBe(0);
    expect(newState.filters.maxDepth).toBe(2);
    expect(newState.filters.showChunks).toBe(false);
    expect(newState.filters.searchQuery).toBe('');
    expect(newState.filters.entityTypes.size).toBe(5);
  });
});

// =============================================================================
// GWT TESTS: Selection Actions
// =============================================================================

describe('graphSlice - Selection Actions', () => {
  test('GWT-S01: selectNode adds node to selection (single select)', () => {
    // Given: Initial state with no selection
    const state = initialState;

    // When: selectNode action is dispatched with multiSelect=false
    const newState = graphReducer(
      state,
      selectNode({ nodeId: 'node-1', multiSelect: false })
    );

    // Then: Only the specified node should be selected
    expect(newState.selection.selectedNodeIds.size).toBe(1);
    expect(newState.selection.selectedNodeIds.has('node-1')).toBe(true);
  });

  test('GWT-S02: selectNode adds to selection (multi-select)', () => {
    // Given: State with one node already selected
    const state = {
      ...initialState,
      selection: {
        ...initialState.selection,
        selectedNodeIds: new Set(['node-1']),
      },
    };

    // When: selectNode action is dispatched with multiSelect=true
    const newState = graphReducer(
      state,
      selectNode({ nodeId: 'node-2', multiSelect: true })
    );

    // Then: Both nodes should be selected
    expect(newState.selection.selectedNodeIds.size).toBe(2);
    expect(newState.selection.selectedNodeIds.has('node-1')).toBe(true);
    expect(newState.selection.selectedNodeIds.has('node-2')).toBe(true);
  });

  test('GWT-S03: selectNode replaces selection (single select)', () => {
    // Given: State with one node already selected
    const state = {
      ...initialState,
      selection: {
        ...initialState.selection,
        selectedNodeIds: new Set(['node-1']),
      },
    };

    // When: selectNode action is dispatched with multiSelect=false
    const newState = graphReducer(
      state,
      selectNode({ nodeId: 'node-2', multiSelect: false })
    );

    // Then: Only the new node should be selected
    expect(newState.selection.selectedNodeIds.size).toBe(1);
    expect(newState.selection.selectedNodeIds.has('node-1')).toBe(false);
    expect(newState.selection.selectedNodeIds.has('node-2')).toBe(true);
  });

  test('GWT-S04: deselectNode removes node from selection', () => {
    // Given: State with multiple nodes selected
    const state = {
      ...initialState,
      selection: {
        ...initialState.selection,
        selectedNodeIds: new Set(['node-1', 'node-2', 'node-3']),
      },
    };

    // When: deselectNode action is dispatched
    const newState = graphReducer(state, deselectNode('node-2'));

    // Then: The specified node should be removed
    expect(newState.selection.selectedNodeIds.size).toBe(2);
    expect(newState.selection.selectedNodeIds.has('node-2')).toBe(false);
    expect(newState.selection.selectedNodeIds.has('node-1')).toBe(true);
    expect(newState.selection.selectedNodeIds.has('node-3')).toBe(true);
  });

  test('GWT-S05: clearSelection removes all selections', () => {
    // Given: State with nodes selected and highlighted
    const state = {
      ...initialState,
      selection: {
        selectedNodeIds: new Set(['node-1', 'node-2']),
        highlightedNodeIds: new Set(['node-3', 'node-4']),
      },
    };

    // When: clearSelection action is dispatched
    const newState = graphReducer(state, clearSelection());

    // Then: All selections and highlights should be cleared
    expect(newState.selection.selectedNodeIds.size).toBe(0);
    expect(newState.selection.highlightedNodeIds.size).toBe(0);
  });

  test('GWT-S06: setHighlightedNodes sets highlighted nodes', () => {
    // Given: Initial state with no highlights
    const state = initialState;

    // When: setHighlightedNodes action is dispatched
    const nodeIds = ['node-1', 'node-2', 'node-3'];
    const newState = graphReducer(state, setHighlightedNodes(nodeIds));

    // Then: The specified nodes should be highlighted
    expect(newState.selection.highlightedNodeIds.size).toBe(3);
    expect(newState.selection.highlightedNodeIds.has('node-1')).toBe(true);
    expect(newState.selection.highlightedNodeIds.has('node-2')).toBe(true);
    expect(newState.selection.highlightedNodeIds.has('node-3')).toBe(true);
  });

  test('GWT-S07: clearHighlights removes all highlights', () => {
    // Given: State with highlighted nodes
    const state = {
      ...initialState,
      selection: {
        ...initialState.selection,
        highlightedNodeIds: new Set(['node-1', 'node-2']),
      },
    };

    // When: clearHighlights action is dispatched
    const newState = graphReducer(state, clearHighlights());

    // Then: All highlights should be cleared
    expect(newState.selection.highlightedNodeIds.size).toBe(0);
  });
});

// =============================================================================
// GWT TESTS: Viewport Actions
// =============================================================================

describe('graphSlice - Viewport Actions', () => {
  test('GWT-V01: setViewport updates viewport transform', () => {
    // Given: Initial viewport state
    const state = initialState;

    // When: setViewport action is dispatched
    const newViewport = { scale: 2.0, x: 100, y: 200 };
    const newState = graphReducer(state, setViewport(newViewport));

    // Then: Viewport should be updated
    expect(newState.viewport.scale).toBe(2.0);
    expect(newState.viewport.x).toBe(100);
    expect(newState.viewport.y).toBe(200);
  });

  test('GWT-V02: resetViewport restores default viewport', () => {
    // Given: State with modified viewport
    const state = {
      ...initialState,
      viewport: { scale: 3.0, x: 500, y: 600 },
    };

    // When: resetViewport action is dispatched
    const newState = graphReducer(state, resetViewport());

    // Then: Viewport should be reset to defaults
    expect(newState.viewport.scale).toBe(1.0);
    expect(newState.viewport.x).toBe(0);
    expect(newState.viewport.y).toBe(0);
  });
});

// =============================================================================
// GWT TESTS: Selectors
// =============================================================================

describe('graphSlice - Selectors', () => {
  const mockRootState = {
    graph: initialState,
  } as any;

  test('GWT-SEL01: selectFilters returns filters state', () => {
    // Given: Mock root state
    // When: selectFilters selector is called
    const filters = selectFilters(mockRootState);

    // Then: Filters should be returned
    expect(filters).toEqual(initialState.filters);
  });

  test('GWT-SEL02: selectSelection returns selection state', () => {
    // Given: Mock root state
    // When: selectSelection selector is called
    const selection = selectSelection(mockRootState);

    // Then: Selection should be returned
    expect(selection).toEqual(initialState.selection);
  });

  test('GWT-SEL03: selectViewport returns viewport state', () => {
    // Given: Mock root state
    // When: selectViewport selector is called
    const viewport = selectViewport(mockRootState);

    // Then: Viewport should be returned
    expect(viewport).toEqual(initialState.viewport);
  });

  test('GWT-SEL04: selectSelectedNodeIds returns array', () => {
    // Given: State with selected nodes
    const stateWithSelection = {
      graph: {
        ...initialState,
        selection: {
          ...initialState.selection,
          selectedNodeIds: new Set(['node-1', 'node-2']),
        },
      },
    } as any;

    // When: selectSelectedNodeIds selector is called
    const nodeIds = selectSelectedNodeIds(stateWithSelection);

    // Then: Array of selected node IDs should be returned
    expect(Array.isArray(nodeIds)).toBe(true);
    expect(nodeIds).toHaveLength(2);
    expect(nodeIds).toContain('node-1');
    expect(nodeIds).toContain('node-2');
  });

  test('GWT-SEL05: selectHighlightedNodeIds returns array', () => {
    // Given: State with highlighted nodes
    const stateWithHighlights = {
      graph: {
        ...initialState,
        selection: {
          ...initialState.selection,
          highlightedNodeIds: new Set(['node-3', 'node-4']),
        },
      },
    } as any;

    // When: selectHighlightedNodeIds selector is called
    const nodeIds = selectHighlightedNodeIds(stateWithHighlights);

    // Then: Array of highlighted node IDs should be returned
    expect(Array.isArray(nodeIds)).toBe(true);
    expect(nodeIds).toHaveLength(2);
    expect(nodeIds).toContain('node-3');
    expect(nodeIds).toContain('node-4');
  });

  test('GWT-SEL06: selectEntityTypesArray returns array', () => {
    // Given: Mock root state
    // When: selectEntityTypesArray selector is called
    const types = selectEntityTypesArray(mockRootState);

    // Then: Array of entity types should be returned
    expect(Array.isArray(types)).toBe(true);
    expect(types).toHaveLength(5);
    expect(types).toContain('person');
    expect(types).toContain('organization');
  });

  test('GWT-SEL07: selectSelectionCount returns count', () => {
    // Given: State with 3 selected nodes
    const stateWithSelection = {
      graph: {
        ...initialState,
        selection: {
          ...initialState.selection,
          selectedNodeIds: new Set(['node-1', 'node-2', 'node-3']),
        },
      },
    } as any;

    // When: selectSelectionCount selector is called
    const count = selectSelectionCount(stateWithSelection);

    // Then: Count should be 3
    expect(count).toBe(3);
  });
});

// =============================================================================
// GWT TESTS: Edge Cases
// =============================================================================

describe('graphSlice - Edge Cases', () => {
  test('GWT-E01: deselectNode on non-existent node does nothing', () => {
    // Given: State with no selection
    const state = initialState;

    // When: deselectNode is called for non-existent node
    const newState = graphReducer(state, deselectNode('non-existent'));

    // Then: State should remain unchanged
    expect(newState.selection.selectedNodeIds.size).toBe(0);
  });

  test('GWT-E02: toggleEntityType with all types removed', () => {
    // Given: State with one entity type
    const state = {
      ...initialState,
      filters: {
        ...initialState.filters,
        entityTypes: new Set<EntityType>(['person']),
      },
    };

    // When: The last entity type is toggled off
    const newState = graphReducer(state, toggleEntityType('person'));

    // Then: entityTypes should be empty
    expect(newState.filters.entityTypes.size).toBe(0);
  });

  test('GWT-E03: Multiple actions chained correctly', () => {
    // Given: Initial state
    let state = initialState;

    // When: Multiple actions are dispatched in sequence
    state = graphReducer(state, setMinCitations(5));
    state = graphReducer(state, setMaxDepth(4));
    state = graphReducer(
      state,
      selectNode({ nodeId: 'node-1', multiSelect: false })
    );

    // Then: All changes should be applied
    expect(state.filters.minCitations).toBe(5);
    expect(state.filters.maxDepth).toBe(4);
    expect(state.selection.selectedNodeIds.has('node-1')).toBe(true);
  });
});

/**
 * Test Coverage Summary:
 * - Filter actions: 8 tests (GWT-F01 to GWT-F08)
 * - Selection actions: 7 tests (GWT-S01 to GWT-S07)
 * - Viewport actions: 2 tests (GWT-V01 to GWT-V02)
 * - Selectors: 7 tests (GWT-SEL01 to GWT-SEL07)
 * - Edge cases: 3 tests (GWT-E01 to GWT-E03)
 *
 * Total: 27 tests
 * Expected Coverage: >90%
 * JPL Compliance: ✓ All test functions < 60 lines
 */
