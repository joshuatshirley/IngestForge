/**
 * Graph State Slice - Redux state for knowledge mesh visualization
 *
 * US-1402.1: Interactive Mesh D3 UI
 * Epic AC: Filter by type, search, multi-select
 *
 * JPL Power of Ten Compliance:
 * - Rule #9: Complete type hints
 * - Rule #4: All functions < 60 lines
 */

import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import type { RootState } from '../index';

// =============================================================================
// TYPE DEFINITIONS (Rule #9: Complete type hints)
// =============================================================================

/** Entity types for filtering */
export type EntityType =
  | 'person'
  | 'organization'
  | 'location'
  | 'concept'
  | 'document'
  | 'chunk'
  | 'entity'
  | 'event'
  | 'date';

/** Filter state */
export interface GraphFilters {
  entityTypes: Set<EntityType>;
  minCitations: number;
  maxDepth: number;
  showChunks: boolean;
  searchQuery: string;
}

/** Selection state */
export interface GraphSelection {
  selectedNodeIds: Set<string>;
  highlightedNodeIds: Set<string>;
}

/** Viewport transform state */
export interface GraphViewport {
  scale: number;
  x: number;
  y: number;
}

/** Complete graph state */
export interface GraphState {
  filters: GraphFilters;
  selection: GraphSelection;
  viewport: GraphViewport;
}

// =============================================================================
// INITIAL STATE
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
// SLICE DEFINITION (Rule #4: Reducers < 60 lines)
// =============================================================================

const graphSlice = createSlice({
  name: 'graph',
  initialState,
  reducers: {
    // Filter actions
    setMinCitations(state, action: PayloadAction<number>) {
      state.filters.minCitations = action.payload;
    },

    setMaxDepth(state, action: PayloadAction<number>) {
      state.filters.maxDepth = action.payload;
    },

    setShowChunks(state, action: PayloadAction<boolean>) {
      state.filters.showChunks = action.payload;
    },

    setSearchQuery(state, action: PayloadAction<string>) {
      state.filters.searchQuery = action.payload;
    },

    toggleEntityType(state, action: PayloadAction<EntityType>) {
      const type = action.payload;
      const newTypes = new Set(state.filters.entityTypes);

      if (newTypes.has(type)) {
        newTypes.delete(type);
      } else {
        newTypes.add(type);
      }

      state.filters.entityTypes = newTypes;
    },

    setEntityTypes(state, action: PayloadAction<EntityType[]>) {
      state.filters.entityTypes = new Set(action.payload);
    },

    resetFilters(state) {
      state.filters = initialState.filters;
    },

    // Selection actions
    selectNode(
      state,
      action: PayloadAction<{ nodeId: string; multiSelect: boolean }>
    ) {
      const { nodeId, multiSelect } = action.payload;
      const newSelection = new Set(
        multiSelect ? state.selection.selectedNodeIds : []
      );
      newSelection.add(nodeId);
      state.selection.selectedNodeIds = newSelection;
    },

    deselectNode(state, action: PayloadAction<string>) {
      const newSelection = new Set(state.selection.selectedNodeIds);
      newSelection.delete(action.payload);
      state.selection.selectedNodeIds = newSelection;
    },

    clearSelection(state) {
      state.selection.selectedNodeIds = new Set();
      state.selection.highlightedNodeIds = new Set();
    },

    setHighlightedNodes(state, action: PayloadAction<string[]>) {
      state.selection.highlightedNodeIds = new Set(action.payload);
    },

    clearHighlights(state) {
      state.selection.highlightedNodeIds = new Set();
    },

    // Viewport actions
    setViewport(
      state,
      action: PayloadAction<{ scale: number; x: number; y: number }>
    ) {
      state.viewport = action.payload;
    },

    resetViewport(state) {
      state.viewport = initialState.viewport;
    },
  },
});

// =============================================================================
// ACTIONS EXPORT
// =============================================================================

export const {
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
} = graphSlice.actions;

// =============================================================================
// SELECTORS (Memoized)
// =============================================================================

/** Get current filters */
export const selectFilters = (state: RootState): GraphFilters =>
  state.graph.filters;

/** Get current selection */
export const selectSelection = (state: RootState): GraphSelection =>
  state.graph.selection;

/** Get current viewport */
export const selectViewport = (state: RootState): GraphViewport =>
  state.graph.viewport;

/** Get selected node IDs as array */
export const selectSelectedNodeIds = (state: RootState): string[] =>
  Array.from(state.graph.selection.selectedNodeIds);

/** Get highlighted node IDs as array */
export const selectHighlightedNodeIds = (state: RootState): string[] =>
  Array.from(state.graph.selection.highlightedNodeIds);

/** Get entity types as array */
export const selectEntityTypesArray = (state: RootState): EntityType[] =>
  Array.from(state.graph.filters.entityTypes);

/** Check if a node is selected */
export const selectIsNodeSelected = (nodeId: string) => (
  state: RootState
): boolean => state.graph.selection.selectedNodeIds.has(nodeId);

/** Check if a node is highlighted */
export const selectIsNodeHighlighted = (nodeId: string) => (
  state: RootState
): boolean => state.graph.selection.highlightedNodeIds.has(nodeId);

/** Get selection count */
export const selectSelectionCount = (state: RootState): number =>
  state.graph.selection.selectedNodeIds.size;

// =============================================================================
// REDUCER EXPORT
// =============================================================================

export default graphSlice.reducer;
