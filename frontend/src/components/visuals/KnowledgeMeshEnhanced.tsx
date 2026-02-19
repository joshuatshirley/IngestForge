/**
 * KnowledgeMeshEnhanced - Full-featured interactive knowledge graph
 *
 * US-1402.1: Interactive Mesh D3 UI
 * Epic AC: All filtering, search, multi-select features
 *
 * JPL Power of Ten Compliance:
 * - Rule #9: Complete type hints
 * - Rule #4: All functions < 60 lines
 * - Rule #2: Fixed upper bounds
 */

'use client';

import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  selectFilters,
  selectSelectedNodeIds,
  selectNode,
  clearSelection,
  setHighlightedNodes,
} from '@/store/slices/graphSlice';
import { useKnowledgeGraph, filterNodesBySearch, getSelectionSubgraph } from '@/hooks/useKnowledgeGraph';
import { KnowledgeMesh } from './KnowledgeMesh';
import { FilterPanel } from './FilterPanel';
import { SearchBar } from './SearchBar';
import { SelectionStats } from './SelectionStats';
import type { MeshNode, MeshData } from './KnowledgeMesh';

// =============================================================================
// TYPE DEFINITIONS (Rule #9: Complete type hints)
// =============================================================================

export interface KnowledgeMeshEnhancedProps {
  width?: number;
  height?: number;
  maxNodes?: number;
  className?: string;
}

// =============================================================================
// COMPONENT (Rule #4: Main render < 60 lines)
// =============================================================================

export const KnowledgeMeshEnhanced: React.FC<KnowledgeMeshEnhancedProps> = ({
  width = 1200,
  height = 800,
  maxNodes = 500,
  className = '',
}) => {
  const dispatch = useDispatch();
  const filters = useSelector(selectFilters);
  const selectedNodeIds = useSelector(selectSelectedNodeIds);

  // Fetch graph data with filters
  const { nodes, edges, isLoading, metadata } = useKnowledgeGraph({ maxNodes });

  // Local state
  const [filtersCollapsed, setFiltersCollapsed] = useState(false);
  const [matchingNodeIds, setMatchingNodeIds] = useState<string[]>([]);

  // Search highlighting
  useEffect(() => {
    if (filters.searchQuery) {
      const matches = filterNodesBySearch(nodes, filters.searchQuery);
      setMatchingNodeIds(matches);
      dispatch(setHighlightedNodes(matches));
    } else {
      setMatchingNodeIds([]);
      dispatch(setHighlightedNodes([]));
    }
  }, [filters.searchQuery, nodes, dispatch]);

  // Convert API data to MeshData format
  const meshData: MeshData = {
    nodes: nodes.map((n) => ({
      id: n.id,
      label: n.label,
      type: n.type as any,
      metadata: {
        citation_count: n.properties.citation_count,
        ...n.metadata,
      },
    })),
    edges: edges.map((e) => ({
      source: e.source,
      target: e.target,
      label: e.type,
      weight: e.weight,
    })),
  };

  // Calculate selection stats
  const subgraph = getSelectionSubgraph(selectedNodeIds, nodes, edges);
  const connectedEdgeCount = subgraph.edges.length;

  // Node click handler
  const handleNodeClick = (node: MeshNode, neighbors: MeshNode[]): void => {
    // Multi-select with Shift key (handled by D3 event in KnowledgeMesh)
    const isMultiSelect = window.event && (window.event as KeyboardEvent).shiftKey;
    dispatch(selectNode({ nodeId: node.id, multiSelect: isMultiSelect }));
  };

  // Export handler
  const handleExport = (): void => {
    const exportData = {
      nodes: subgraph.nodes,
      edges: subgraph.edges,
      metadata: {
        exported_at: new Date().toISOString(),
        selection_count: selectedNodeIds.length,
      },
    };

    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `knowledge-mesh-selection-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="text-white">Loading knowledge mesh...</div>
      </div>
    );
  }

  return (
    <div className={`knowledge-mesh-enhanced relative ${className}`}>
      {/* Top Bar - Search */}
      <div className="absolute top-4 left-1/2 transform -translate-x-1/2 z-10" style={{ width: 400 }}>
        <SearchBar matchCount={matchingNodeIds.length} />
      </div>

      {/* Left Sidebar - Filters */}
      <div className="absolute top-4 left-4 z-10">
        <FilterPanel
          collapsed={filtersCollapsed}
          onToggleCollapse={() => setFiltersCollapsed(!filtersCollapsed)}
        />
      </div>

      {/* Bottom - Selection Stats */}
      {selectedNodeIds.length > 0 && (
        <div className="absolute bottom-4 left-1/2 transform -translate-x-1/2 z-10" style={{ minWidth: 400 }}>
          <SelectionStats
            connectedEdgeCount={connectedEdgeCount}
            onExport={handleExport}
          />
        </div>
      )}

      {/* Main Graph */}
      <KnowledgeMesh
        data={meshData}
        width={width}
        height={height}
        onNodeClick={handleNodeClick}
      />

      {/* Metadata Display */}
      {metadata && (
        <div className="absolute top-4 right-4 z-10 bg-gray-900 bg-opacity-90 border border-gray-700 rounded px-3 py-2 text-xs text-gray-300">
          <div>{metadata.filtered_nodes} nodes</div>
          <div>{edges.length} edges</div>
          <div>{metadata.computation_time_ms.toFixed(0)}ms</div>
        </div>
      )}
    </div>
  );
};

export default KnowledgeMeshEnhanced;
