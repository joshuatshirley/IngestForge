/**
 * FilterPanel - Controls for filtering knowledge graph
 *
 * US-1402.1: Interactive Mesh D3 UI
 * Epic AC: Filter by type, citations, depth
 *
 * JPL Power of Ten Compliance:
 * - Rule #9: Complete type hints
 * - Rule #4: All functions < 60 lines
 */

'use client';

import React from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  selectFilters,
  selectEntityTypesArray,
  toggleEntityType,
  setMinCitations,
  setMaxDepth,
  setShowChunks,
  resetFilters,
  type EntityType,
} from '@/store/slices/graphSlice';

// =============================================================================
// TYPE DEFINITIONS (Rule #9: Complete type hints)
// =============================================================================

export interface FilterPanelProps {
  className?: string;
  collapsed?: boolean;
  onToggleCollapse?: () => void;
}

// =============================================================================
// COMPONENT (Rule #4: < 60 lines)
// =============================================================================

export const FilterPanel: React.FC<FilterPanelProps> = ({
  className = '',
  collapsed = false,
  onToggleCollapse,
}) => {
  const dispatch = useDispatch();
  const filters = useSelector(selectFilters);
  const entityTypesArray = useSelector(selectEntityTypesArray);

  const entityTypeOptions: EntityType[] = [
    'person',
    'organization',
    'location',
    'concept',
    'document',
  ];

  if (collapsed) {
    return (
      <div className={`filter-panel-collapsed ${className}`}>
        <button
          onClick={onToggleCollapse}
          className="px-3 py-2 bg-gray-800 text-white rounded hover:bg-gray-700"
          aria-label="Expand filters"
        >
          Filters ▶
        </button>
      </div>
    );
  }

  return (
    <div
      className={`filter-panel bg-gray-900 border border-gray-700 rounded-lg p-4 ${className}`}
      style={{ width: 280 }}
    >
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Filters</h3>
        <div className="flex gap-2">
          <button
            onClick={() => dispatch(resetFilters())}
            className="text-xs text-blue-400 hover:text-blue-300"
          >
            Reset
          </button>
          {onToggleCollapse && (
            <button
              onClick={onToggleCollapse}
              className="text-gray-400 hover:text-gray-300"
              aria-label="Collapse filters"
            >
              ◀
            </button>
          )}
        </div>
      </div>

      {/* Entity Type Checkboxes */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Entity Types
        </label>
        <div className="space-y-2">
          {entityTypeOptions.map((type) => (
            <label
              key={type}
              className="flex items-center text-sm text-gray-300 hover:text-white cursor-pointer"
            >
              <input
                type="checkbox"
                checked={entityTypesArray.includes(type)}
                onChange={() => dispatch(toggleEntityType(type))}
                className="mr-2 rounded"
              />
              <span className="capitalize">{type}</span>
            </label>
          ))}
        </div>
      </div>

      {/* Citation Count Slider */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Min Citations: {filters.minCitations}
        </label>
        <input
          type="range"
          min={0}
          max={100}
          value={filters.minCitations}
          onChange={(e) =>
            dispatch(setMinCitations(parseInt(e.target.value, 10)))
          }
          className="w-full"
        />
        <div className="flex justify-between text-xs text-gray-500">
          <span>0</span>
          <span>100</span>
        </div>
      </div>

      {/* Graph Depth Slider */}
      <div className="mb-4">
        <label className="block text-sm font-medium text-gray-300 mb-2">
          Graph Depth: {filters.maxDepth} hops
        </label>
        <input
          type="range"
          min={1}
          max={5}
          value={filters.maxDepth}
          onChange={(e) =>
            dispatch(setMaxDepth(parseInt(e.target.value, 10)))
          }
          className="w-full"
        />
        <div className="flex justify-between text-xs text-gray-500">
          <span>1</span>
          <span>5</span>
        </div>
      </div>

      {/* Show Chunks Toggle */}
      <div className="mb-2">
        <label className="flex items-center text-sm text-gray-300 hover:text-white cursor-pointer">
          <input
            type="checkbox"
            checked={filters.showChunks}
            onChange={(e) => dispatch(setShowChunks(e.target.checked))}
            className="mr-2 rounded"
          />
          <span>Show chunk nodes</span>
        </label>
      </div>
    </div>
  );
};

export default FilterPanel;
