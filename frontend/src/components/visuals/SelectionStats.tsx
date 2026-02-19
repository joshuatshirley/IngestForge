/**
 * SelectionStats - Display statistics for selected nodes
 *
 * US-1402.1: Interactive Mesh D3 UI
 * Epic AC: Multi-select with aggregated statistics
 *
 * JPL Power of Ten Compliance:
 * - Rule #9: Complete type hints
 * - Rule #4: All functions < 60 lines
 */

'use client';

import React from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { selectSelectionCount, clearSelection } from '@/store/slices/graphSlice';

// =============================================================================
// TYPE DEFINITIONS (Rule #9: Complete type hints)
// =============================================================================

export interface SelectionStatsProps {
  className?: string;
  connectedEdgeCount?: number;
  onExport?: () => void;
  onTag?: () => void;
}

// =============================================================================
// COMPONENT (Rule #4: < 60 lines)
// =============================================================================

export const SelectionStats: React.FC<SelectionStatsProps> = ({
  className = '',
  connectedEdgeCount = 0,
  onExport,
  onTag,
}) => {
  const dispatch = useDispatch();
  const selectionCount = useSelector(selectSelectionCount);

  if (selectionCount === 0) {
    return null;
  }

  return (
    <div
      className={`selection-stats bg-blue-900 bg-opacity-90 border border-blue-700 rounded-lg p-3 ${className}`}
    >
      <div className="flex items-center justify-between gap-4">
        {/* Stats */}
        <div className="flex items-center gap-4 text-sm text-white">
          <div>
            <span className="font-semibold">{selectionCount}</span> node
            {selectionCount !== 1 ? 's' : ''} selected
          </div>
          <div className="text-gray-300">
            {connectedEdgeCount} connection
            {connectedEdgeCount !== 1 ? 's' : ''}
          </div>
        </div>

        {/* Actions */}
        <div className="flex items-center gap-2">
          {onExport && (
            <button
              onClick={onExport}
              className="px-3 py-1 text-xs bg-blue-700 hover:bg-blue-600 text-white rounded transition"
              aria-label="Export selection"
            >
              Export
            </button>
          )}
          {onTag && (
            <button
              onClick={onTag}
              className="px-3 py-1 text-xs bg-blue-700 hover:bg-blue-600 text-white rounded transition"
              aria-label="Tag selection"
            >
              Tag
            </button>
          )}
          <button
            onClick={() => dispatch(clearSelection())}
            className="px-3 py-1 text-xs bg-gray-700 hover:bg-gray-600 text-white rounded transition"
            aria-label="Clear selection"
          >
            Clear
          </button>
        </div>
      </div>
    </div>
  );
};

export default SelectionStats;
