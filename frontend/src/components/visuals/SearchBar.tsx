/**
 * SearchBar - Search and highlight nodes in graph
 *
 * US-1402.1: Interactive Mesh D3 UI
 * Epic AC: Search to highlight functionality
 *
 * JPL Power of Ten Compliance:
 * - Rule #9: Complete type hints
 * - Rule #4: All functions < 60 lines
 */

'use client';

import React, { useState, useEffect } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
  selectFilters,
  setSearchQuery,
  clearHighlights,
} from '@/store/slices/graphSlice';
import { debounce } from '@/utils/d3Helpers';

// =============================================================================
// TYPE DEFINITIONS (Rule #9: Complete type hints)
// =============================================================================

export interface SearchBarProps {
  className?: string;
  placeholder?: string;
  matchCount?: number;
  onSearch?: (query: string) => void;
}

// =============================================================================
// COMPONENT (Rule #4: < 60 lines)
// =============================================================================

export const SearchBar: React.FC<SearchBarProps> = ({
  className = '',
  placeholder = 'Search entities...',
  matchCount = 0,
  onSearch,
}) => {
  const dispatch = useDispatch();
  const filters = useSelector(selectFilters);
  const [localQuery, setLocalQuery] = useState(filters.searchQuery);

  // Debounced dispatch to Redux (300ms delay)
  useEffect(() => {
    const debouncedUpdate = debounce((query: string) => {
      dispatch(setSearchQuery(query));
      if (onSearch) {
        onSearch(query);
      }
    }, 300);

    debouncedUpdate(localQuery);
  }, [localQuery, dispatch, onSearch]);

  const handleClear = (): void => {
    setLocalQuery('');
    dispatch(setSearchQuery(''));
    dispatch(clearHighlights());
  };

  return (
    <div
      className={`search-bar bg-gray-900 border border-gray-700 rounded-lg ${className}`}
    >
      <div className="relative flex items-center">
        {/* Search Icon */}
        <div className="absolute left-3 text-gray-400">
          <svg
            xmlns="http://www.w3.org/2000/svg"
            className="h-5 w-5"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
            />
          </svg>
        </div>

        {/* Input */}
        <input
          type="text"
          value={localQuery}
          onChange={(e) => setLocalQuery(e.target.value)}
          placeholder={placeholder}
          className="w-full pl-10 pr-20 py-2 bg-transparent text-white placeholder-gray-500 focus:outline-none"
          aria-label="Search graph nodes"
        />

        {/* Match Count */}
        {localQuery && (
          <div className="absolute right-12 text-xs text-gray-400">
            {matchCount} match{matchCount !== 1 ? 'es' : ''}
          </div>
        )}

        {/* Clear Button */}
        {localQuery && (
          <button
            onClick={handleClear}
            className="absolute right-3 text-gray-400 hover:text-white"
            aria-label="Clear search"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="h-5 w-5"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        )}
      </div>
    </div>
  );
};

export default SearchBar;
