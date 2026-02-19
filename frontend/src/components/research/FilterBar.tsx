import React from 'react';
import { Filter, X } from 'lucide-react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '@/store';
import { setFilters, resetFilters } from '@/store/slices/searchSlice';

/**
 * US-152: SRC-UI FilterBar Component.
 * Enhanced with entity types and active filter badges.
 */

const DOC_TYPES = ['PDF', 'EPUB', 'MD', 'DOCX', 'CODE'];
const ENTITY_TYPES = ['PERSON', 'ORG', 'LOC', 'DATE', 'CVE', 'DOCKET'];

export const FilterBar: React.FC = () => {
  const dispatch = useDispatch();
  const { filters } = useSelector((state: RootState) => state.search);

  const hasFilters = filters.docTypes.length > 0 || filters.sources.length > 0 || filters.entityTypes.length > 0;

  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-4 py-2 overflow-x-auto no-scrollbar">
        <FilterIndicator />
        <div className="h-6 w-px bg-gray-800 mx-2" />
        
        <FilterGroup 
          items={DOC_TYPES} 
          activeItems={filters.docTypes} 
          onToggle={(next) => dispatch(setFilters({ docTypes: next }))} 
          colorClass="bg-forge-crimson"
        />

        <div className="h-6 w-px bg-gray-800 mx-2" />

        <FilterGroup 
          items={ENTITY_TYPES} 
          activeItems={filters.entityTypes} 
          onToggle={(next) => dispatch(setFilters({ entityTypes: next }))} 
          colorClass="bg-forge-cyan text-black"
        />

        {hasFilters && (
          <button onClick={() => dispatch(resetFilters())} className="flex items-center gap-2 px-3 py-2 text-[10px] font-bold text-red-400 hover:text-red-300 transition-colors uppercase tracking-widest ml-auto">
            <X size={12} /> Clear All
          </button>
        )}
      </div>
    </div>
  );
};

// --- Sub-components for Rule #4 Compliance ---

const FilterIndicator = () => (
  <div className="flex items-center gap-2 px-4 py-2 bg-gray-900 border border-gray-800 rounded-xl text-gray-400">
    <Filter size={14} />
    <span className="text-[10px] font-bold uppercase tracking-widest">Filters</span>
  </div>
);

const FilterGroup: React.FC<{ items: string[]; activeItems: string[]; onToggle: (next: string[]) => void; colorClass: string }> = ({ 
  items, activeItems, onToggle, colorClass 
}) => {
  // JPL Rule #2: Safety bound on filter items
  const displayItems = items.slice(0, 20);

  const toggle = (item: string) => {
    const next = activeItems.includes(item)
      ? activeItems.filter(i => i !== item)
      : [...activeItems, item];
    onToggle(next);
  };

  return (
    <div className="flex items-center gap-2">
      {displayItems.map(item => (
        <button
          key={item}
          onClick={() => toggle(item)}
          className={`px-4 py-2 rounded-xl text-[10px] font-bold transition-all border ${
            activeItems.includes(item)
              ? `${colorClass} border-transparent shadow-lg`
              : 'bg-gray-900/50 border-gray-800 text-gray-500 hover:border-gray-700'
          }`}
        >
          {item}
        </button>
      ))}
    </div>
  );
};
