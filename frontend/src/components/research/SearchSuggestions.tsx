import React from 'react';
import { Search, History } from 'lucide-react';

/**
 * US-152: SRC-UI Search Suggestions.
 * Enhanced with keyboard selection support.
 */

interface SearchSuggestionsProps {
  visible: boolean;
  history: string[];
  selectedIndex: number;
  onSelect: (query: string) => void;
}

export const SearchSuggestions: React.FC<SearchSuggestionsProps> = ({ 
  visible, 
  history, 
  selectedIndex,
  onSelect 
}) => {
  if (!visible) return null;

  return (
    <div className="absolute top-full mt-4 w-full bg-gray-900 border border-gray-800 rounded-3xl shadow-2xl overflow-hidden z-50 animate-in fade-in slide-in-from-top-2 duration-200">
      <div className="p-4 border-b border-gray-800">
        <h4 className="text-[10px] font-black text-gray-500 uppercase tracking-widest flex items-center gap-2">
          <History size={12} />
          Recent Searches
        </h4>
      </div>
      <div className="max-h-64 overflow-y-auto custom-scrollbar">
        {history.slice(0, 10).map((h, i) => (
          <button
            key={i}
            onClick={() => onSelect(h)}
            className={`w-full text-left px-6 py-4 transition-colors flex items-center gap-4 group ${
              selectedIndex === i ? 'bg-gray-800 border-l-4 border-forge-crimson' : 'hover:bg-gray-800'
            }`}
          >
            <Search size={14} className={`text-gray-600 ${selectedIndex === i ? 'text-forge-crimson' : 'group-hover:text-forge-crimson'}`} />
            <span className={`text-sm ${selectedIndex === i ? 'text-white' : 'text-gray-300'}`}>{h}</span>
          </button>
        ))}
      </div>
    </div>
  );
};
