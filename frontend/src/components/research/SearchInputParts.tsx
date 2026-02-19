import React from 'react';
import { Search, X, Command, Loader2, Globe } from 'lucide-react';

/**
 * US-152: SRC-UI SearchInput sub-components.
 */

export const SearchIconOverlay: React.FC<{ isSearching: boolean }> = ({ isSearching }) => (
  <div className="absolute left-6 top-1/2 -translate-y-1/2 text-gray-500">
    {isSearching ? <Loader2 size={20} className="animate-spin" /> : <Search size={20} />}
  </div>
);

export const RemoteToggle: React.FC<{ active: boolean; onToggle: () => void }> = ({ active, onToggle }) => (
  <button 
    type="button"
    onClick={onToggle}
    className={`p-2 rounded-lg border transition-all flex items-center gap-2 ${active ? 'bg-forge-cyan/20 border-forge-cyan text-forge-cyan' : 'bg-gray-800 border-gray-700 text-gray-500 hover:border-gray-600'}`}
    title="Remote Search Toggle"
  >
    <Globe size={14} />
    <span className="text-[10px] font-bold uppercase hidden md:inline">Remote</span>
  </button>
);

export const SearchControls: React.FC<{ query: string; onClear: () => void }> = ({ query, onClear }) => (
  <div className="flex items-center gap-3">
    {query && (
      <button type="button" onClick={onClear} className="p-2 hover:bg-gray-800 rounded-full text-gray-500 transition-colors">
        <X size={18} />
      </button>
    )}
    <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg border border-gray-800 bg-gray-900 text-gray-500 text-[10px] font-bold uppercase tracking-tighter">
      <Command size={10} />
      <span>K</span>
    </div>
  </div>
);
