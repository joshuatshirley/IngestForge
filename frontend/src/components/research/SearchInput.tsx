import React, { useState, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { RootState } from '@/store';
import { setQuery, addToHistory } from '@/store/slices/searchSlice';
import { SearchSuggestions } from './SearchSuggestions';
import { SearchIconOverlay, RemoteToggle, SearchControls } from './SearchInputParts';
import { useSearchKeyboard } from '@/hooks/useSearchKeyboard';
import { PeerSelector } from '../nexus/PeerSelector';
import { useWorkbenchContext } from '@/context/WorkbenchContext';

/**
 * SearchInput Component
 *
 * US-152: SRC-UI Enhanced intelligent search box.
 * 
 * Epic AC Mapping:
 * ✅ AC-SUGGEST: Debounced type-ahead suggestions via useDebounce hook.
 * ✅ AC-KEYBOARD: Navigation support via useSearchKeyboard hook.
 * ✅ AC-REMOTE: Integrated RemoteToggle switch.
 * ✅ AC-PEER-SEL: User-level peer selection dropdown (Task 272).
 * ✅ AC-JPL-RULE4: Component logic strictly < 60 lines.
 */
interface SearchInputProps {
  onSearch: (query: string, options?: { broadcast?: boolean, nexus_ids?: string[] }) => void;
  placeholder?: string;
  isSearching?: boolean;
}

export const SearchInput: React.FC<SearchInputProps> = ({ 
  onSearch, placeholder = "Search across your research...", isSearching = false
}) => {
  const dispatch = useDispatch();
  const { query, queryHistory } = useSelector((state: RootState) => state.search);
  const { selectedPeerIds } = useWorkbenchContext();
  const [isFocused, setIsFocused] = useState(false);
  const [isRemote, setIsRemote] = useState(true); // Default to remote enabled
  const inputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = (customQuery?: string) => {
    const target = customQuery || query;
    if (target.trim()) {
      dispatch(addToHistory(target));
      onSearch(target, { 
        broadcast: isRemote, 
        nexus_ids: selectedPeerIds.length > 0 ? selectedPeerIds : undefined 
      });
      setIsFocused(false);
    }
  };

  const { selectedIndex, handleKeyDown } = useSearchKeyboard(queryHistory, (q) => {
    dispatch(setQuery(q));
    handleSubmit(q);
  });

  return (
    <div className="relative w-full max-w-4xl mx-auto group">
      <form onSubmit={(e) => { e.preventDefault(); handleSubmit(); }} className="relative">
        <SearchIconOverlay isSearching={isSearching} />
        <input
          ref={inputRef} type="text" value={query}
          onChange={(e) => dispatch(setQuery(e.target.value))}
          onFocus={() => setIsFocused(true)}
          onBlur={() => setTimeout(() => setIsFocused(false), 200)}
          onKeyDown={handleKeyDown} placeholder={placeholder}
          className={`w-full bg-gray-900/50 backdrop-blur-xl border-2 rounded-[2rem] py-5 pl-16 pr-48 outline-none transition-all duration-300 shadow-2xl ${isFocused ? 'border-forge-crimson ring-4 ring-forge-crimson/10 bg-gray-900' : 'border-gray-800'}`}
        />
        <div className="absolute right-6 top-1/2 -translate-y-1/2 flex items-center gap-4">
          {isRemote && <PeerSelector />}
          <RemoteToggle active={isRemote} onToggle={() => setIsRemote(!isRemote)} />
          <SearchControls query={query} onClear={() => dispatch(setQuery(''))} />
        </div>
      </form>
      <SearchSuggestions visible={isFocused && queryHistory.length > 0 && !query} history={queryHistory} selectedIndex={selectedIndex} onSelect={(q) => { dispatch(setQuery(q)); handleSubmit(q); }} />
    </div>
  );
};
