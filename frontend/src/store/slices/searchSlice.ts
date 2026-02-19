import { createSlice, PayloadAction } from '@reduxjs/toolkit';
import { Message } from '@/components/chat/types';

/**
 * Search Store Slice
 *
 * US-0202: NL Search Redesign state management.
 * US-0204: Conversational Query state.
 * US-152: Extended filters for SRC-UI.
 */

const MAX_HISTORY = 50;

export interface SearchFilters {
  dateStart?: string;
  dateEnd?: string;
  docTypes: string[];
  sources: string[];
  entityTypes: string[]; // US-152
}

export interface SearchState {
  query: string;
  filters: SearchFilters;
  sortBy: 'relevance' | 'date' | 'author';
  queryHistory: string[];
  conversation: Message[];
  isSearching: boolean;
}

const initialState: SearchState = {
  query: '',
  filters: {
    docTypes: [],
    sources: [],
    entityTypes: [],
  },
  sortBy: 'relevance',
  queryHistory: [],
  conversation: [{ role: 'ai', text: "Hello! I've indexed your documents. How can I help you synthesize this research today?" }],
  isSearching: false,
};

export const searchSlice = createSlice({
  name: 'search',
  initialState,
  reducers: {
    setQuery: (state, action: PayloadAction<string>) => {
      state.query = action.payload;
    },
    setFilters: (state, action: PayloadAction<Partial<SearchFilters>>) => {
      state.filters = { ...state.filters, ...action.payload };
    },
    resetFilters: (state) => {
      state.filters = initialState.filters;
    },
    setSortBy: (state, action: PayloadAction<SearchState['sortBy']>) => {
      state.sortBy = action.payload;
    },
    addToHistory: (state, action: PayloadAction<string>) => {
      const query = action.payload.trim();
      if (!query) return;

      const filteredHistory = state.queryHistory.filter(q => q !== query);
      state.queryHistory = [query, ...filteredHistory].slice(0, MAX_HISTORY);
    },
    addMessage: (state, action: PayloadAction<Message>) => {
      state.conversation = [...state.conversation, action.payload].slice(-MAX_HISTORY);
    },
    clearSearch: (state) => {
      state.query = '';
      state.filters = initialState.filters;
      state.sortBy = 'relevance';
    },
    setSearching: (state, action: PayloadAction<boolean>) => {
      state.isSearching = action.payload;
    },
  },
});

export const { 
  setQuery, 
  setFilters, 
  resetFilters,
  setSortBy, 
  addToHistory, 
  addMessage,
  clearSearch, 
  setSearching 
} = searchSlice.actions;

export default searchSlice.reducer;
