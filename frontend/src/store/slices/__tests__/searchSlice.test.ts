import searchReducer, { 
  setQuery, 
  setFilters, 
  addToHistory, 
  clearSearch 
} from '../searchSlice';

/**
 * Search Slice Unit Tests
 * 
 * US-0202: NL Search Redesign
 * JPL Rule #2: Bounded query history
 */

describe('searchSlice Reducer', () => {
  const initialState = {
    query: '',
    filters: { docTypes: [], sources: [] },
    sortBy: 'relevance' as const,
    queryHistory: [],
    isSearching: false,
  };

  it('GIVEN initial state WHEN setQuery is dispatched THEN updates query', () => {
    const action = setQuery('quantum physics');
    const state = searchReducer(initialState, action);
    expect(state.query).toBe('quantum physics');
  });

  it('GIVEN initial state WHEN setFilters is dispatched THEN merges filters', () => {
    const action = setFilters({ docTypes: ['PDF'] });
    const state = searchReducer(initialState, action);
    expect(state.filters.docTypes).toEqual(['PDF']);
    expect(state.filters.sources).toEqual([]);
  });

  it('GIVEN a query history WHEN addToHistory is dispatched THEN prepends query and bounds list', () => {
    let state = initialState;
    
    // Fill history to limit
    for (let i = 0; i < 60; i++) {
      state = searchReducer(state, addToHistory(`query ${i}`));
    }

    // JPL Rule #2: Enforce max 50
    expect(state.queryHistory).toHaveLength(50);
    expect(state.queryHistory[0]).toBe('query 59');
  });

  it('GIVEN an existing query in history WHEN added again THEN moves to top without duplicates', () => {
    const stateWithHistory = {
      ...initialState,
      queryHistory: ['old', 'older', 'oldest']
    };
    
    const state = searchReducer(stateWithHistory, addToHistory('older'));
    
    expect(state.queryHistory).toEqual(['older', 'old', 'oldest']);
    expect(state.queryHistory).toHaveLength(3);
  });

  it('GIVEN active search WHEN clearSearch is dispatched THEN resets state', () => {
    const activeState = {
      ...initialState,
      query: 'some query',
      filters: { docTypes: ['PDF'], sources: ['Local'] }
    };
    
    const state = searchReducer(activeState, clearSearch());
    
    expect(state.query).toBe('');
    expect(state.filters.docTypes).toHaveLength(0);
  });
});
