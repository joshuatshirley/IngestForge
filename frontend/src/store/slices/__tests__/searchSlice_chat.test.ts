import searchReducer, { addMessage } from '../searchSlice';

/**
 * Chat History Unit Tests
 * US-0204: Conversational Query
 */

describe('searchSlice: Chat History Bounding', () => {
  const initialState = {
    query: '',
    filters: { docTypes: [], sources: [] },
    sortBy: 'relevance' as const,
    queryHistory: [],
    conversation: [],
    isSearching: false,
  };

  it('GIVEN a full conversation WHEN addMessage is dispatched THEN it enforces JPL Rule #2 limit', () => {
    let state = initialState;
    
    // Fill beyond limit (50)
    for (let i = 0; i < 60; i++) {
      state = searchReducer(state, addMessage({ role: 'user', text: `Message ${i}` }));
    }

    // THEN history is capped at 50
    expect(state.conversation).toHaveLength(50);
    // AND it kept the LATEST 50
    expect(state.conversation[0].text).toBe('Message 10');
    expect(state.conversation[49].text).toBe('Message 59');
  });
});
