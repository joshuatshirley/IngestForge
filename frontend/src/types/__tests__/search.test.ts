/**
 * GWT Unit Tests for Search Types - Task 151.
 */

import { createSearchQuery, SearchQuery } from '../search';

describe('Search Types & Factories', () => {
  
  // GIVEN: A search string
  describe('createSearchQuery factory', () => {
    
    test('GIVEN a text string WHEN called THEN returns valid SearchQuery with defaults', () => {
      // Given
      const text = 'Find security vulnerabilities';
      
      // When
      const query = createSearchQuery(text);
      
      // Then
      expect(query.text).toBe(text);
      expect(query.top_k).toBe(10);
      expect(query.broadcast).toBe(false);
      expect(query.filters).toEqual({});
    });

    test('GIVEN overrides WHEN called THEN merges them correctly', () => {
      // Given
      const text = 'test';
      const overrides: Partial<SearchQuery> = {
        top_k: 50,
        broadcast: true,
        filters: { domain: 'cyber' }
      };
      
      // When
      const query = createSearchQuery(text, overrides);
      
      // Then
      expect(query.top_k).toBe(50);
      expect(query.broadcast).toBe(true);
      expect(query.filters.domain).toBe('cyber');
    });

  });
});
