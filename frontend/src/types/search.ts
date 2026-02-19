/**
 * Unified Search Types for IngestForge.
 * 
 * Task 151: Define unified Search/Result TS models.
 * Strictly mirrors backend Pydantic models in ingestforge/core/models/search.py.
 * 
 * Rule #9: Complete type safety for API interaction.
 */

export interface SearchQuery {
  text: string;
  top_k: number;
  filters: Record<string, any>;
  broadcast: boolean;
  min_confidence: number;
}

export interface SearchResult {
  content: string;
  score: number; // 0.0 - 1.0
  confidence: number; // 0.0 - 1.0
  metadata: Record<string, any>;
  
  // Federated Attribution
  nexus_id: string; // 'local' or peer UUID
  artifact_id: string;
  document_id: string;
  
  // Coordinate Mapping
  page?: number;
}

export interface PeerFailure {
  nexus_id: string;
  error_type: 'TIMEOUT' | 'AUTHENTICATION_FAILED' | 'SERVER_ERROR' | 'CONNECTION_REFUSED' | 'UNKNOWN';
  status_code?: number;
  message: string;
}

export interface SearchResponse {
  results: SearchResult[];
  total_hits: number;
  query_time_ms: number;
  nexus_count: number;
  peer_failures: PeerFailure[];
}

/**
 * Filter factory for constructing complex search parameters.
 * Rule #4: Small, testable utility.
 */
export const createSearchQuery = (
  text: string, 
  overrides: Partial<SearchQuery> = {}
): SearchQuery => ({
  text,
  top_k: 10,
  filters: {},
  broadcast: false,
  min_confidence: 0.0,
  ...overrides
});
