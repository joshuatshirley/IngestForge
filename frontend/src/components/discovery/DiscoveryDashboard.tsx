/**
 * Discovery Dashboard Component - US-603
 *
 * Displays recommended discovery tasks from Proactive Scout.
 *
 * JPL Power of Ten Compliance:
 * - Rule #4: All functions < 60 lines
 * - Rule #9: Complete type hints
 *
 * Epic: EP-06 (Agentic Intelligence)
 * Feature: US-603 (Proactive Scout)
 * Date: 2026-02-18
 */

import React, { useState, useEffect, useCallback } from 'react';

// =============================================================================
// TYPES (Rule #9: Complete type hints)
// =============================================================================

export interface DiscoveryIntent {
  intent_id: string;
  target_entity: string;
  entity_type: string;
  missing_link_type: string;
  rationale: string;
  confidence: number;
  priority_score: number;
  current_references: number;
  suggested_searches: string[];
}

export interface DiscoveryResponse {
  total_intents: number;
  returned_intents: number;
  intents: DiscoveryIntent[];
  filters_applied: {
    limit: number;
    min_confidence: number;
  };
}

export interface DiscoveryDashboardProps {
  apiBaseUrl?: string;
  limit?: number;
  minConfidence?: number;
  onIntentClick?: (intent: DiscoveryIntent) => void;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

// =============================================================================
// CONSTANTS (JPL Rule #2: Fixed upper bounds)
// =============================================================================

const DEFAULT_API_BASE_URL = '';
const DEFAULT_LIMIT = 20;
const DEFAULT_MIN_CONFIDENCE = 0.6;
const DEFAULT_REFRESH_INTERVAL = 30000; // 30 seconds
const MAX_RETRIES = 3;
const MAX_RENDER_ITEMS = 100; // JPL Rule #2: Bounded rendering

// Confidence color mapping
const CONFIDENCE_COLORS: Record<string, string> = {
  high: '#4CAF50', // Green
  medium: '#FFEB3B', // Yellow
  low: '#F44336', // Red
};

// =============================================================================
// MAIN COMPONENT (Rule #4: Main render < 60 lines)
// =============================================================================

export const DiscoveryDashboard: React.FC<DiscoveryDashboardProps> = ({
  apiBaseUrl = DEFAULT_API_BASE_URL,
  limit = DEFAULT_LIMIT,
  minConfidence = DEFAULT_MIN_CONFIDENCE,
  onIntentClick,
  autoRefresh = false,
  refreshInterval = DEFAULT_REFRESH_INTERVAL,
}) => {
  const [intents, setIntents] = useState<DiscoveryIntent[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [totalIntents, setTotalIntents] = useState<number>(0);
  const [sortField, setSortField] = useState<keyof DiscoveryIntent>('priority_score');
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('desc');

  // Fetch discovery intents
  const fetchIntents = useCallback(async (): Promise<void> => {
    try {
      setLoading(true);
      setError(null);

      const response = await fetch(
        `${apiBaseUrl}/v1/discovery/recommended?limit=${limit}&min_confidence=${minConfidence}`
      );

      if (!response.ok) {
        throw new Error(`Failed to fetch intents: ${response.statusText}`);
      }

      const data: DiscoveryResponse = await response.json();

      setIntents(data.intents);
      setTotalIntents(data.total_intents);
      setLoading(false);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Unknown error';
      setError(errorMessage);
      setLoading(false);
    }
  }, [apiBaseUrl, limit, minConfidence]);

  // Initial fetch
  useEffect(() => {
    fetchIntents();
  }, [fetchIntents]);

  // Auto-refresh
  useEffect(() => {
    if (!autoRefresh) return;

    const intervalId = setInterval(() => {
      fetchIntents();
    }, refreshInterval);

    return () => clearInterval(intervalId);
  }, [autoRefresh, refreshInterval, fetchIntents]);

  // Sort intents
  const sortedIntents = React.useMemo(() => {
    return sortIntents(intents, sortField, sortDirection);
  }, [intents, sortField, sortDirection]);

  // Handle sort
  const handleSort = useCallback((field: keyof DiscoveryIntent): void => {
    if (field === sortField) {
      setSortDirection((prev) => (prev === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  }, [sortField]);

  // Handle intent click
  const handleIntentClick = useCallback(
    (intent: DiscoveryIntent): void => {
      if (onIntentClick) {
        onIntentClick(intent);
      }
    },
    [onIntentClick]
  );

  return (
    <div className="discovery-dashboard" data-testid="discovery-dashboard">
      <div className="dashboard-header">
        <h2>Recommended Discoveries</h2>
        <div className="dashboard-stats">
          <span>
            Showing {sortedIntents.length} of {totalIntents} recommendations
          </span>
          <button onClick={fetchIntents} disabled={loading} className="refresh-button">
            Refresh
          </button>
        </div>
      </div>

      {loading && <div className="loading-spinner">Loading recommendations...</div>}

      {error && (
        <div className="error-message" data-testid="error-message">
          Error: {error}
        </div>
      )}

      {!loading && !error && sortedIntents.length === 0 && (
        <div className="empty-state">No discovery recommendations found.</div>
      )}

      {!loading && !error && sortedIntents.length > 0 && (
        <table className="intents-table" data-testid="intents-table">
          <thead>
            <tr>
              <th onClick={() => handleSort('priority_score')} className="sortable">
                Priority {sortField === 'priority_score' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th onClick={() => handleSort('target_entity')} className="sortable">
                Entity {sortField === 'target_entity' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th>Type</th>
              <th>Missing Link</th>
              <th onClick={() => handleSort('confidence')} className="sortable">
                Confidence {sortField === 'confidence' && (sortDirection === 'asc' ? '↑' : '↓')}
              </th>
              <th>Rationale</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {/* JPL Rule #2: Bounded rendering to MAX_RENDER_ITEMS */}
            {sortedIntents.slice(0, MAX_RENDER_ITEMS).map((intent) => (
              <tr
                key={intent.intent_id}
                onClick={() => handleIntentClick(intent)}
                className="intent-row"
                data-testid={`intent-row-${intent.intent_id}`}
              >
                <td>
                  <PriorityBadge score={intent.priority_score} />
                </td>
                <td className="entity-cell">
                  <strong>{intent.target_entity}</strong>
                  <span className="reference-count">
                    ({intent.current_references} refs)
                  </span>
                </td>
                <td className="type-cell">{intent.entity_type}</td>
                <td className="link-type-cell">{formatLinkType(intent.missing_link_type)}</td>
                <td>
                  <ConfidenceBadge confidence={intent.confidence} />
                </td>
                <td className="rationale-cell">{intent.rationale}</td>
                <td>
                  <button
                    className="search-button"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleIntentClick(intent);
                    }}
                  >
                    Explore
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

// =============================================================================
// HELPER COMPONENTS (Rule #4: Each < 60 lines)
// =============================================================================

interface PriorityBadgeProps {
  score: number;
}

const PriorityBadge: React.FC<PriorityBadgeProps> = ({ score }) => {
  const level = score >= 0.8 ? 'high' : score >= 0.6 ? 'medium' : 'low';
  const color = CONFIDENCE_COLORS[level];

  return (
    <span
      className={`priority-badge priority-${level}`}
      style={{ backgroundColor: color }}
      data-testid="priority-badge"
    >
      {(score * 100).toFixed(0)}%
    </span>
  );
};

interface ConfidenceBadgeProps {
  confidence: number;
}

const ConfidenceBadge: React.FC<ConfidenceBadgeProps> = ({ confidence }) => {
  const level = confidence >= 0.8 ? 'high' : confidence >= 0.6 ? 'medium' : 'low';
  const color = CONFIDENCE_COLORS[level];

  return (
    <span
      className={`confidence-badge confidence-${level}`}
      style={{ color }}
      data-testid="confidence-badge"
    >
      {(confidence * 100).toFixed(0)}%
    </span>
  );
};

// =============================================================================
// HELPER FUNCTIONS (Rule #4: Each < 60 lines)
// =============================================================================

function sortIntents(
  intents: DiscoveryIntent[],
  field: keyof DiscoveryIntent,
  direction: 'asc' | 'desc'
): DiscoveryIntent[] {
  const sorted = [...intents].sort((a, b) => {
    const aVal = a[field];
    const bVal = b[field];

    if (typeof aVal === 'number' && typeof bVal === 'number') {
      return direction === 'asc' ? aVal - bVal : bVal - aVal;
    }

    if (typeof aVal === 'string' && typeof bVal === 'string') {
      return direction === 'asc'
        ? aVal.localeCompare(bVal)
        : bVal.localeCompare(aVal);
    }

    return 0;
  });

  return sorted;
}

function formatLinkType(linkType: string): string {
  // JPL Rule #2: Bounded string split
  const MAX_PARTS = 10;
  const parts = linkType.split('_').slice(0, MAX_PARTS);

  // Convert snake_case to Title Case
  return parts
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
}

export default DiscoveryDashboard;
