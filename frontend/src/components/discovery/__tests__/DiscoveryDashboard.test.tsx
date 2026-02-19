/**
 * Unit tests for DiscoveryDashboard component - US-603
 *
 * Tests rendering, sorting, and interaction with discovery intents.
 *
 * JPL Power of Ten Compliance:
 * - Rule #4: All test functions < 60 lines
 * - Rule #9: Complete type hints
 *
 * Epic: EP-06 (Agentic Intelligence)
 * Feature: US-603 (Proactive Scout)
 * Test Date: 2026-02-18
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { DiscoveryDashboard, DiscoveryIntent, DiscoveryResponse } from '../DiscoveryDashboard';

// =============================================================================
// TEST FIXTURES
// =============================================================================

const mockIntents: DiscoveryIntent[] = [
  {
    intent_id: 'intent-1',
    target_entity: 'John Doe',
    entity_type: 'PERSON',
    missing_link_type: 'cross_document_reference',
    rationale: 'Entity appears in only one document.',
    confidence: 0.85,
    priority_score: 0.9,
    current_references: 1,
    suggested_searches: ['John Doe', 'John Doe PERSON'],
  },
  {
    intent_id: 'intent-2',
    target_entity: 'Apple Inc',
    entity_type: 'ORG',
    missing_link_type: 'additional_context',
    rationale: 'Entity has weak connectivity.',
    confidence: 0.72,
    priority_score: 0.75,
    current_references: 1,
    suggested_searches: ['Apple Inc', 'Apple Inc ORG'],
  },
  {
    intent_id: 'intent-3',
    target_entity: 'Python',
    entity_type: 'TECH',
    missing_link_type: 'related_sources',
    rationale: 'Could benefit from more sources.',
    confidence: 0.68,
    priority_score: 0.65,
    current_references: 3,
    suggested_searches: ['Python', 'Python TECH'],
  },
];

const mockResponse: DiscoveryResponse = {
  total_intents: 3,
  returned_intents: 3,
  intents: mockIntents,
  filters_applied: {
    limit: 20,
    min_confidence: 0.6,
  },
};

// =============================================================================
// BASIC RENDERING TESTS (5 tests)
// =============================================================================

describe('DiscoveryDashboard - Rendering', () => {
  beforeEach(() => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })
    ) as jest.Mock;
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('GIVEN component WHEN rendered THEN displays dashboard', async () => {
    render(<DiscoveryDashboard />);

    await waitFor(() => {
      expect(screen.getByTestId('discovery-dashboard')).toBeInTheDocument();
    });
  });

  test('GIVEN loading state WHEN rendered THEN shows loading spinner', () => {
    global.fetch = jest.fn(() => new Promise(() => {})) as jest.Mock;

    render(<DiscoveryDashboard />);

    expect(screen.getByText(/loading recommendations/i)).toBeInTheDocument();
  });

  test('GIVEN intents loaded WHEN rendered THEN displays table', async () => {
    render(<DiscoveryDashboard />);

    await waitFor(() => {
      expect(screen.getByTestId('intents-table')).toBeInTheDocument();
    });
  });

  test('GIVEN intents loaded WHEN rendered THEN shows correct count', async () => {
    render(<DiscoveryDashboard />);

    await waitFor(() => {
      expect(screen.getByText(/showing 3 of 3 recommendations/i)).toBeInTheDocument();
    });
  });

  test('GIVEN no intents WHEN rendered THEN shows empty state', async () => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () =>
          Promise.resolve({
            total_intents: 0,
            returned_intents: 0,
            intents: [],
            filters_applied: {},
          }),
      })
    ) as jest.Mock;

    render(<DiscoveryDashboard />);

    await waitFor(() => {
      expect(screen.getByText(/no discovery recommendations found/i)).toBeInTheDocument();
    });
  });
});

// =============================================================================
// ERROR HANDLING TESTS (3 tests)
// =============================================================================

describe('DiscoveryDashboard - Error Handling', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('GIVEN fetch fails WHEN rendered THEN displays error message', async () => {
    global.fetch = jest.fn(() =>
      Promise.reject(new Error('Network error'))
    ) as jest.Mock;

    render(<DiscoveryDashboard />);

    await waitFor(() => {
      expect(screen.getByTestId('error-message')).toBeInTheDocument();
      expect(screen.getByText(/network error/i)).toBeInTheDocument();
    });
  });

  test('GIVEN HTTP error WHEN rendered THEN displays error message', async () => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: false,
        statusText: 'Internal Server Error',
      })
    ) as jest.Mock;

    render(<DiscoveryDashboard />);

    await waitFor(() => {
      expect(screen.getByTestId('error-message')).toBeInTheDocument();
      expect(screen.getByText(/failed to fetch intents/i)).toBeInTheDocument();
    });
  });

  test('GIVEN error state WHEN refresh clicked THEN retries fetch', async () => {
    let callCount = 0;
    global.fetch = jest.fn(() => {
      callCount++;
      if (callCount === 1) {
        return Promise.reject(new Error('Network error'));
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      });
    }) as jest.Mock;

    render(<DiscoveryDashboard />);

    await waitFor(() => {
      expect(screen.getByTestId('error-message')).toBeInTheDocument();
    });

    const refreshButton = screen.getByText(/refresh/i);
    fireEvent.click(refreshButton);

    await waitFor(() => {
      expect(screen.getByTestId('intents-table')).toBeInTheDocument();
    });
  });
});

// =============================================================================
// SORTING TESTS (4 tests)
// =============================================================================

describe('DiscoveryDashboard - Sorting', () => {
  beforeEach(() => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })
    ) as jest.Mock;
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('GIVEN intents table WHEN priority header clicked THEN sorts by priority', async () => {
    render(<DiscoveryDashboard />);

    await waitFor(() => {
      expect(screen.getByTestId('intents-table')).toBeInTheDocument();
    });

    // Initially sorted by priority descending
    const rows = screen.getAllByRole('row');
    expect(rows[1]).toHaveTextContent('John Doe'); // priority 0.9

    // Click priority header to reverse
    const priorityHeader = screen.getByText(/priority/i);
    fireEvent.click(priorityHeader);

    await waitFor(() => {
      const updatedRows = screen.getAllByRole('row');
      expect(updatedRows[1]).toHaveTextContent('Python'); // priority 0.65
    });
  });

  test('GIVEN intents table WHEN entity header clicked THEN sorts alphabetically', async () => {
    render(<DiscoveryDashboard />);

    await waitFor(() => {
      expect(screen.getByTestId('intents-table')).toBeInTheDocument();
    });

    const entityHeader = screen.getByText(/entity/i);
    fireEvent.click(entityHeader);

    await waitFor(() => {
      const rows = screen.getAllByRole('row');
      // Sorted ascending alphabetically: Apple, John, Python
      expect(rows[1]).toHaveTextContent('Apple Inc');
    });
  });

  test('GIVEN sorted by entity WHEN header clicked again THEN reverses sort', async () => {
    render(<DiscoveryDashboard />);

    await waitFor(() => {
      expect(screen.getByTestId('intents-table')).toBeInTheDocument();
    });

    const entityHeader = screen.getByText(/entity/i);
    fireEvent.click(entityHeader); // Sort ascending

    await waitFor(() => {
      const rows = screen.getAllByRole('row');
      expect(rows[1]).toHaveTextContent('Apple Inc');
    });

    fireEvent.click(entityHeader); // Sort descending

    await waitFor(() => {
      const rows = screen.getAllByRole('row');
      expect(rows[1]).toHaveTextContent('Python');
    });
  });

  test('GIVEN intents table WHEN confidence header clicked THEN sorts by confidence', async () => {
    render(<DiscoveryDashboard />);

    await waitFor(() => {
      expect(screen.getByTestId('intents-table')).toBeInTheDocument();
    });

    const confidenceHeader = screen.getByText(/confidence/i);
    fireEvent.click(confidenceHeader);

    await waitFor(() => {
      const rows = screen.getAllByRole('row');
      // Sorted descending by confidence: 0.85, 0.72, 0.68
      expect(rows[1]).toHaveTextContent('John Doe');
    });
  });
});

// =============================================================================
// INTERACTION TESTS (4 tests)
// =============================================================================

describe('DiscoveryDashboard - Interaction', () => {
  beforeEach(() => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })
    ) as jest.Mock;
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('GIVEN intent row WHEN clicked THEN calls onIntentClick', async () => {
    const onIntentClick = jest.fn();

    render(<DiscoveryDashboard onIntentClick={onIntentClick} />);

    await waitFor(() => {
      expect(screen.getByTestId('intents-table')).toBeInTheDocument();
    });

    const firstRow = screen.getByTestId('intent-row-intent-1');
    fireEvent.click(firstRow);

    expect(onIntentClick).toHaveBeenCalledWith(
      expect.objectContaining({
        intent_id: 'intent-1',
        target_entity: 'John Doe',
      })
    );
  });

  test('GIVEN explore button WHEN clicked THEN calls onIntentClick', async () => {
    const onIntentClick = jest.fn();

    render(<DiscoveryDashboard onIntentClick={onIntentClick} />);

    await waitFor(() => {
      expect(screen.getByTestId('intents-table')).toBeInTheDocument();
    });

    const exploreButtons = screen.getAllByText(/explore/i);
    fireEvent.click(exploreButtons[0]);

    expect(onIntentClick).toHaveBeenCalledWith(
      expect.objectContaining({
        intent_id: 'intent-1',
      })
    );
  });

  test('GIVEN no callback WHEN row clicked THEN does not error', async () => {
    render(<DiscoveryDashboard />);

    await waitFor(() => {
      expect(screen.getByTestId('intents-table')).toBeInTheDocument();
    });

    const firstRow = screen.getByTestId('intent-row-intent-1');

    expect(() => fireEvent.click(firstRow)).not.toThrow();
  });

  test('GIVEN refresh button WHEN clicked THEN refetches intents', async () => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })
    ) as jest.Mock;

    render(<DiscoveryDashboard />);

    await waitFor(() => {
      expect(screen.getByTestId('intents-table')).toBeInTheDocument();
    });

    expect(global.fetch).toHaveBeenCalledTimes(1);

    const refreshButton = screen.getByText(/refresh/i);
    fireEvent.click(refreshButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });
  });
});

// =============================================================================
// DISPLAY TESTS (3 tests)
// =============================================================================

describe('DiscoveryDashboard - Display', () => {
  beforeEach(() => {
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })
    ) as jest.Mock;
  });

  afterEach(() => {
    jest.restoreAllMocks();
  });

  test('GIVEN high priority intent WHEN rendered THEN shows priority badge', async () => {
    render(<DiscoveryDashboard />);

    await waitFor(() => {
      const badges = screen.getAllByTestId('priority-badge');
      expect(badges[0]).toHaveTextContent('90%'); // 0.9 * 100
    });
  });

  test('GIVEN intent WHEN rendered THEN shows confidence badge', async () => {
    render(<DiscoveryDashboard />);

    await waitFor(() => {
      const badges = screen.getAllByTestId('confidence-badge');
      expect(badges[0]).toHaveTextContent('85%'); // 0.85 * 100
    });
  });

  test('GIVEN intent WHEN rendered THEN shows rationale', async () => {
    render(<DiscoveryDashboard />);

    await waitFor(() => {
      expect(screen.getByText(/entity appears in only one document/i)).toBeInTheDocument();
    });
  });
});

// =============================================================================
// AUTO-REFRESH TESTS (2 tests)
// =============================================================================

describe('DiscoveryDashboard - Auto-Refresh', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    global.fetch = jest.fn(() =>
      Promise.resolve({
        ok: true,
        json: () => Promise.resolve(mockResponse),
      })
    ) as jest.Mock;
  });

  afterEach(() => {
    jest.useRealTimers();
    jest.restoreAllMocks();
  });

  test('GIVEN autoRefresh enabled WHEN timer elapses THEN refetches', async () => {
    render(<DiscoveryDashboard autoRefresh={true} refreshInterval={5000} />);

    await waitFor(() => {
      expect(screen.getByTestId('intents-table')).toBeInTheDocument();
    });

    expect(global.fetch).toHaveBeenCalledTimes(1);

    // Fast-forward 5 seconds
    jest.advanceTimersByTime(5000);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledTimes(2);
    });
  });

  test('GIVEN autoRefresh disabled WHEN timer elapses THEN does not refetch', async () => {
    render(<DiscoveryDashboard autoRefresh={false} />);

    await waitFor(() => {
      expect(screen.getByTestId('intents-table')).toBeInTheDocument();
    });

    expect(global.fetch).toHaveBeenCalledTimes(1);

    // Fast-forward 30 seconds
    jest.advanceTimersByTime(30000);

    // Should still be 1 call (no auto-refresh)
    expect(global.fetch).toHaveBeenCalledTimes(1);
  });
});
