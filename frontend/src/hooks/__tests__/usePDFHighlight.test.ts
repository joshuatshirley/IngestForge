/**
 * Unit tests for usePDFHighlight hook - US-1404.1
 *
 * Tests highlight state management and rendering logic.
 *
 * JPL Power of Ten Compliance:
 * - Rule #4: All test functions < 60 lines
 * - Rule #9: Complete type hints
 *
 * Epic: EP-14 (Foundry UI)
 * Feature: FE-11-01 (Visualization)
 * Test Date: 2026-02-18
 */

import { renderHook, act } from '@testing-library/react';
import { usePDFHighlight, getConfidenceColor, groupHighlightsByPage, createHighlightFromEntity, calculateHighlightArea } from '../usePDFHighlight';
import { BoundingBox } from '../../utils/pdfCoordinates';

// =============================================================================
// TEST FIXTURES
// =============================================================================

const mockBBox: BoundingBox = {
  x1: 0.1,
  y1: 0.2,
  x2: 0.3,
  y2: 0.25,
};

const mockContainerRef = {
  current: document.createElement('div'),
};

// =============================================================================
// ADD HIGHLIGHT TESTS (5 tests)
// =============================================================================

describe('usePDFHighlight - Add Highlight', () => {
  test('GIVEN valid bbox and page WHEN addHighlight called THEN adds highlight to state', () => {
    const { result } = renderHook(() => usePDFHighlight(mockContainerRef));

    act(() => {
      const id = result.current.addHighlight(mockBBox, 1);
      expect(id).toBeTruthy();
    });

    expect(result.current.highlights).toHaveLength(1);
    expect(result.current.highlights[0].bbox).toEqual(mockBBox);
    expect(result.current.highlights[0].page).toBe(1);
  });

  test('GIVEN custom color WHEN addHighlight called with options THEN uses custom color', () => {
    const { result } = renderHook(() => usePDFHighlight(mockContainerRef));

    act(() => {
      result.current.addHighlight(mockBBox, 1, { color: '#FF0000' });
    });

    expect(result.current.highlights[0].color).toBe('#FF0000');
  });

  test('GIVEN custom opacity WHEN addHighlight called with options THEN uses custom opacity', () => {
    const { result } = renderHook(() => usePDFHighlight(mockContainerRef));

    act(() => {
      result.current.addHighlight(mockBBox, 1, { opacity: 0.5 });
    });

    expect(result.current.highlights[0].opacity).toBe(0.5);
  });

  test('GIVEN default options WHEN addHighlight called THEN uses default color and opacity', () => {
    const { result } = renderHook(() => usePDFHighlight(mockContainerRef));

    act(() => {
      result.current.addHighlight(mockBBox, 1);
    });

    expect(result.current.highlights[0].color).toBe('#FFEB3B'); // Yellow
    expect(result.current.highlights[0].opacity).toBe(0.3);
  });

  test('GIVEN 100 highlights already exist WHEN addHighlight called THEN does not add more', () => {
    const { result } = renderHook(() => usePDFHighlight(mockContainerRef));

    // Add 100 highlights
    act(() => {
      for (let i = 0; i < 100; i++) {
        result.current.addHighlight(mockBBox, 1);
      }
    });

    expect(result.current.highlights).toHaveLength(100);

    // Try to add 101st highlight
    act(() => {
      const id = result.current.addHighlight(mockBBox, 1);
      expect(id).toBe(''); // Should return empty string when max reached
    });

    expect(result.current.highlights).toHaveLength(100); // Still 100
  });
});

// =============================================================================
// REMOVE HIGHLIGHT TESTS (3 tests)
// =============================================================================

describe('usePDFHighlight - Remove Highlight', () => {
  test('GIVEN existing highlight WHEN removeHighlight called THEN removes highlight', () => {
    const { result } = renderHook(() => usePDFHighlight(mockContainerRef));

    let highlightId: string = '';

    act(() => {
      highlightId = result.current.addHighlight(mockBBox, 1);
    });

    expect(result.current.highlights).toHaveLength(1);

    act(() => {
      result.current.removeHighlight(highlightId);
    });

    expect(result.current.highlights).toHaveLength(0);
  });

  test('GIVEN multiple highlights WHEN removeHighlight called THEN removes only specified highlight', () => {
    const { result } = renderHook(() => usePDFHighlight(mockContainerRef));

    let id1: string = '';
    let id2: string = '';

    act(() => {
      id1 = result.current.addHighlight(mockBBox, 1);
      id2 = result.current.addHighlight(mockBBox, 2);
    });

    expect(result.current.highlights).toHaveLength(2);

    act(() => {
      result.current.removeHighlight(id1);
    });

    expect(result.current.highlights).toHaveLength(1);
    expect(result.current.highlights[0].id).toBe(id2);
  });

  test('GIVEN non-existent ID WHEN removeHighlight called THEN does nothing', () => {
    const { result } = renderHook(() => usePDFHighlight(mockContainerRef));

    act(() => {
      result.current.addHighlight(mockBBox, 1);
    });

    expect(result.current.highlights).toHaveLength(1);

    act(() => {
      result.current.removeHighlight('non-existent-id');
    });

    expect(result.current.highlights).toHaveLength(1);
  });
});

// =============================================================================
// CLEAR HIGHLIGHTS TESTS (2 tests)
// =============================================================================

describe('usePDFHighlight - Clear Highlights', () => {
  test('GIVEN multiple highlights WHEN clearHighlights called THEN removes all highlights', () => {
    const { result } = renderHook(() => usePDFHighlight(mockContainerRef));

    act(() => {
      result.current.addHighlight(mockBBox, 1);
      result.current.addHighlight(mockBBox, 2);
      result.current.addHighlight(mockBBox, 3);
    });

    expect(result.current.highlights).toHaveLength(3);

    act(() => {
      result.current.clearHighlights();
    });

    expect(result.current.highlights).toHaveLength(0);
  });

  test('GIVEN no highlights WHEN clearHighlights called THEN does not error', () => {
    const { result } = renderHook(() => usePDFHighlight(mockContainerRef));

    expect(result.current.highlights).toHaveLength(0);

    expect(() => {
      act(() => {
        result.current.clearHighlights();
      });
    }).not.toThrow();
  });
});

// =============================================================================
// SCROLL TO HIGHLIGHT TESTS (2 tests)
// =============================================================================

describe('usePDFHighlight - Scroll To Highlight', () => {
  test('GIVEN existing highlight WHEN scrollToHighlight called THEN scrolls container', async () => {
    const mockScrollTo = jest.fn();
    mockContainerRef.current.scrollTo = mockScrollTo;

    const { result } = renderHook(() => usePDFHighlight(mockContainerRef));

    let highlightId: string = '';

    act(() => {
      highlightId = result.current.addHighlight(mockBBox, 1);
    });

    await act(async () => {
      await result.current.scrollToHighlight(highlightId);
    });

    // scrollTo should be called
    expect(mockScrollTo).toHaveBeenCalled();
  });

  test('GIVEN non-existent highlight ID WHEN scrollToHighlight called THEN does nothing', async () => {
    const mockScrollTo = jest.fn();
    mockContainerRef.current.scrollTo = mockScrollTo;

    const { result } = renderHook(() => usePDFHighlight(mockContainerRef));

    await act(async () => {
      await result.current.scrollToHighlight('non-existent');
    });

    expect(mockScrollTo).not.toHaveBeenCalled();
  });
});

// =============================================================================
// CONFIDENCE COLOR TESTS (4 tests)
// =============================================================================

describe('usePDFHighlight - Confidence Colors', () => {
  test('GIVEN confidence >= 0.9 WHEN getConfidenceColor called THEN returns green', () => {
    expect(getConfidenceColor(0.95)).toBe('#4CAF50');
    expect(getConfidenceColor(0.9)).toBe('#4CAF50');
  });

  test('GIVEN confidence 0.7-0.9 WHEN getConfidenceColor called THEN returns yellow', () => {
    expect(getConfidenceColor(0.85)).toBe('#FFEB3B');
    expect(getConfidenceColor(0.7)).toBe('#FFEB3B');
  });

  test('GIVEN confidence < 0.7 WHEN getConfidenceColor called THEN returns red', () => {
    expect(getConfidenceColor(0.65)).toBe('#F44336');
    expect(getConfidenceColor(0.5)).toBe('#F44336');
  });

  test('GIVEN boundary values WHEN getConfidenceColor called THEN returns correct colors', () => {
    expect(getConfidenceColor(1.0)).toBe('#4CAF50'); // Max
    expect(getConfidenceColor(0.0)).toBe('#F44336'); // Min
  });
});

// =============================================================================
// GROUP HIGHLIGHTS TESTS (4 tests)
// =============================================================================

describe('usePDFHighlight - Group Highlights', () => {
  test('GIVEN highlights on different pages WHEN groupHighlightsByPage called THEN groups by page', () => {
    const highlights = [
      { id: '1', bbox: mockBBox, page: 1, color: '#000', opacity: 0.3 },
      { id: '2', bbox: mockBBox, page: 2, color: '#000', opacity: 0.3 },
      { id: '3', bbox: mockBBox, page: 1, color: '#000', opacity: 0.3 },
    ];

    const grouped = groupHighlightsByPage(highlights);

    expect(grouped.size).toBe(2);
    expect(grouped.get(1)).toHaveLength(2);
    expect(grouped.get(2)).toHaveLength(1);
  });

  test('GIVEN all highlights on same page WHEN groupHighlightsByPage called THEN creates single group', () => {
    const highlights = [
      { id: '1', bbox: mockBBox, page: 5, color: '#000', opacity: 0.3 },
      { id: '2', bbox: mockBBox, page: 5, color: '#000', opacity: 0.3 },
    ];

    const grouped = groupHighlightsByPage(highlights);

    expect(grouped.size).toBe(1);
    expect(grouped.get(5)).toHaveLength(2);
  });

  test('GIVEN no highlights WHEN groupHighlightsByPage called THEN returns empty map', () => {
    const grouped = groupHighlightsByPage([]);

    expect(grouped.size).toBe(0);
  });

  test('GIVEN highlights with various page numbers WHEN groupHighlightsByPage called THEN groups correctly', () => {
    const highlights = [
      { id: '1', bbox: mockBBox, page: 1, color: '#000', opacity: 0.3 },
      { id: '2', bbox: mockBBox, page: 3, color: '#000', opacity: 0.3 },
      { id: '3', bbox: mockBBox, page: 2, color: '#000', opacity: 0.3 },
      { id: '4', bbox: mockBBox, page: 1, color: '#000', opacity: 0.3 },
    ];

    const grouped = groupHighlightsByPage(highlights);

    expect(grouped.size).toBe(3);
    expect(grouped.get(1)?.map(h => h.id)).toEqual(['1', '4']);
    expect(grouped.get(2)?.map(h => h.id)).toEqual(['3']);
    expect(grouped.get(3)?.map(h => h.id)).toEqual(['2']);
  });
});

// =============================================================================
// CREATE FROM ENTITY TESTS (3 tests)
// =============================================================================

describe('usePDFHighlight - Create From Entity', () => {
  test('GIVEN entity data WHEN createHighlightFromEntity called THEN creates highlight with entity info', () => {
    const highlight = createHighlightFromEntity(
      'entity-123',
      mockBBox,
      5,
      0.95
    );

    expect(highlight.entityId).toBe('entity-123');
    expect(highlight.bbox).toEqual(mockBBox);
    expect(highlight.page).toBe(5);
    expect(highlight.confidence).toBe(0.95);
  });

  test('GIVEN high confidence WHEN createHighlightFromEntity called THEN uses green color', () => {
    const highlight = createHighlightFromEntity(
      'entity-123',
      mockBBox,
      1,
      0.95
    );

    expect(highlight.color).toBe('#4CAF50'); // Green
  });

  test('GIVEN low confidence WHEN createHighlightFromEntity called THEN uses red color', () => {
    const highlight = createHighlightFromEntity(
      'entity-123',
      mockBBox,
      1,
      0.6
    );

    expect(highlight.color).toBe('#F44336'); // Red
  });
});

// =============================================================================
// CALCULATE AREA TESTS (3 tests)
// =============================================================================

describe('usePDFHighlight - Calculate Area', () => {
  test('GIVEN single highlight WHEN calculateHighlightArea called THEN returns correct area', () => {
    const highlights = [
      {
        id: '1',
        bbox: { x1: 0.0, y1: 0.0, x2: 0.5, y2: 0.5 }, // 0.25 area
        page: 1,
        color: '#000',
        opacity: 0.3,
      },
    ];

    const area = calculateHighlightArea(highlights);

    expect(area).toBe(0.25);
  });

  test('GIVEN multiple highlights WHEN calculateHighlightArea called THEN sums areas', () => {
    const highlights = [
      {
        id: '1',
        bbox: { x1: 0.0, y1: 0.0, x2: 0.5, y2: 0.5 }, // 0.25 area
        page: 1,
        color: '#000',
        opacity: 0.3,
      },
      {
        id: '2',
        bbox: { x1: 0.0, y1: 0.0, x2: 0.2, y2: 0.2 }, // 0.04 area
        page: 1,
        color: '#000',
        opacity: 0.3,
      },
    ];

    const area = calculateHighlightArea(highlights);

    expect(area).toBeCloseTo(0.29, 2);
  });

  test('GIVEN no highlights WHEN calculateHighlightArea called THEN returns 0', () => {
    const area = calculateHighlightArea([]);

    expect(area).toBe(0);
  });
});
