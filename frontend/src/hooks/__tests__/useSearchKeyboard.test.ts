import { renderHook, act } from '@testing-library/react';
import { useSearchKeyboard } from '../useSearchKeyboard';

/**
 * US-152: useSearchKeyboard GWT Tests
 */

describe('useSearchKeyboard Hook', () => {
  const history = ['search 1', 'search 2', 'search 3'];
  const mockOnSelect = jest.fn();

  beforeEach(() => {
    mockOnSelect.mockClear();
  });

  it('GIVEN a search history WHEN ArrowDown is pressed THEN selectedIndex increments', () => {
    const { result } = renderHook(() => useSearchKeyboard(history, mockOnSelect));

    expect(result.current.selectedIndex).toBe(-1);

    act(() => {
      result.current.handleKeyDown({ key: 'ArrowDown', preventDefault: jest.fn() } as any);
    });
    expect(result.current.selectedIndex).toBe(0);

    act(() => {
      result.current.handleKeyDown({ key: 'ArrowDown', preventDefault: jest.fn() } as any);
    });
    expect(result.current.selectedIndex).toBe(1);
  });

  it('GIVEN a selection WHEN Enter is pressed THEN onSelect is called with the history item', () => {
    const { result } = renderHook(() => useSearchKeyboard(history, mockOnSelect));

    act(() => {
      result.current.handleKeyDown({ key: 'ArrowDown', preventDefault: jest.fn() } as any); // Select index 0
    });

    act(() => {
      result.current.handleKeyDown({ key: 'Enter', preventDefault: jest.fn() } as any);
    });

    expect(mockOnSelect).toHaveBeenCalledWith('search 1');
    expect(result.current.selectedIndex).toBe(-1); // Reset
  });

  it('GIVEN a selection at the top WHEN ArrowUp is pressed THEN it stops at -1', () => {
    const { result } = renderHook(() => useSearchKeyboard(history, mockOnSelect));

    act(() => {
      result.current.handleKeyDown({ key: 'ArrowDown', preventDefault: jest.fn() } as any); // Select index 0
    });
    
    act(() => {
      result.current.handleKeyDown({ key: 'ArrowUp', preventDefault: jest.fn() } as any);
    });
    expect(result.current.selectedIndex).toBe(-1);

    act(() => {
      result.current.handleKeyDown({ key: 'ArrowUp', preventDefault: jest.fn() } as any);
    });
    expect(result.current.selectedIndex).toBe(-1); // Stays at -1
  });
});
