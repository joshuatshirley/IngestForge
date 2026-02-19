import { renderHook, act } from '@testing-library/react';
import { useDebounce } from '../useDebounce';

/**
 * US-152: useDebounce GWT Tests
 */

describe('useDebounce Hook', () => {
  jest.useFakeTimers();

  it('GIVEN a value WHEN it changes THEN it only updates the debounced value after the delay', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      { initialProps: { value: 'initial', delay: 500 } }
    );

    expect(result.current).toBe('initial');

    // Change value
    rerender({ value: 'updated', delay: 500 });
    expect(result.current).toBe('initial'); // Still initial

    // Fast-forward time
    act(() => {
      jest.advanceTimersByTime(500);
    });

    expect(result.current).toBe('updated');
  });

  it('GIVEN multiple changes WHEN occurring within the delay THEN only the last value is used', () => {
    const { result, rerender } = renderHook(
      ({ value, delay }) => useDebounce(value, delay),
      { initialProps: { value: '1', delay: 500 } }
    );

    rerender({ value: '2', delay: 500 });
    rerender({ value: '3', delay: 500 });

    act(() => {
      jest.advanceTimersByTime(500);
    });

    expect(result.current).toBe('3');
  });
});
