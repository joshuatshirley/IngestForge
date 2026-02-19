import { useState, useEffect } from 'react';

/**
 * US-152: Custom hook for search keyboard navigation.
 * Rule #4: Extracted from SearchInput.
 */
export function useSearchKeyboard(history: string[], onSelect: (query: string) => void) {
  const [selectedIndex, setSelectedIndex] = useState(-1);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setSelectedIndex(prev => Math.min(prev + 1, history.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setSelectedIndex(prev => Math.max(prev - 1, -1));
    } else if (e.key === 'Enter' && selectedIndex >= 0) {
      e.preventDefault();
      onSelect(history[selectedIndex]);
      setSelectedIndex(-1);
    }
  };

  // Reset index when history changes or input cleared
  useEffect(() => {
    setSelectedIndex(-1);
  }, [history]);

  return { selectedIndex, handleKeyDown };
}
