import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { SearchInput } from '../SearchInput';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import searchReducer from '@/store/slices/searchSlice';

/**
 * US-152: SearchInput GWT Tests
 */

const renderWithRedux = (ui: React.ReactElement, initialState = {}) => {
  const store = configureStore({
    reducer: { search: searchReducer },
    preloadedState: { search: { ...initialState } } as any
  });
  return render(<Provider store={store}>{ui}</Provider>);
};

describe('SearchInput Component', () => {
  const mockOnSearch = jest.fn();

  beforeEach(() => {
    mockOnSearch.mockClear();
  });

  it('GIVEN a SearchInput WHEN text is entered and Enter is pressed THEN onSearch is called', () => {
    renderWithRedux(<SearchInput onSearch={mockOnSearch} />);
    
    const input = screen.getByPlaceholderText(/Search across your research/i);
    fireEvent.change(input, { target: { value: 'quantum computing' } });
    fireEvent.keyDown(input, { key: 'Enter', code: 'Enter' });

    expect(mockOnSearch).toHaveBeenCalledWith('quantum computing');
  });

  it('GIVEN search history WHEN the input is focused THEN suggestions are visible', () => {
    const history = ['past search 1', 'past search 2'];
    renderWithRedux(<SearchInput onSearch={mockOnSearch} />, { queryHistory: history });
    
    const input = screen.getByPlaceholderText(/Search across your research/i);
    fireEvent.focus(input);

    expect(screen.getByText(/Recent Searches/i)).toBeInTheDocument();
    expect(screen.getByText('past search 1')).toBeInTheDocument();
  });

  it('GIVEN suggestions WHEN ArrowDown is pressed THEN the first suggestion is highlighted', () => {
    const history = ['item 1', 'item 2'];
    renderWithRedux(<SearchInput onSearch={mockOnSearch} />, { queryHistory: history });
    
    const input = screen.getByPlaceholderText(/Search across your research/i);
    fireEvent.focus(input);
    fireEvent.keyDown(input, { key: 'ArrowDown' });

    // The selection state is internal to useSearchKeyboard, 
    // but the SearchSuggestions component should reflect it.
    const suggestion = screen.getByText('item 1').closest('button');
    expect(suggestion).toHaveClass('bg-gray-800');
  });

  it('GIVEN isSearching is true WHEN rendered THEN a loader is visible', () => {
    renderWithRedux(<SearchInput onSearch={mockOnSearch} isSearching={true} />);
    // Loader2 has data-testid or just check for animate-spin class if using lucide
    // Better to check for the icon component behavior
    expect(document.querySelector('.animate-spin')).toBeInTheDocument();
  });
});
