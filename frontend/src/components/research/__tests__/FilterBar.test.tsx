import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { FilterBar } from '../FilterBar';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import searchReducer from '@/store/slices/searchSlice';

/**
 * US-152: FilterBar GWT Tests
 */

const renderWithRedux = (initialFilters = {}) => {
  const store = configureStore({
    reducer: { search: searchReducer },
    preloadedState: { 
      search: { 
        filters: { docTypes: [], sources: [], entityTypes: [], ...initialFilters } 
      } 
    } as any
  });
  return {
    ...render(<Provider store={store}><FilterBar /></Provider>),
    store
  };
};

describe('FilterBar Component', () => {
  it('GIVEN a FilterBar WHEN a document type is clicked THEN the filter is toggled', () => {
    const { store } = renderWithRedux();
    
    const pdfButton = screen.getByText('PDF');
    fireEvent.click(pdfButton);

    expect(store.getState().search.filters.docTypes).toContain('PDF');
    expect(pdfButton).toHaveClass('bg-forge-crimson');
  });

  it('GIVEN active filters WHEN "Clear All" is clicked THEN filters are reset', () => {
    const { store } = renderWithRedux({ docTypes: ['PDF'], entityTypes: ['PERSON'] });
    
    const clearButton = screen.getByText(/Clear All/i);
    fireEvent.click(clearButton);

    expect(store.getState().search.filters.docTypes).toHaveLength(0);
    expect(store.getState().search.filters.entityTypes).toHaveLength(0);
  });

  it('GIVEN entity type filters WHEN rendered THEN they are visible with correct colors', () => {
    renderWithRedux();
    const personButton = screen.getByText('PERSON');
    expect(personButton).toBeInTheDocument();
    
    fireEvent.click(personButton);
    expect(personButton).toHaveClass('bg-forge-cyan');
  });
});
