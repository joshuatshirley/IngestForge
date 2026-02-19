import React from 'react';
import { render, screen } from '@testing-library/react';
import { SearchResultCard } from '../SearchResultCard';
import { WorkbenchProvider } from '@/context/WorkbenchContext';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import { ingestforgeApi } from '@/store/api/ingestforgeApi';

/**
 * US-152: SearchResultCard GWT Tests
 */

const mockStore = configureStore({
  reducer: { [ingestforgeApi.reducerPath]: ingestforgeApi.reducer },
  middleware: (getDefault) => getDefault().concat(ingestforgeApi.middleware)
});

describe('SearchResultCard Component', () => {
  const mockProps = {
    id: 'res-1',
    content: 'This is a test research snippet containing Einstein.',
    score: 0.95,
    source: 'Relativity.pdf',
    section: 'Introduction',
    page: 5,
    entities: [{ type: 'person', text: 'Einstein' }],
    query: 'Einstein',
    onViewSource: jest.fn(),
  };

  it('GIVEN a SearchResultCard WHEN rendered THEN it displays content and metadata', () => {
    render(
      <Provider store={mockStore}>
        <WorkbenchProvider>
          <SearchResultCard {...mockProps} />
        </WorkbenchProvider>
      </Provider>
    );

    expect(screen.getByText('Relativity.pdf')).toBeInTheDocument();
    expect(screen.getByText('PAGE 5')).toBeInTheDocument();
    expect(screen.getByText(/95.0%/)).toBeInTheDocument();
    expect(screen.getByText('Einstein')).toBeInTheDocument();
  });

  it('GIVEN an active node in context WHEN it matches the card THEN it highlights', () => {
    render(
      <Provider store={mockStore}>
        <WorkbenchProvider initialState={{ activeNodeId: 'res-1' }}>
          <SearchResultCard {...mockProps} />
        </WorkbenchProvider>
      </Provider>
    );

    // Check for active border class
    const card = screen.getByText('Relativity.pdf').closest('.forge-card');
    expect(card).toHaveClass('border-forge-cyan');
  });

  it('GIVEN an active entity node in context WHEN it matches a badge THEN it highlights the badge', () => {
    render(
      <Provider store={mockStore}>
        <WorkbenchProvider initialState={{ activeNodeId: 'Einstein' }}>
          <SearchResultCard {...mockProps} />
        </WorkbenchProvider>
      </Provider>
    );

    const badge = screen.getByText('Einstein').closest('div');
    expect(badge).toHaveClass('bg-forge-cyan');
  });
});
