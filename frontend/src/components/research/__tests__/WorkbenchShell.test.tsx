import React from 'react';
import { render, screen } from '@testing-library/react';
import { WorkbenchShell } from '../WorkbenchShell';

/**
 * US-1401.2.1: Workbench Shell Tests
 * Verifies the 3-pane layout and responsiveness requirements.
 */

describe('WorkbenchShell Layout', () => {
  const mockProps = {
    viewMode: 'grid' as const,
    setViewMode: jest.fn(),
    conversation: [],
    isSearching: false,
    isChatting: false,
    currentResults: [],
    timelineData: null,
    isLoadingTimeline: false,
    query: '',
    onViewSource: jest.fn(),
  };

  it('GIVEN a large viewport WHEN the WorkbenchShell is rendered THEN all three panes are initialized', () => {
    render(<WorkbenchShell {...mockProps} />);
    
    // Check for pane headers
    expect(screen.getByText(/Knowledge Mesh/i)).toBeInTheDocument();
    expect(screen.getByText(/Evidence & Source/i)).toBeInTheDocument();
    expect(screen.getByText(/Research Assistant/i)).toBeInTheDocument();
  });

  it('GIVEN a workbench WHEN viewMode is timeline THEN the DocumentViewerPane renders the timeline content', () => {
    const props = { 
      ...mockProps, 
      viewMode: 'timeline' as const,
      timelineData: { events: [{ timestamp: '2026-02-18', description: 'Test Event' }] } 
    };
    
    render(<WorkbenchShell {...props} />);
    
    // Grid results should be gone
    expect(screen.queryByText(/No results to display/i)).not.toBeInTheDocument();
    // Timeline marker should be present
    expect(screen.getByText(/Test Event/i)).toBeInTheDocument();
  });

  it('GIVEN a workbench WHEN results are provided THEN the DocumentViewerPane renders search cards', () => {
    const props = { 
      ...mockProps, 
      currentResults: [{ 
        id: '1', 
        content: 'Search Result', 
        score: 0.9, 
        metadata: { source: 'doc.pdf' } 
      }] as any 
    };
    
    render(<WorkbenchShell {...props} />);
    
    expect(screen.getByText(/Search Result/i)).toBeInTheDocument();
  });
});
