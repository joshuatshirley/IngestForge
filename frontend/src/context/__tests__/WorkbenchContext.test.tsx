import React from 'react';
import { render, screen, act } from '@testing-library/react';
import { WorkbenchProvider, useWorkbenchContext } from '../WorkbenchContext';

/**
 * US-1401.2.2: Workbench Context Tests
 * Verifies global state synchronization across modular panes.
 */

const TestConsumer = () => {
  const { 
    activeNodeId, 
    currentDocumentId, 
    setActiveNode, 
    setCurrentDocument,
    resetWorkbench 
  } = useWorkbenchContext();

  return (
    <div>
      <div data-testid="node-id">{activeNodeId || 'null'}</div>
      <div data-testid="doc-id">{currentDocumentId || 'null'}</div>
      <button onClick={() => setActiveNode('node-123')}>Set Node</button>
      <button onClick={() => setCurrentDocument('doc-456')}>Set Doc</button>
      <button onClick={resetWorkbench}>Reset</button>
    </div>
  );
};

describe('WorkbenchContext', () => {
  it('GIVEN a WorkbenchProvider WHEN initialized THEN it provides null defaults', () => {
    render(
      <WorkbenchProvider>
        <TestConsumer />
      </WorkbenchProvider>
    );

    expect(screen.getByTestId('node-id')).toHaveTextContent('null');
    expect(screen.getByTestId('doc-id')).toHaveTextContent('null');
  });

  it('GIVEN a workbench WHEN setters are called THEN the state is updated globally', () => {
    render(
      <WorkbenchProvider>
        <TestConsumer />
      </WorkbenchProvider>
    );

    act(() => {
      screen.getByText('Set Node').click();
    });
    expect(screen.getByTestId('node-id')).toHaveTextContent('node-123');

    act(() => {
      screen.getByText('Set Doc').click();
    });
    expect(screen.getByTestId('doc-id')).toHaveTextContent('doc-456');
  });

  it('GIVEN an active workbench state WHEN reset is called THEN all IDs return to null', () => {
    render(
      <WorkbenchProvider initialState={{ activeNodeId: 'active' }}>
        <TestConsumer />
      </WorkbenchProvider>
    );

    expect(screen.getByTestId('node-id')).toHaveTextContent('active');

    act(() => {
      screen.getByText('Reset').click();
    });

    expect(screen.getByTestId('node-id')).toHaveTextContent('null');
  });

  it('GIVEN a consumer outside of a provider WHEN rendered THEN it throws an error', () => {
    // Suppress console error for expected throw
    const spy = jest.spyOn(console, 'error').mockImplementation(() => {});
    
    expect(() => render(<TestConsumer />)).toThrow('useWorkbenchContext must be used within a WorkbenchProvider');
    
    spy.mockRestore();
  });
});
