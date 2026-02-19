"use client";

import React, { createContext, useContext, useState, ReactNode, useCallback, useEffect } from 'react';

/**
 * US-1401.2.2: Workbench Context for Pane Synchronization.
 * Tracks shared state across Mesh, PDF, and Assistant panes.
 * Rule #9: Complete type hints.
 */

export interface WorkbenchState {
  activeNodeId: string | null;
  currentDocumentId: string | null;
  activePage: number | null;
  hoveredFactId: string | null;
  zoomLevel: number;
  selectedPeerIds: string[]; // Task 272: Federated peer selection
}

interface WorkbenchContextType extends WorkbenchState {
  setActiveNode: (id: string | null) => void;
  setCurrentDocument: (id: string | null) => void;
  setActivePage: (page: number | null) => void;
  setHoveredFact: (id: string | null) => void;
  setZoom: (zoom: number) => void;
  setSelectedPeerIds: (ids: string[]) => void;
  resetWorkbench: () => void;
}

const WorkbenchContext = createContext<WorkbenchContextType | undefined>(undefined);

const STORAGE_KEY = 'if_workbench_session';

export const useWorkbenchContext = () => {
  const context = useContext(WorkbenchContext);
  if (!context) {
    throw new Error('useWorkbenchContext must be used within a WorkbenchProvider');
  }
  return context;
};

interface WorkbenchProviderProps {
  children: ReactNode;
  initialState?: Partial<WorkbenchState>;
}

export const WorkbenchProvider: React.FC<WorkbenchProviderProps> = ({ children, initialState }) => {
  // US-1401.2.3: Load session from localStorage (Task 103)
  const [activeNodeId, setActiveNodeId] = useState<string | null>(initialState?.activeNodeId || null);
  const [currentDocumentId, setCurrentDocumentId] = useState<string | null>(initialState?.currentDocumentId || null);
  const [activePage, setActivePage] = useState<number | null>(initialState?.activePage || null);
  const [hoveredFactId, setHoveredFactId] = useState<string | null>(initialState?.hoveredFactId || null);
  const [zoomLevel, setZoomLevel] = useState<number>(initialState?.zoomLevel || 1.0);
  const [selectedPeerIds, setSelectedPeerIdsState] = useState<string[]>(initialState?.selectedPeerIds || []);

  // Initialize from localStorage on mount
  useEffect(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        if (parsed.currentDocumentId) setCurrentDocumentId(parsed.currentDocumentId);
        if (parsed.zoomLevel) setZoomLevel(parsed.zoomLevel);
        if (parsed.selectedPeerIds) setSelectedPeerIdsState(parsed.selectedPeerIds);
      } catch (e) {
        console.error('Failed to load workbench session:', e);
      }
    }
  }, []);

  // Persist to localStorage on change
  useEffect(() => {
    const session = { currentDocumentId, zoomLevel, selectedPeerIds };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(session));
  }, [currentDocumentId, zoomLevel, selectedPeerIds]);

  const setActiveNode = useCallback((id: string | null) => setActiveNodeId(id), []);
  const setCurrentDocument = useCallback((id: string | null) => setCurrentDocumentId(id), []);
  const setPage = useCallback((page: number | null) => setActivePage(page), []);
  const setHoveredFact = useCallback((id: string | null) => setHoveredFactId(id), []);
  const setZoom = useCallback((zoom: number) => setZoomLevel(zoom), []);
  const setSelectedPeerIds = useCallback((ids: string[]) => setSelectedPeerIdsState(ids), []);

  const resetWorkbench = useCallback(() => {
    setActiveNodeId(null);
    setCurrentDocumentId(null);
    setActivePage(null);
    setHoveredFactId(null);
    setZoomLevel(1.0);
    setSelectedPeerIdsState([]);
  }, []);

  const value: WorkbenchContextType = {
    activeNodeId,
    currentDocumentId,
    activePage,
    hoveredFactId,
    zoomLevel,
    selectedPeerIds,
    setActiveNode,
    setCurrentDocument,
    setActivePage: setPage,
    setHoveredFact,
    setZoom,
    setSelectedPeerIds,
    resetWorkbench,
  };

  return (
    <WorkbenchContext.Provider value={value}>
      {children}
    </WorkbenchContext.Provider>
  );
};
