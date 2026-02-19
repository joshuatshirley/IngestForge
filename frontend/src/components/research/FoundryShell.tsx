"use client";

import React, { useState, useRef, useCallback } from 'react';
import { KnowledgeMeshPane } from './panes/KnowledgeMeshPane';
import { DocumentViewerPane } from './panes/DocumentViewerPane';
import { ResearchAssistantPane } from './panes/ResearchAssistantPane';
import { WorkbenchState } from '../../context/WorkbenchContext';
import { SourcedResult } from '../chat/types';

/**
 * US-1401.2.2: Foundry Shell
 * Orchestrates the 3-pane workbench with Global Context.
 * Task 101: 3-pane responsive grid shell.
 * Task 102: Draggable dividers.
 * Rule #4: Component logic < 60 lines.
 */

interface FoundryShellProps {
  viewMode: 'grid' | 'timeline';
  setViewMode: (mode: 'grid' | 'timeline') => void;
  conversation: any[];
  isSearching: boolean;
  isChatting: boolean;
  currentResults: SourcedResult[];
  timelineData: any;
  isLoadingTimeline: boolean;
  query: string;
  onViewSource: (id: string) => void;
  initialState?: Partial<WorkbenchState>;
}

export const FoundryShell: React.FC<FoundryShellProps> = (props) => {
  const [leftWidth, setLeftWidth] = useState(25); // Percent
  const [rightWidth, setRightWidth] = useState(30); // Percent
  const isResizing = useRef<'left' | 'right' | null>(null);

  const handleMouseDown = (side: 'left' | 'right') => {
    isResizing.current = side;
    document.addEventListener('mousemove', handleMouseMove);
    document.addEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'col-resize';
  };

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing.current) return;
    
    // JPL Rule #2: Bound coordinate calculations
    const percentage = Math.max(0, Math.min(100, (e.clientX / window.innerWidth) * 100));
    
    if (isResizing.current === 'left') {
      if (percentage > 15 && percentage < 40) setLeftWidth(percentage);
    } else {
      const rightPercentage = 100 - percentage;
      if (rightPercentage > 20 && rightPercentage < 45) setRightWidth(rightPercentage);
    }
  }, []);

  const handleMouseUp = () => {
    isResizing.current = null;
    document.removeEventListener('mousemove', handleMouseMove);
    document.removeEventListener('mouseup', handleMouseUp);
    document.body.style.cursor = 'default';
  };

  return (
    <div 
      className="flex-1 min-h-0 grid gap-0 overflow-hidden forge-card !p-0 border-gray-800 shadow-2xl transition-all duration-75"
      style={{ gridTemplateColumns: `${leftWidth}% 1px 1fr 1px ${rightWidth}%` }}
    >
      <KnowledgeMeshPane />
      <div onMouseDown={() => handleMouseDown('left')} className="w-[1px] h-full bg-gray-800 hover:bg-forge-cyan cursor-col-resize z-10 transition-colors" />
      <DocumentViewerPane {...props} />
      <div onMouseDown={() => handleMouseDown('right')} className="w-[1px] h-full bg-gray-800 hover:bg-forge-cyan cursor-col-resize z-10 transition-colors" />
      <ResearchAssistantPane {...props} />
    </div>
  );
};
