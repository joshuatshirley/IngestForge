"use client";

import React from 'react';
import { KnowledgeGraphCanvas } from '../KnowledgeGraphCanvas';
import { useWorkbenchContext } from '../../../context/WorkbenchContext';

/**
 * US-1401.2.1: Knowledge Mesh Pane
 * Left column of the research workbench.
 */

interface KnowledgeMeshPaneProps {
  isLoading?: boolean;
}

export const KnowledgeMeshPane: React.FC<KnowledgeMeshPaneProps> = ({ isLoading }) => {
  const { activeNodeId, setActiveNode } = useWorkbenchContext();

  return (
    <div className="h-full flex flex-col border-r border-gray-800 bg-black bg-opacity-20 overflow-hidden">
      <div className="p-4 border-b border-gray-800 flex items-center justify-between bg-forge-navy bg-opacity-30">
        <h2 className="text-xs font-bold uppercase tracking-widest text-gray-400">Knowledge Mesh</h2>
        {activeNodeId && (
          <span className="text-[10px] text-forge-cyan font-mono truncate max-w-[100px]">
            Node: {activeNodeId}
          </span>
        )}
      </div>
      <div className="flex-1 relative">
        {isLoading ? (
          <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50 z-10">
            <div className="w-8 h-8 border-4 border-forge-cyan border-t-transparent rounded-full animate-spin" />
          </div>
        ) : null}
        {/* Pass state/setters to canvas if it supports them, or use context inside canvas */}
        <KnowledgeGraphCanvas />
      </div>
    </div>
  );
};
