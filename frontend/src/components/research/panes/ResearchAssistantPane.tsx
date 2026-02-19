"use client";

import React from 'react';
import { MessageList } from '../../chat/MessageList';
import { ViewGridIcon, ClockIcon } from '@heroicons/react/outline';
import { useWorkbenchContext } from '../../../context/WorkbenchContext';
import { FoundryController } from '../FoundryController';

/**
 * US-1401.2.1: Research Assistant Pane
 * Right column of the research workbench.
 */

interface ResearchAssistantPaneProps {
  conversation: any[];
  isSearching: boolean;
  isChatting: boolean;
  viewMode: 'grid' | 'timeline';
  setViewMode: (mode: 'grid' | 'timeline') => void;
}

export const ResearchAssistantPane: React.FC<ResearchAssistantPaneProps> = ({
  conversation,
  isSearching,
  isChatting,
  viewMode,
  setViewMode,
}) => {
  const { hoveredFactId } = useWorkbenchContext();
  const [pipelineStatus, setPipelineStatus] = React.useState<'running' | 'paused' | 'failed' | 'idle'>('idle');

  return (
    <div className="h-full flex flex-col border-l border-gray-800 bg-black bg-opacity-20 overflow-hidden">
      <div className="p-4 border-b border-gray-800 flex items-center justify-between bg-forge-navy bg-opacity-30">
        <h2 className="text-xs font-bold uppercase tracking-widest text-gray-400">Research Assistant</h2>
        
        <div className="flex items-center gap-3">
          {hoveredFactId && (
            <span className="text-[10px] text-forge-cyan font-mono animate-pulse">
              Fact Linked
            </span>
          )}
          <div className="flex bg-black bg-opacity-40 p-1 rounded-lg border border-gray-800">
            <button 
              onClick={() => setViewMode('grid')}
              className={`p-1.5 rounded-md transition-all ${viewMode === 'grid' ? 'bg-forge-cyan text-black shadow-lg shadow-forge-cyan/20' : 'text-gray-500 hover:text-gray-300'}`}
              title="Card View"
            >
              <ViewGridIcon className="w-4 h-4" />
            </button>
            <button 
              onClick={() => setViewMode('timeline')}
              className={`p-1.5 rounded-md transition-all ${viewMode === 'timeline' ? 'bg-forge-cyan text-black shadow-lg shadow-forge-cyan/20' : 'text-gray-500 hover:text-gray-300'}`}
              title="Timeline View"
            >
              <ClockIcon className="w-4 h-4" />
            </button>
          </div>
        </div>
      </div>

      <div className="p-4 bg-black bg-opacity-10 border-b border-gray-800/50">
        <FoundryController 
          status={pipelineStatus}
          onPause={() => setPipelineStatus('paused')}
          onResume={() => setPipelineStatus('running')}
          onKill={() => setPipelineStatus('idle')}
          onBlueprintChange={(id) => console.log('Blueprint changed:', id)}
        />
      </div>

      <div className="flex-1 overflow-hidden relative">
        <MessageList messages={conversation} isSearching={isSearching || isChatting} />
      </div>
    </div>
  );
};
