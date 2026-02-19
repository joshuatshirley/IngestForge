"use client";

import React from 'react';
import { 
  PlayIcon, 
  PauseIcon, 
  XIcon, 
  TerminalIcon,
  RefreshIcon
} from '@heroicons/react/outline';
import { useWorkbenchContext } from '../../context/WorkbenchContext';

/**
 * US-1104.1: Foundry Controller Interface
 * Granular control over the background intelligence extraction.
 */

interface FoundryControllerProps {
  streamId?: string;
  status: 'running' | 'paused' | 'failed' | 'idle';
  onPause: () => void;
  onResume: () => void;
  onKill: () => void;
  onBlueprintChange: (blueprintId: string) => void;
}

export const FoundryController: React.FC<FoundryControllerProps> = ({
  streamId,
  status,
  onPause,
  onResume,
  onKill,
  onBlueprintChange
}) => {
  const { currentDocumentId } = useWorkbenchContext();

  return (
    <div className="bg-forge-navy/50 border border-gray-800 rounded-xl p-4 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-gray-400">
          <TerminalIcon className="w-4 h-4" />
          <span className="text-[10px] font-bold uppercase tracking-widest">Pipeline Controller</span>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${status === 'running' ? 'bg-green-500 animate-pulse' : status === 'paused' ? 'bg-yellow-500' : 'bg-gray-600'}`} />
          <span className="text-[10px] text-gray-500 font-mono capitalize">{status}</span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {status === 'running' ? (
          <button 
            onClick={onPause}
            className="flex-1 flex items-center justify-center gap-2 bg-yellow-500/10 hover:bg-yellow-500/20 text-yellow-500 py-2 rounded-lg border border-yellow-500/30 transition-all text-xs font-bold"
          >
            <PauseIcon className="w-4 h-4" />
            Pause
          </button>
        ) : (
          <button 
            onClick={onResume}
            disabled={status === 'idle'}
            className="flex-1 flex items-center justify-center gap-2 bg-green-500/10 hover:bg-green-500/20 text-green-500 py-2 rounded-lg border border-green-500/30 transition-all text-xs font-bold disabled:opacity-30"
          >
            <PlayIcon className="w-4 h-4" />
            Resume
          </button>
        )}
        
        <button 
          onClick={onKill}
          disabled={status === 'idle'}
          className="flex items-center justify-center p-2 bg-red-500/10 hover:bg-red-500/20 text-red-500 rounded-lg border border-red-500/30 transition-all disabled:opacity-30"
          title="Stop Pipeline"
        >
          <XIcon className="w-4 h-4" />
        </button>
      </div>

      <div className="pt-2 border-t border-gray-800">
        <p className="text-[9px] text-gray-600 uppercase font-bold tracking-wider mb-2">Active Blueprint</p>
        <select 
          onChange={(e) => onBlueprintChange(e.target.value)}
          className="w-full bg-black/40 border border-gray-800 rounded-md py-1.5 px-3 text-[11px] text-gray-300 focus:outline-none focus:border-forge-cyan transition-all"
        >
          <option value="legal">Legal Pleading (Deep)</option>
          <option value="cyber">Cyber Threat (Technical)</option>
          <option value="medical">Medical Record (HIPAA)</option>
          <option value="generic">Generic Semantic</option>
        </select>
      </div>

      {streamId && (
        <div className="flex items-center justify-between pt-2">
          <span className="text-[9px] text-gray-600 font-mono">ID: {streamId.slice(0, 8)}...</span>
          <button className="text-[9px] text-forge-cyan hover:underline flex items-center gap-1">
            <RefreshIcon className="w-2.5 h-2.5" />
            Reconnect
          </button>
        </div>
      )}
    </div>
  );
};
