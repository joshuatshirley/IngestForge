"use client";

import React from 'react';
import { Terminal, Pause, Play, Loader2, ChevronRight } from 'lucide-react';
import { AgentJobStatus } from './types';

interface ReasoningChainProps {
  jobId: string;
  status: AgentJobStatus;
  onPause: () => void;
  onResume: () => void;
}

export const ReasoningChain: React.FC<ReasoningChainProps> = ({
  jobId,
  status,
  onPause,
  onResume,
}) => {
  return (
    <div className="forge-card p-8 border-forge-crimson/20 bg-gray-900/40">
      <div className="flex justify-between items-center mb-8">
        <div className="flex items-center gap-4">
          <div className="p-3 rounded-xl bg-gray-800">
            <Terminal size={20} className="text-forge-crimson" />
          </div>
          <div>
            <h3 className="font-bold text-white uppercase tracking-widest text-xs">ReAct Reasoning Chain</h3>
            <p className="text-[10px] text-gray-500 mt-1">Job ID: {jobId.split('-')[0]}</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          {status.status === 'RUNNING' ? (
            <button onClick={onPause} className="p-2 bg-gray-800 hover:bg-forge-crimson/20 text-forge-crimson rounded-lg border border-gray-700 transition-all">
              <Pause size={14} />
            </button>
          ) : status.status === 'PAUSED' ? (
            <button onClick={onResume} className="p-2 bg-gray-800 hover:bg-green-900/20 text-green-500 rounded-lg border border-gray-700 transition-all">
              <Play size={14} />
            </button>
          ) : null}
          <div className="flex items-center gap-2 bg-gray-800 px-4 py-1.5 rounded-full border border-gray-700">
            <div className={`w-2 h-2 rounded-full ${status.status === 'RUNNING' ? 'bg-forge-crimson animate-pulse' : status.status === 'PAUSED' ? 'bg-yellow-500' : 'bg-green-500'}`} />
            <span className="text-[10px] font-black text-white uppercase tracking-tighter">{status.status}</span>
          </div>
        </div>
      </div>

      <div className="space-y-4 max-h-[400px] overflow-y-auto custom-scrollbar pr-4">
        {status.steps.map((step, i) => (
          <div key={i} className="space-y-3 animate-in slide-in-from-left-4 duration-300">
            <div className="flex items-start gap-4">
              <div className="mt-1 w-6 h-6 rounded-full bg-gray-800 flex items-center justify-center text-[10px] font-bold border border-gray-700">{i + 1}</div>
              <div className="flex-1 space-y-2">
                <p className="text-xs text-gray-300 leading-relaxed font-medium bg-gray-800/50 p-4 rounded-2xl border border-gray-800 italic">"{step.thought}"</p>
                {step.action && (
                  <div className="flex items-center gap-2 text-[10px] font-bold text-forge-crimson uppercase tracking-widest ml-2">
                    <ChevronRight size={12} />
                    Action: {step.action}
                  </div>
                )}
              </div>
            </div>
          </div>
        ))}
        {status.status === 'RUNNING' && (
          <div className="flex items-center gap-4 text-gray-500 animate-pulse py-4">
            <Loader2 size={16} className="animate-spin" />
            <span className="text-[10px] font-bold uppercase tracking-widest">Thinking...</span>
          </div>
        )}
      </div>
    </div>
  );
};
