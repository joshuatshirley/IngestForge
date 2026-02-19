"use client";

import React from 'react';
import { Target, Loader2, Zap, PlayCircle } from 'lucide-react';

interface MissionControlProps {
  task: string;
  setTask: (task: string) => void;
  roadmap: any;
  setRoadmap: (roadmap: any) => void;
  currentJobId: string | null;
  setCurrentJobId: (id: string | null) => void;
  isPlanning: boolean;
  isStarting: boolean;
  jobStatus: any;
  onPlan: (e: React.FormEvent) => void;
  onStart: () => void;
}

export const AgentMissionControl: React.FC<MissionControlProps> = ({
  task,
  setTask,
  roadmap,
  setRoadmap,
  currentJobId,
  setCurrentJobId,
  isPlanning,
  isStarting,
  jobStatus,
  onPlan,
  onStart,
}) => {
  return (
    <div className="forge-card p-8 bg-gradient-to-br from-forge-blue/20 to-transparent">
      <h3 className="font-bold mb-6 flex items-center gap-2 text-sm text-white">
        <Target size={18} className="text-forge-crimson" />
        Mission Control
      </h3>
      <div className="space-y-6">
        <div className="space-y-2">
          <label className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Research Objective</label>
          <textarea 
            value={task}
            onChange={(e) => setTask(e.target.value)}
            disabled={!!roadmap}
            placeholder="e.g. Analyze the implications of quantum decoherence..."
            className="w-full bg-gray-900 border border-gray-800 rounded-2xl p-4 text-sm focus:border-forge-crimson outline-none h-32 transition-all disabled:opacity-50"
          />
        </div>
        
        {!roadmap ? (
          <button 
            onClick={onPlan}
            disabled={isPlanning || !task.trim()}
            className="w-full btn-primary py-4 rounded-2xl flex items-center justify-center gap-2 group shadow-xl shadow-forge-crimson/20 disabled:opacity-50"
          >
            {isPlanning ? <Loader2 size={20} className="animate-spin" /> : <Zap size={18} />}
            <span className="font-bold">Generate Research Plan</span>
          </button>
        ) : (
          <div className="space-y-3 animate-in zoom-in-95">
            <button 
              onClick={onStart}
              disabled={isStarting || (jobStatus?.status === 'RUNNING')}
              className="w-full bg-green-600 hover:bg-green-500 text-white py-4 rounded-2xl flex items-center justify-center gap-2 group shadow-xl shadow-green-900/20 disabled:opacity-50 transition-all"
            >
              <PlayCircle size={18} />
              <span className="font-bold uppercase tracking-widest text-xs">Execute Mission</span>
            </button>
            <button 
              onClick={() => { setRoadmap(null); setCurrentJobId(null); }}
              className="w-full bg-gray-800 hover:bg-gray-700 text-gray-400 py-3 rounded-xl text-[10px] font-bold uppercase tracking-widest transition-all"
            >
              Reset & Edit Objective
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
