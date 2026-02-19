"use client";

import React from 'react';
import { 
  CheckCircle2, 
  Circle, 
  Clock, 
  AlertCircle, 
  PlayCircle,
  GanttChartSquare
} from 'lucide-react';
import { motion } from 'framer-motion';

interface Task {
  id: string;
  description: str;
  estimated_effort: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed' | 'skipped';
}

interface RoadmapProps {
  objective: string;
  tasks: Task[];
}

export const ResearchRoadmap = ({ objective, tasks }: RoadmapProps) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed': return <CheckCircle2 size={18} className="text-green-500" />;
      case 'in_progress': return <PlayCircle size={18} className="text-forge-crimson animate-pulse" />;
      case 'failed': return <AlertCircle size={18} className="text-red-500" />;
      default: return <Circle size={18} className="text-gray-700" />;
    }
  };

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex items-center gap-3 border-b border-gray-800 pb-4">
        <GanttChartSquare size={20} className="text-forge-crimson" />
        <h3 className="font-bold text-sm uppercase tracking-widest text-white">Research Roadmap</h3>
      </div>

      <div className="relative space-y-0">
        {/* Timeline Line */}
        <div className="absolute left-[9px] top-2 bottom-2 w-0.5 bg-gray-800" />

        {tasks.map((task, i) => (
          <motion.div 
            key={task.id}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: i * 0.1 }}
            className="relative pl-8 pb-8 group"
          >
            {/* Status Dot */}
            <div className="absolute left-0 top-1 bg-forge-navy z-10">
              {getStatusIcon(task.status)}
            </div>

            <div className={`p-4 rounded-2xl border transition-all ${
              task.status === 'in_progress' 
                ? 'bg-forge-crimson/5 border-forge-crimson/30' 
                : 'bg-gray-900/40 border-gray-800 group-hover:border-gray-700'
            }`}>
              <div className="flex justify-between items-start gap-4">
                <div>
                  <p className="text-[10px] font-black text-gray-500 uppercase tracking-tighter mb-1">{task.id}</p>
                  <p className="text-sm font-medium text-gray-200 leading-relaxed">{task.description}</p>
                </div>
                <span className="text-[9px] font-bold text-gray-600 uppercase bg-gray-800 px-2 py-0.5 rounded border border-gray-700 whitespace-nowrap">
                  {task.estimated_effort}
                </span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {tasks.length === 0 && (
        <div className="py-12 text-center opacity-30 italic text-xs">
          Strategizing roadmap...
        </div>
      )}
    </div>
  );
};
