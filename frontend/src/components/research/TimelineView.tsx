"use client";

import React from 'react';
import { format } from 'date-fns';
import { 
  ClockIcon, 
  ExternalLinkIcon,
  InformationCircleIcon
} from '@heroicons/react/outline';
import { useWorkbenchContext } from '../../context/WorkbenchContext';

/**
 * US-1103.1: Evidence Timeline UI
 * Chronological fact visualization component.
 * JPL Rule #2: Max 100 events to ensure memory safety.
 */

const MAX_TIMELINE_EVENTS = 100;

interface TimelineEvent {
  timestamp: string;
  event_type: string;
  description: string;
  source: string;
  chunk_ids: string[];
  entity_id?: string;
  severity: string;
  actors: string[];
}

interface TimelineViewProps {
  events: TimelineEvent[];
  onViewSource: (chunkId: string) => void;
  isLoading?: boolean;
}

export const TimelineView: React.FC<TimelineViewProps> = ({ 
  events, 
  onViewSource,
  isLoading 
}) => {
  const { setActiveNode, setCurrentDocument } = useWorkbenchContext();

  if (isLoading) {
    return (
      <div className="flex flex-col gap-4 animate-pulse">
        {[1, 2, 3].map((i) => (
          <div key={i} className="h-24 bg-gray-800 rounded-lg opacity-50" />
        ))}
      </div>
    );
  }

  const handleEventClick = (event: TimelineEvent) => {
    if (event.entity_id) {
      setActiveNode(event.entity_id);
    }
    if (event.source) {
      // Assuming event.source is the document ID or we can derive it
      // For now, if we have a chunk, we might need its document mapping
    }
    if (event.chunk_ids.length > 0) {
      onViewSource(event.chunk_ids[0]);
    }
  };

  if (!events || events.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-gray-500 border-2 border-dashed border-gray-800 rounded-xl">
        <ClockIcon className="w-12 h-12 mb-4 opacity-20" />
        <p className="text-sm">No chronological facts found in current results.</p>
      </div>
    );
  }

  return (
    <div className="relative border-l border-gray-800 ml-4 py-4 space-y-8 animate-in fade-in slide-in-from-left-4 duration-500">
      {events.slice(0, MAX_TIMELINE_EVENTS).map((event, idx) => (
        <div key={`${event.timestamp}-${idx}`} className="relative pl-8">
          {/* Timeline Dot */}
          <div className={`absolute -left-[9px] top-1 w-4 h-4 rounded-full border-4 border-forge-navy ${getSeverityColor(event.severity)}`} />
          
          {/* Content Card */}
          <div 
            onClick={() => handleEventClick(event)}
            className="forge-card hover:border-forge-cyan transition-all cursor-pointer group"
          >
            <div className="flex items-start justify-between mb-2">
              <div className="flex items-center gap-2">
                <span className="text-xs font-mono text-forge-cyan">
                  {format(new Date(event.timestamp), 'yyyy-MM-dd HH:mm:ss')}
                </span>
                <span className="text-[10px] px-2 py-0.5 rounded-full bg-gray-800 text-gray-400 uppercase font-bold tracking-wider">
                  {event.event_type}
                </span>
              </div>
              <div className="flex items-center gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                <ExternalLinkIcon className="w-4 h-4 text-forge-cyan" />
              </div>
            </div>

            <p className="text-sm text-gray-200 mb-3 leading-relaxed">
              {event.description}
            </p>

            <div className="flex flex-wrap items-center gap-4 mt-2 pt-2 border-t border-gray-800 border-opacity-50">
              <div className="flex items-center gap-1.5 text-[11px] text-gray-500">
                <InformationCircleIcon className="w-3.5 h-3.5" />
                <span>Source: {event.source}</span>
              </div>
              
              {event.actors.length > 0 && (
                <div className="flex items-center gap-1.5 text-[11px] text-gray-500">
                  <ClockIcon className="w-3.5 h-3.5" />
                  <span>Actors: {event.actors.join(', ')}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      ))}
    </div>
  );
};

const getSeverityColor = (severity: string): string => {
  switch (severity.toLowerCase()) {
    case 'critical': return 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]';
    case 'high': return 'bg-orange-500 shadow-[0_0_10px_rgba(249,115,22,0.5)]';
    case 'medium': return 'bg-yellow-500';
    case 'low': return 'bg-blue-500';
    default: return 'bg-gray-500';
  }
};

const getSeverityColor = (severity: string): string => {
  switch (severity.toLowerCase()) {
    case 'critical': return 'bg-red-500 shadow-[0_0_10px_rgba(239,68,68,0.5)]';
    case 'high': return 'bg-orange-500 shadow-[0_0_10px_rgba(249,115,22,0.5)]';
    case 'medium': return 'bg-yellow-500';
    case 'low': return 'bg-blue-500';
    default: return 'bg-gray-500';
  }
};
