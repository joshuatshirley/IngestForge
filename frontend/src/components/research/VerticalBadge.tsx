"use client";

import React from 'react';
import { ShieldCheckIcon, AdjustmentsIcon } from '@heroicons/react/outline';
import { useWorkbenchContext } from '../../context/WorkbenchContext';

/**
 * US-1501.4: Domain Detection Badge
 * Displays detected vertical and confidence score.
 */

interface VerticalBadgeProps {
  domain?: string;
  confidence?: number;
}

export const VerticalBadge: React.FC<VerticalBadgeProps> = ({ 
  domain = 'Generic', 
  confidence = 0.95 
}) => {
  const { currentDocumentId } = useWorkbenchContext();

  if (!currentDocumentId) return null;

  return (
    <div className="flex items-center gap-2 px-3 py-1.5 bg-forge-navy/50 border border-gray-800 rounded-full animate-in fade-in zoom-in-95 duration-300">
      <ShieldCheckIcon className={`w-4 h-4 ${confidence > 0.9 ? 'text-green-500' : 'text-yellow-500'}`} />
      <span className="text-[10px] font-bold uppercase tracking-widest text-gray-300">
        {domain}
      </span>
      <div className="w-[1px] h-3 bg-gray-800 mx-1" />
      <span className="text-[10px] font-mono text-gray-500">
        {(confidence * 100).toFixed(0)}% Match
      </span>
      <button className="ml-2 p-1 hover:bg-gray-800 rounded text-gray-500 hover:text-white transition-colors" title="Change Domain">
        <AdjustmentsIcon className="w-3 h-3" />
      </button>
    </div>
  );
};
