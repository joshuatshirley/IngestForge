/**
 * US-130: Federated Error Indicator
 * Displays warnings for partial search results.
 * JPL Rule #4: Strictly < 60 lines.
 */

import React from 'react';
import { ExclamationIcon } from '@heroicons/react/solid';
import { PeerFailure } from '@/types/search';

interface NexusErrorIndicatorProps {
  failures: PeerFailure[];
}

export const NexusErrorIndicator: React.FC<NexusErrorIndicatorProps> = ({ failures }) => {
  if (failures.length === 0) return null;

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-amber-900/20 border border-amber-500/30 rounded-lg animate-in fade-in duration-300">
      <ExclamationIcon className="w-5 h-5 text-amber-500" />
      <div className="text-sm">
        <span className="font-semibold text-amber-400">Partial Results:</span>
        <span className="ml-1 text-amber-200/80">
          {failures.length} peer{failures.length > 1 ? 's' : ''} failed to respond.
        </span>
      </div>
      
      <button 
        className="ml-auto text-xs font-mono text-amber-500 hover:underline uppercase tracking-wider"
        title={failures.map(f => `${f.nexus_id}: ${f.message}`).join('
')}
      >
        Details
      </button>
    </div>
  );
};
