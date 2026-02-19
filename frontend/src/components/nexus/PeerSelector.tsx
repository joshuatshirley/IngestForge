'use client';

import React, { useState, useEffect } from 'react';
import { nexusService, NexusPeer, NexusStatus } from '@/services/nexusService';
import { useWorkbenchContext } from '@/context/WorkbenchContext';
import { Database, Check, ChevronDown, Globe } from 'lucide-react';

export const PeerSelector: React.FC = () => {
  const { selectedPeerIds, setSelectedPeerIds } = useWorkbenchContext();
  const [peers, setPeers] = useState<NexusPeer[]>([]);
  const [isOpen, setIsOpen] = useState(false);

  useEffect(() => {
    const fetchPeers = async () => {
      try {
        const data = await nexusService.getPeers();
        setPeers(data.filter(p => p.status !== NexusStatus.REVOKED));
      } catch (err) {
        console.error('Failed to load peers for selector', err);
      }
    };
    fetchPeers();
  }, []);

  const togglePeer = (id: string) => {
    if (selectedPeerIds.includes(id)) {
      setSelectedPeerIds(selectedPeerIds.filter(p => p !== id));
    } else {
      setSelectedPeerIds([...selectedPeerIds, id]);
    }
  };

  const isLocalOnly = selectedPeerIds.length === 0;

  return (
    <div className="relative">
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className={`flex items-center gap-2 px-3 py-1.5 rounded-full border transition-all text-[10px] font-bold uppercase tracking-widest ${
          isLocalOnly 
            ? 'bg-gray-800 border-gray-700 text-gray-500' 
            : 'bg-cyan-900/20 border-cyan-500/50 text-cyan-400 shadow-lg shadow-cyan-500/10'
        }`}
      >
        {isLocalOnly ? <Globe size={12} /> : <Database size={12} />}
        {isLocalOnly ? 'Local Only' : `${selectedPeerIds.length} Peers`}
        <ChevronDown size={10} className={`transition-transform ${isOpen ? 'rotate-180' : ''}`} />
      </button>

      {isOpen && (
        <>
          <div className="fixed inset-0 z-40" onClick={() => setIsOpen(false)} />
          <div className="absolute right-0 mt-2 w-64 bg-forge-navy border border-gray-800 rounded-2xl shadow-2xl z-50 p-2 animate-in fade-in zoom-in-95 duration-200">
            <div className="px-3 py-2 text-[10px] font-black text-gray-500 uppercase tracking-tighter border-b border-gray-800 mb-2">
              Broadcast Targets
            </div>
            <div className="max-h-48 overflow-y-auto space-y-1 custom-scrollbar">
              {peers.length === 0 && (
                <div className="px-3 py-4 text-center text-xs text-gray-600 italic">No peers available</div>
              )}
              {peers.map((peer) => (
                <button
                  key={peer.id}
                  onClick={() => togglePeer(peer.id)}
                  className="w-full flex items-center justify-between px-3 py-2 rounded-xl hover:bg-gray-800/50 transition-colors text-left"
                >
                  <div className="flex flex-col">
                    <span className="text-xs font-bold text-gray-200">{peer.name}</span>
                    <span className={`text-[8px] uppercase ${peer.status === NexusStatus.ONLINE ? 'text-green-500' : 'text-red-500'}`}>
                      {peer.status}
                    </span>
                  </div>
                  {selectedPeerIds.includes(peer.id) && (
                    <Check size={14} className="text-cyan-400" />
                  )}
                </button>
              ))}
            </div>
            {peers.length > 0 && (
              <div className="mt-2 pt-2 border-t border-gray-800 flex justify-between gap-2">
                <button 
                  onClick={() => setSelectedPeerIds([])}
                  className="flex-1 px-2 py-1.5 text-[9px] font-bold text-gray-500 hover:text-white uppercase transition-colors"
                >
                  Clear All
                </button>
                <button 
                  onClick={() => setSelectedPeerIds(peers.map(p => p.id))}
                  className="flex-1 px-2 py-1.5 text-[9px] font-bold text-cyan-500 hover:text-cyan-300 uppercase transition-colors"
                >
                  Select All
                </button>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
};
