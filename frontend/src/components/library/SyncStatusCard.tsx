"use client";

import React from 'react';
import { Cloud, CloudOff, RefreshCw, CheckCircle2, ArrowUpCircle, Loader2 } from 'lucide-react';
import { usePushSyncMutation } from '@/store/api/ingestforgeApi';
import { useToast } from '@/components/ToastProvider';

interface SyncProps {
  localCount: number;
  remoteCount: number;
}

export const SyncStatusCard = ({ localCount, remoteCount }: SyncProps) => {
  const { showToast } = useToast();
  const [pushSync, { isLoading: isSyncing }] = usePushSyncMutation();
  
  const outOfSync = localCount - remoteCount;
  const isUpToDate = outOfSync <= 0;

  const handleSync = async () => {
    try {
      await pushSync().unwrap();
      showToast('Corpus synchronization started', 'success');
    } catch (err) {
      showToast('Cloud connection failed', 'error');
    }
  };

  return (
    <div className="forge-card border-none bg-gradient-to-br from-forge-blue/30 to-black/40 p-6 flex items-center justify-between group">
      <div className="flex items-center gap-5">
        <div className={`p-4 rounded-[1.5rem] bg-gray-900 border border-gray-800 transition-colors ${isUpToDate ? 'text-green-500' : 'text-forge-accent animate-pulse'}`}>
          {isUpToDate ? <Cloud size={24} /> : <CloudOff size={24} />}
        </div>
        
        <div>
          <h3 className="font-bold text-white text-sm uppercase tracking-widest">Cloud Synchronization</h3>
          <p className="text-[10px] text-gray-500 font-bold mt-1 uppercase">
            {isUpToDate ? 'Everything is backed up' : `${outOfSync} chunks pending upload`}
          </p>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="text-right hidden md:block">
          <p className="text-[9px] font-black text-gray-600 uppercase">Local Repository</p>
          <p className="text-sm font-bold text-gray-300">{localCount.toLocaleString()} Chunks</p>
        </div>

        <button 
          onClick={handleSync}
          disabled={isSyncing || isUpToDate}
          className={`flex items-center gap-2 px-6 py-2.5 rounded-xl text-[10px] font-black uppercase tracking-[0.1em] transition-all ${
            isUpToDate 
              ? 'bg-gray-800 text-gray-600 border border-gray-700 cursor-not-allowed opacity-50' 
              : 'btn-primary shadow-xl shadow-forge-crimson/20'
          }`}
        >
          {isSyncing ? <Loader2 size={14} className="animate-spin" /> : <ArrowUpCircle size={14} />}
          {isSyncing ? 'Pushing...' : 'Sync to Cloud'}
        </button>
      </div>
    </div>
  );
};
