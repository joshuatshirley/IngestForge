"use client";

import React, { useState } from 'react';
import { Globe, X, Cloud, Send, Loader2, Info } from 'lucide-react';
import { useIngestRemoteDocumentMutation } from '@/store/api/ingestforgeApi';
import { useToast } from '@/components/ToastProvider';

interface ModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const RemoteSourceModal = ({ isOpen, onClose }: ModalProps) => {
  const { showToast } = useToast();
  const [platform, setPlatform] = useState('gdocs');
  const [sourceId, setSourceId] = useState('');
  const [ingestRemote, { isLoading }] = useIngestRemoteDocumentMutation();

  if (!isOpen) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sourceId.trim()) return;

    try {
      await ingestRemote({ platform, source_id: sourceId }).unwrap();
      showToast('Remote ingestion task queued', 'success');
      onClose();
    } catch (err) {
      showToast('Cloud connection failed', 'error');
    }
  };

  return (
    <div className="fixed inset-0 z-[300] bg-black/60 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="forge-card max-w-md w-full p-8 space-y-8 animate-in zoom-in-95 duration-300">
        <div className="flex justify-between items-start">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-xl bg-forge-crimson/10 text-forge-crimson">
              <Cloud size={24} />
            </div>
            <div>
              <h3 className="font-bold text-lg text-white">Import from Cloud</h3>
              <p className="text-[10px] font-bold text-gray-500 uppercase tracking-widest">Multi-Source Ingestion</p>
            </div>
          </div>
          <button onClick={onClose} className="p-2 text-gray-500 hover:text-white"><X size={20} /></button>
        </div>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div className="space-y-4">
            <div className="grid grid-cols-2 gap-3">
              {['gdocs', 'notion'].map((p) => (
                <button 
                  key={p} type="button" onClick={() => setPlatform(p)}
                  className={`py-3 rounded-xl border text-[10px] font-black uppercase tracking-widest transition-all ${
                    platform === p ? 'bg-forge-crimson border-forge-crimson text-white' : 'bg-gray-900 border-gray-800 text-gray-500'
                  }`}
                >
                  {p === 'gdocs' ? 'Google Docs' : 'Notion (Beta)'}
                </button>
              ))}
            </div>

            <div className="space-y-2">
              <label className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Document ID or URL</label>
              <input 
                required value={sourceId} onChange={(e) => setSourceId(e.target.value)}
                placeholder="e.g. 1aBC... or https://docs.google.com/..."
                className="w-full bg-gray-900 border border-gray-800 rounded-xl px-4 py-3 text-sm focus:border-forge-crimson outline-none"
              />
            </div>
          </div>

          <div className="bg-blue-900/10 border border-blue-900/30 p-4 rounded-xl flex gap-3 items-start">
            <Info size={16} className="text-blue-400 shrink-0 mt-0.5" />
            <p className="text-[10px] text-blue-300 leading-relaxed font-medium">
              Note: Ensure the IngestForge service account has 'Viewer' access to this document if private.
            </p>
          </div>

          <button 
            type="submit" disabled={isLoading}
            className="w-full btn-primary py-4 rounded-2xl flex items-center justify-center gap-2 group shadow-xl shadow-forge-crimson/20 disabled:opacity-50"
          >
            {isLoading ? <Loader2 size={20} className="animate-spin" /> : <Send size={18} />}
            <span className="font-bold">Queue Ingestion</span>
          </button>
        </form>
      </div>
    </div>
  );
};
