"use client";

import React from 'react';
import { X, ExternalLink, Loader2 } from 'lucide-react';
import { useGetChunkContextQuery } from '@/store/api/ingestforgeApi';

interface PreviewModalProps {
  chunkId: string | null;
  onClose: () => void;
}

export const PreviewModal = ({ chunkId, onClose }: PreviewModalProps) => {
  const { data, isLoading } = useGetChunkContextQuery(chunkId || '', {
    skip: !chunkId,
  });

  if (!chunkId) return null;

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="w-full max-w-4xl bg-forge-blue border border-gray-700 rounded-[2.5rem] shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header */}
        <div className="p-6 border-b border-gray-800 flex items-center justify-between bg-forge-navy/30">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-forge-accent/20 rounded-lg flex items-center justify-center text-forge-accent">
              <Maximize2 size={18} />
            </div>
            <h2 className="font-bold">Document Context Preview</h2>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-gray-800 rounded-full transition-colors text-gray-500 hover:text-white">
            <X size={24} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-8 space-y-6 custom-scrollbar bg-forge-navy/10">
          {isLoading ? (
            <div className="h-64 flex flex-col items-center justify-center gap-4 text-gray-500">
              <Loader2 size={40} className="animate-spin text-forge-crimson" />
              <p className="text-sm italic">Loading document context...</p>
            </div>
          ) : (
            <div className="space-y-8">
              {data?.context.map((item: any) => (
                <div 
                  key={item.id} 
                  className={`p-6 rounded-3xl transition-all ${
                    item.is_target 
                      ? 'bg-forge-crimson/10 border-2 border-forge-crimson shadow-xl shadow-forge-crimson/5' 
                      : 'bg-gray-800/20 border border-gray-800 opacity-60'
                  }`}
                >
                  {item.is_target && (
                    <div className="text-[10px] font-bold text-forge-crimson uppercase tracking-widest mb-4">
                      Target Passage
                    </div>
                  )}
                  <p className={`leading-relaxed ${item.is_target ? 'text-white text-lg' : 'text-gray-400'}`}>
                    {item.content}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-gray-800 bg-forge-navy/30 flex justify-between items-center">
          <p className="text-xs text-gray-500 italic font-mono">{chunkId}</p>
          <button className="flex items-center gap-2 text-xs font-bold text-forge-accent hover:text-white transition-colors">
            <ExternalLink size={14} />
            Open Source Document
          </button>
        </div>
      </div>
    </div>
  );
};

import { Maximize2 } from 'lucide-react';
