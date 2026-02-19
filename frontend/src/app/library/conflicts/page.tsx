"use client";

import React from 'react';
import { 
  ShieldAlert, 
  ArrowLeftRight, 
  FileText, 
  CheckCircle2, 
  AlertCircle,
  Loader2,
  RefreshCcw,
  Search
} from 'lucide-react';
import { useGetContradictionsQuery } from '@/store/api/ingestforgeApi';
import { useToast } from '@/components/ToastProvider';

export default function ConflictCenterPage() {
  const { showToast } = useToast();
  const { data, isLoading, refetch, isFetching } = useGetContradictionsQuery();

  return (
    <div className="max-w-6xl mx-auto space-y-10 animate-in fade-in duration-500 pb-20">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <ShieldAlert size={32} className="text-forge-crimson" />
            Contradiction Center
          </h1>
          <p className="text-gray-400 mt-2">Automatically identifying logical conflicts and factual disagreements across your corpus.</p>
        </div>
        <button 
          onClick={() => refetch()}
          disabled={isLoading || isFetching}
          className="flex items-center gap-2 text-xs bg-gray-800 hover:bg-gray-700 px-6 py-3 rounded-xl border border-gray-700 transition-all text-white font-bold uppercase tracking-widest"
        >
          {isLoading || isFetching ? <Loader2 size={16} className="animate-spin" /> : <RefreshCcw size={16} />}
          Re-Audit Corpus
        </button>
      </div>

      {isLoading ? (
        <div className="h-96 flex flex-col items-center justify-center space-y-4 text-gray-500">
          <Loader2 size={48} className="animate-spin text-forge-crimson" />
          <p className="text-sm font-medium animate-pulse uppercase tracking-widest">Performing Logical NLI Audit...</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 gap-8">
          {data?.conflicts.map((conflict: any, i: number) => (
            <div key={i} className="forge-card p-0 overflow-hidden border-forge-crimson/20 bg-gray-900/20">
              <div className="bg-forge-crimson/10 px-8 py-3 border-b border-forge-crimson/20 flex justify-between items-center">
                <div className="flex items-center gap-2 text-forge-crimson">
                  <AlertCircle size={14} />
                  <span className="text-[10px] font-black uppercase tracking-widest">Factual Discrepancy Detected</span>
                </div>
                <span className="text-[10px] font-bold text-gray-500 uppercase">Confidence: {(conflict.confidence * 100).toFixed(0)}%</span>
              </div>

              <div className="p-8 flex flex-col md:flex-row items-center gap-8 relative">
                {/* Claim A */}
                <div className="flex-1 space-y-4">
                  <div className="flex items-center gap-2 text-[9px] font-black text-gray-500 uppercase">
                    <FileText size={12} />
                    Source: {conflict.claim_a.source}
                  </div>
                  <p className="text-sm text-gray-200 leading-relaxed italic">"{conflict.claim_a.text}"</p>
                </div>

                {/* Divider */}
                <div className="hidden md:flex flex-col items-center gap-2 opacity-30">
                  <div className="w-px h-12 bg-gray-700" />
                  <ArrowLeftRight size={20} className="text-forge-crimson" />
                  <div className="w-px h-12 bg-gray-700" />
                </div>

                {/* Claim B */}
                <div className="flex-1 space-y-4 text-right md:text-left">
                  <div className="flex items-center justify-end md:justify-start gap-2 text-[9px] font-black text-gray-500 uppercase">
                    <FileText size={12} />
                    Source: {conflict.claim_b.source}
                  </div>
                  <p className="text-sm text-gray-200 leading-relaxed italic">"{conflict.claim_b.text}"</p>
                </div>
              </div>

              <div className="bg-black/40 p-4 border-t border-gray-800 flex justify-end gap-4">
                <button className="text-[10px] font-bold text-gray-500 hover:text-white uppercase tracking-tighter transition-colors">Ignore</button>
                <button className="text-[10px] font-bold text-forge-accent hover:text-white uppercase tracking-tighter transition-colors">Resolve Conflict</button>
              </div>
            </div>
          ))}

          {data?.conflicts.length === 0 && (
            <div className="h-64 flex flex-col items-center justify-center text-center opacity-30 space-y-4">
              <CheckCircle2 size={48} className="text-green-500" />
              <p className="text-sm italic">No factual contradictions found. Your corpus is logically consistent.</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
