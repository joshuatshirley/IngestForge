"use client";

import React, { useState } from 'react';
import { 
  BrainCircuit, 
  Trash2, 
  Search, 
  ShieldCheck, 
  History, 
  ExternalLink,
  Loader2,
  Calendar,
  Filter
} from 'lucide-react';
import { useGetAgentMemoryQuery, useDeleteAgentMemoryMutation } from '@/store/api/ingestforgeApi';
import { useToast } from '@/components/ToastProvider';

export default function MemoryBankPage() {
  const { showToast } = useToast();
  const [search, setSearch] = useState('');
  const { data, isLoading } = useGetAgentMemoryQuery();
  const [deleteFact] = useDeleteAgentMemoryMutation();

  const handleDelete = async (id: number) => {
    try {
      await deleteFact(id).unwrap();
      showToast('Fact removed from memory', 'success');
    } catch (err) {
      showToast('Failed to remove fact', 'error');
    }
  };

  const filteredFacts = data?.facts.filter((f: any) => 
    f.fact.toLowerCase().includes(search.toLowerCase())
  ) || [];

  return (
    <div className="max-w-6xl mx-auto space-y-10 animate-in fade-in duration-500">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <BrainCircuit size={32} className="text-forge-crimson" />
            Agent Memory Bank
          </h1>
          <p className="text-gray-400 mt-2">Browse and curate the long-term knowledge synthesized by your agents.</p>
        </div>
      </div>

      <div className="flex flex-col md:flex-row gap-4 items-center">
        <div className="flex-1 relative w-full">
          <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500" size={18} />
          <input 
            type="text" 
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter learned facts..."
            className="w-full bg-gray-900 border border-gray-800 rounded-2xl py-3 pl-12 pr-4 focus:border-forge-crimson outline-none transition-all"
          />
        </div>
        <button className="flex items-center gap-2 text-xs bg-gray-800 hover:bg-gray-700 px-6 py-3 rounded-xl border border-gray-700 transition-all">
          <Filter size={14} />
          Refine History
        </button>
      </div>

      <div className="grid grid-cols-1 gap-4">
        {isLoading ? (
          <div className="h-64 flex items-center justify-center text-gray-500">
            <Loader2 className="animate-spin" />
          </div>
        ) : filteredFacts.map((fact: any) => (
          <div key={fact.id} className="forge-card p-6 flex items-start justify-between group hover:bg-forge-blue/10 transition-all border-gray-800">
            <div className="space-y-4 flex-1 pr-8">
              <div className="flex items-center gap-3">
                <span className="text-[10px] font-black text-forge-accent bg-forge-accent/10 px-2 py-0.5 rounded uppercase tracking-widest">
                  Verified Fact
                </span>
                <div className="flex items-center gap-2 text-[10px] text-gray-500 font-bold uppercase">
                  <Calendar size={12} />
                  Learned: {fact.learned_at}
                </div>
              </div>
              
              <p className="text-sm text-gray-200 leading-relaxed">
                {fact.fact}
              </p>

              <div className="flex items-center gap-6">
                <div className="flex items-center gap-2 text-[10px] text-gray-500 font-bold uppercase">
                  <ShieldCheck size={14} className="text-green-500" />
                  Confidence: {(fact.confidence * 100).toFixed(0)}%
                </div>
                {fact.source && (
                  <div className="flex items-center gap-2 text-[10px] text-gray-500 font-bold uppercase">
                    <History size={14} />
                    Source: {fact.source}
                  </div>
                )}
              </div>
            </div>

            <div className="flex flex-col gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
              <button 
                onClick={() => handleDelete(fact.id)}
                className="p-2 text-gray-500 hover:text-red-500 transition-colors"
              >
                <Trash2 size={18} />
              </button>
              <button className="p-2 text-gray-500 hover:text-white transition-colors">
                <ExternalLink size={18} />
              </button>
            </div>
          </div>
        ))}

        {!isLoading && filteredFacts.length === 0 && (
          <div className="h-64 flex flex-col items-center justify-center text-center opacity-30 space-y-4">
            <BrainCircuit size={48} />
            <p className="text-sm italic">Memory bank is empty. Launch missions to archive new findings.</p>
          </div>
        )}
      </div>
    </div>
  );
}
