"use client";

import React, { useState } from 'react';
import { 
  Wand2, 
  Trash2, 
  Zap, 
  Layers, 
  Split, 
  Merge, 
  Filter, 
  Loader2, 
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { useTransformCorpusMutation, useGetLibrariesQuery } from '@/store/api/ingestforgeApi';
import { useToast } from '@/components/ToastProvider';

const TRANSFORM_OPERATIONS = [
  { id: 'clean', name: 'Sanitize & Clean', desc: 'Normalize formatting, fix OCR errors, and remove noise.', icon: Wand2, color: 'text-blue-400' },
  { id: 'enrich', name: 'Auto-Enrich', desc: 'Re-generate embeddings and extract entities for old chunks.', icon: Zap, color: 'text-yellow-400' },
  { id: 'filter', name: 'Prune Chunks', desc: 'Remove low-quality or duplicate chunks from the library.', icon: Filter, color: 'text-red-400' },
  { id: 'merge', name: 'Merge Chunks', desc: 'Combine small adjacent chunks into larger semantic units.', icon: Merge, color: 'text-green-400' },
  { id: 'split', name: 'Deep Split', desc: 'Apply structural splitting to large monolithic chunks.', icon: Split, color: 'text-purple-400' },
];

export default function TransformPage() {
  const { showToast } = useToast();
  const [selectedOp, setSelectedOp] = useState('clean');
  const [targetLib, setTargetLib] = useState('default');
  
  const { data: libData } = useGetLibrariesQuery();
  const [runTransform, { isLoading }] = useTransformCorpusMutation();

  const handleRun = async () => {
    try {
      await runTransform({ operation: selectedOp, target_library: targetLib }).unwrap();
      showToast(`${selectedOp.toUpperCase()} operation queued`, 'success');
    } catch (err) {
      showToast('Transformation failed to start', 'error');
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-10 animate-in fade-in duration-500 pb-20">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Wand2 size={32} className="text-forge-crimson" />
            Corpus Transformer
          </h1>
          <p className="text-gray-400 mt-2">Apply batch processing operations to refine and optimize your knowledge base.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Operation Selection */}
        <div className="lg:col-span-2 space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {TRANSFORM_OPERATIONS.map((op) => (
              <button 
                key={op.id}
                onClick={() => setSelectedOp(op.id)}
                className={`forge-card p-6 text-left flex items-start gap-4 transition-all border-2 ${
                  selectedOp === op.id ? 'border-forge-crimson bg-forge-crimson/5' : 'border-gray-800 hover:border-gray-700'
                }`}
              >
                <div className={`p-3 rounded-xl bg-gray-800 ${op.color}`}>
                  <op.icon size={24} />
                </div>
                <div>
                  <h3 className="font-bold text-sm text-white">{op.name}</h3>
                  <p className="text-[10px] text-gray-500 font-medium leading-relaxed mt-1">{op.desc}</p>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Configuration & Launch */}
        <div className="lg:col-span-1 space-y-6">
          <div className="forge-card p-8 bg-gradient-to-br from-forge-blue/20 to-transparent space-y-8">
            <h3 className="font-bold text-xs uppercase tracking-widest text-gray-400 flex items-center gap-2">
              <Layers size={14} /> Execution Scope
            </h3>
            
            <div className="space-y-4">
              <div className="space-y-2">
                <label className="text-[10px] font-black text-gray-500 uppercase tracking-widest">Target Library</label>
                <select 
                  value={targetLib}
                  onChange={(e) => setTargetLib(e.target.value)}
                  className="w-full bg-gray-900 border border-gray-800 rounded-xl px-4 py-3 text-sm focus:border-forge-crimson outline-none cursor-pointer"
                >
                  {libData?.libraries.map((lib: any) => (
                    <option key={lib.id} value={lib.id}>{lib.name} ({lib.count})</option>
                  ))}
                </select>
              </div>

              <div className="p-4 bg-yellow-900/10 border border-yellow-900/30 rounded-xl flex gap-3">
                <AlertCircle size={16} className="text-yellow-500 shrink-0" />
                <p className="text-[10px] text-yellow-200/70 leading-relaxed font-medium uppercase tracking-tighter">
                  Caution: Transformations are permanent. Cloud sync is recommended before large batch runs.
                </p>
              </div>

              <button 
                onClick={handleRun}
                disabled={isLoading}
                className="w-full btn-primary py-4 rounded-2xl flex items-center justify-center gap-2 group shadow-xl shadow-forge-crimson/20 disabled:opacity-50"
              >
                {isLoading ? <Loader2 size={20} className="animate-spin" /> : <Zap size={18} />}
                <span className="font-bold uppercase tracking-widest text-xs">Run {selectedOp}</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
