"use client";

import React from 'react';
import { 
  Network, 
  RefreshCcw, 
  Filter, 
  Database,
  ZoomIn,
  ZoomOut,
  Focus
} from 'lucide-react';
import { useGetGraphQuery } from '@/store/api/ingestforgeApi';
import { useToast } from '@/components/ToastProvider';
import { KnowledgeGraphCanvas } from '@/components/research/KnowledgeGraphCanvas';

export default function KnowledgeGraphPage() {
  const { showToast } = useToast();
  const { data: graphData, isLoading, refetch } = useGetGraphQuery();

  const handleRefresh = async () => {
    await refetch();
    showToast('Graph updated from latest corpus state', 'info');
  };

  return (
    <div className="h-full flex flex-col gap-6 animate-in fade-in duration-500">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Network size={32} className="text-forge-crimson" />
            Concept Map Canvas
          </h1>
          <p className="text-gray-400 mt-2">Visualizing relationships between entities, concepts, and documents.</p>
        </div>
        <div className="flex gap-3">
          <button className="flex items-center gap-2 text-xs bg-gray-800 hover:bg-gray-700 px-4 py-2 rounded-xl border border-gray-700 transition-all text-gray-300">
            <Filter size={14} />
            Filter Concepts
          </button>
          <button 
            onClick={handleRefresh}
            className="flex items-center gap-2 text-xs bg-gray-800 hover:bg-gray-700 px-4 py-2 rounded-xl border border-gray-700 transition-all text-gray-300"
          >
            <RefreshCcw size={14} className={isLoading ? "animate-spin" : ""} />
            Rebuild Graph
          </button>
        </div>
      </div>

      <div className="flex-1 forge-card p-0 relative overflow-hidden bg-gray-900/40 border-gray-800">
        {/* Graph Controls Overlay */}
        <div className="absolute bottom-6 left-6 flex flex-col gap-2 z-10">
          <button className="p-3 bg-gray-800/80 backdrop-blur-md hover:bg-gray-700 rounded-xl border border-gray-700 transition-all text-gray-400 hover:text-white shadow-2xl">
            <ZoomIn size={18} />
          </button>
          <button className="p-3 bg-gray-800/80 backdrop-blur-md hover:bg-gray-700 rounded-xl border border-gray-700 transition-all text-gray-400 hover:text-white shadow-2xl">
            <ZoomOut size={18} />
          </button>
          <button className="p-3 bg-gray-800/80 backdrop-blur-md hover:bg-gray-700 rounded-xl border border-gray-700 transition-all text-gray-400 hover:text-white shadow-2xl mt-4">
            <Focus size={18} />
          </button>
        </div>

        {/* Legend Overlay */}
        <div className="absolute top-6 right-6 p-4 bg-gray-900/80 backdrop-blur-md rounded-2xl border border-gray-800 z-10 space-y-3 min-w-[160px]">
          <p className="text-[10px] font-bold text-gray-500 uppercase tracking-widest mb-2">Entity Legend</p>
          {[
            { label: 'Document', color: 'bg-blue-500' },
            { label: 'Concept', color: 'bg-forge-crimson' },
            { label: 'Entity', color: 'bg-yellow-500' },
            { label: 'Topic', color: 'bg-purple-500' },
          ].map((item) => (
            <div key={item.label} className="flex items-center gap-3">
              <div className={`w-2 h-2 rounded-full ${item.color}`} />
              <span className="text-xs text-gray-300">{item.label}</span>
            </div>
          ))}
        </div>

        {/* The D3 Canvas */}
        {!isLoading && graphData && (
          <KnowledgeGraphCanvas data={graphData} />
        )}

        {isLoading && (
          <div className="absolute inset-0 flex flex-col items-center justify-center bg-forge-navy/20 backdrop-blur-sm space-y-4">
            <RefreshCcw size={40} className="animate-spin text-forge-crimson" />
            <p className="text-sm font-bold text-gray-500 uppercase tracking-[0.2em]">Synthesizing concept map...</p>
          </div>
        )}

        {!isLoading && (!graphData || graphData.nodes.length === 0) && (
          <div className="absolute inset-0 flex flex-col items-center justify-center text-center opacity-30 space-y-4">
            <Database size={48} />
            <p className="text-sm italic">No concepts found in current corpus. Ingest more documents to build the map.</p>
          </div>
        )}
      </div>
    </div>
  );
}
