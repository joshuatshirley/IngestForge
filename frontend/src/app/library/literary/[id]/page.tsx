"use client";

import React from 'react';
import { useParams } from 'next/navigation';
import { 
  BookOpen, 
  Users, 
  Sparkles, 
  ChevronRight, 
  History, 
  BarChart3,
  Loader2,
  AlertCircle
} from 'lucide-react';
import { useGetLiteraryAnalysisQuery } from '@/store/api/ingestforgeApi';
import { 
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip,
  BarChart, Bar, XAxis, YAxis, CartesianGrid
} from 'recharts';

export default function LiteraryAnalysisPage() {
  const params = useParams();
  const docId = params.id as string;
  const { data, isLoading, error } = useGetLiteraryAnalysisQuery(docId);

  const COLORS = ['#e94560', '#4fc3f7', '#4ecca3', '#9c27b0', '#ffc107'];

  if (isLoading) return (
    <div className="h-96 flex flex-col items-center justify-center space-y-4 text-gray-500">
      <Loader2 size={48} className="animate-spin text-forge-crimson" />
      <p className="text-sm font-medium animate-pulse uppercase tracking-widest">Analyzing Narrative Structures...</p>
    </div>
  );

  if (error) return (
    <div className="h-96 flex flex-col items-center justify-center space-y-4 text-red-500">
      <AlertCircle size={48} />
      <p className="text-sm font-medium uppercase tracking-widest">Analysis Failed</p>
    </div>
  );

  return (
    <div className="max-w-7xl mx-auto space-y-10 animate-in fade-in duration-500 pb-20">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <BookOpen size={32} className="text-forge-crimson" />
            Literary Forge: {data?.document_id}
          </h1>
          <p className="text-gray-400 mt-2">Deep-dive narrative intelligence including character networks and thematic heatmaps.</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* Character Roster */}
        <div className="lg:col-span-1 space-y-6">
          <div className="forge-card p-8 border-gray-800 bg-gray-900/20">
            <h3 className="font-bold mb-8 flex items-center gap-2 text-xs uppercase tracking-widest text-gray-400">
              <Users size={16} className="text-forge-accent" />
              Character Roster
            </h3>
            <div className="space-y-4">
              {data?.characters.map((char: any, i: number) => (
                <div key={i} className="p-4 rounded-xl bg-gray-800/50 border border-gray-700 hover:border-forge-accent transition-all group">
                  <div className="flex justify-between items-center mb-2">
                    <p className="font-bold text-white group-hover:text-forge-accent transition-colors">{char.name}</p>
                    <span className="text-[10px] font-black bg-gray-900 px-2 py-0.5 rounded text-gray-500 uppercase">{char.role || 'Secondary'}</span>
                  </div>
                  <p className="text-xs text-gray-400 leading-relaxed italic">"{char.traits || char.description}"</p>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Thematic Distribution */}
        <div className="lg:col-span-2 space-y-8">
          <div className="forge-card p-8 border-gray-800 bg-gray-900/20 h-[400px]">
            <h3 className="font-bold mb-8 flex items-center gap-2 text-xs uppercase tracking-widest text-gray-400">
              <Sparkles size={16} className="text-yellow-500" />
              Thematic Weighting
            </h3>
            <div className="h-[280px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={data?.themes}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                  <XAxis dataKey="name" stroke="#94a3b8" fontSize={10} tickLine={false} axisLine={false} />
                  <YAxis stroke="#94a3b8" fontSize={10} tickLine={false} axisLine={false} />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#16213e', border: '1px solid #374151', borderRadius: '12px' }}
                    itemStyle={{ color: '#e94560' }}
                  />
                  <Bar dataKey="prominence" fill="#e94560" radius={[4, 4, 0, 0]} barSize={40} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="forge-card p-8 border-dashed border-2 border-gray-800 flex flex-col items-center justify-center text-center space-y-4 opacity-60 hover:opacity-100 transition-all">
              <History size={32} className="text-forge-crimson" />
              <div>
                <h4 className="font-bold text-white text-sm">Chronological Arc</h4>
                <p className="text-[10px] text-gray-500 mt-1">Visualize narrative tension over time.</p>
              </div>
              <button className="text-[10px] font-bold text-forge-crimson border border-forge-crimson/20 px-4 py-2 rounded-lg hover:bg-forge-crimson hover:text-white transition-all uppercase">Analyze Arc</button>
            </div>

            <div className="forge-card p-8 border-dashed border-2 border-gray-800 flex flex-col items-center justify-center text-center space-y-4 opacity-60 hover:opacity-100 transition-all">
              <BarChart3 size={32} className="text-forge-accent" />
              <div>
                <h4 className="font-bold text-white text-sm">Comparison Matrix</h4>
                <p className="text-[10px] text-gray-500 mt-1">Compare this work with others in the library.</p>
              </div>
              <button className="text-[10px] font-bold text-forge-accent border border-forge-accent/20 px-4 py-2 rounded-lg hover:bg-forge-accent hover:text-white transition-all uppercase">Run Matrix</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
