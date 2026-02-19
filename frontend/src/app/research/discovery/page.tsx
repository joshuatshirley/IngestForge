"use client";

import React from 'react';
import { 
  Globe, 
  Search, 
  FilePlus, 
  ExternalLink, 
  Calendar, 
  Users,
  Loader2,
  Sparkles
} from 'lucide-react';
import { useSearchDiscoveryMutation, useIngestDiscoveryMutation } from '@/store/api/ingestforgeApi';
import { useToast } from '@/components/ToastProvider';

export default function DiscoveryPage() {
  const { showToast } = useToast();
  const [query, setQuery] = React.useState('');
  const [source, setSource] = React.useState('arxiv');
  
  const [searchDiscovery, { data, isLoading }] = useSearchDiscoveryMutation();
  const [ingestDiscovery] = useIngestDiscoveryMutation();

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;
    try {
      await searchDiscovery({ query, source }).unwrap();
    } catch (err) {
      showToast('Discovery search failed', 'error');
    }
  };

  const handleIngest = async (url: string, title: string) => {
    try {
      await ingestDiscovery({ url, title }).unwrap();
      showToast(`Started ingesting: ${title}`, 'success');
    } catch (err) {
      showToast('Failed to start ingestion', 'error');
    }
  };

  return (
    <div className="max-w-6xl mx-auto space-y-12 animate-in fade-in duration-500">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Globe size={32} className="text-forge-crimson" />
            Global Discovery
          </h1>
          <p className="text-gray-400 mt-2">Find and import the latest academic research from global repositories.</p>
        </div>
      </div>

      <div className="forge-card border-none bg-gradient-to-r from-forge-blue/40 to-transparent p-10">
        <form onSubmit={handleSearch} className="flex flex-col md:flex-row gap-4">
          <div className="flex-1 relative">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 text-gray-500" size={20} />
            <input 
              type="text" 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search for papers, authors, or topics..."
              className="w-full bg-forge-navy border border-gray-700 rounded-2xl py-4 pl-12 pr-4 focus:border-forge-crimson outline-none transition-all shadow-xl"
            />
          </div>
          <select 
            value={source}
            onChange={(e) => setSource(e.target.value)}
            className="bg-gray-800 border border-gray-700 rounded-2xl px-6 py-4 text-sm font-bold text-gray-300 focus:border-forge-crimson outline-none cursor-pointer"
          >
            <option value="arxiv">arXiv</option>
            <option value="scholar">Semantic Scholar</option>
            <option value="crossref">CrossRef</option>
          </select>
          <button type="submit" disabled={isLoading} className="btn-primary px-10 rounded-2xl shadow-xl flex items-center justify-center gap-2">
            {isLoading ? <Loader2 size={20} className="animate-spin" /> : <Sparkles size={20} />}
            <span className="font-bold uppercase tracking-widest text-xs">Search</span>
          </button>
        </form>
      </div>

      <div className="grid grid-cols-1 gap-6">
        {data?.results.map((paper: any) => (
          <div key={paper.id} className="forge-card p-8 group hover:bg-forge-blue/20 transition-all border-gray-800 flex flex-col md:flex-row gap-8 relative overflow-hidden">
            <div className="flex-1 space-y-4 relative z-10">
              <div className="flex items-center gap-3">
                <span className="text-[10px] font-black text-forge-crimson bg-forge-crimson/10 px-2 py-0.5 rounded uppercase tracking-widest">
                  {paper.source}
                </span>
                <div className="flex items-center gap-2 text-xs text-gray-500">
                  <Calendar size={14} />
                  {paper.date}
                </div>
              </div>
              
              <h3 className="text-xl font-bold text-white group-hover:text-forge-crimson transition-colors leading-snug">
                {paper.title}
              </h3>
              
              <div className="flex items-center gap-4 text-sm text-gray-400">
                <div className="flex items-center gap-2">
                  <Users size={16} />
                  {paper.authors.join(', ')}
                </div>
              </div>

              <p className="text-sm text-gray-500 leading-relaxed line-clamp-2 italic">
                {paper.summary}
              </p>
            </div>

            <div className="flex flex-col justify-center gap-3 relative z-10">
              <button 
                onClick={() => handleIngest(paper.url, paper.title)}
                className="btn-primary py-3 px-6 rounded-xl flex items-center justify-center gap-2 text-xs font-bold uppercase tracking-widest shadow-lg shadow-forge-crimson/20"
              >
                <FilePlus size={18} />
                Ingest to Library
              </button>
              <a 
                href={paper.url} 
                target="_blank" 
                rel="noopener noreferrer"
                className="flex items-center justify-center gap-2 text-xs font-bold text-gray-500 hover:text-white transition-colors"
              >
                <ExternalLink size={14} />
                View Original PDF
              </a>
            </div>
            
            <div className="absolute top-[-20%] right-[-5%] w-48 h-48 bg-forge-crimson opacity-0 group-hover:opacity-5 blur-3xl transition-opacity rounded-full" />
          </div>
        ))}

        {!data && !isLoading && (
          <div className="h-64 flex flex-col items-center justify-center text-center opacity-30 space-y-4">
            <Globe size={48} />
            <p className="text-sm italic">Enter a query to discover academic papers globally.</p>
          </div>
        )}
      </div>
    </div>
  );
}
