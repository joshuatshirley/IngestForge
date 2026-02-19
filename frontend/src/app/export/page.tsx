"use client";

import React from 'react';
import { 
  Download, 
  Share2, 
  FileJson, 
  FileText, 
  Archive,
  ArrowRight,
  ShieldCheck,
  CheckCircle,
  Loader2
} from 'lucide-react';
import { useToast } from '@/components/ToastProvider';

const EXPORT_TYPES = [
  { id: 'markdown', name: 'Research Notes', desc: 'Full research session with inline citations and metadata.', format: 'Markdown (.md)', icon: FileText, color: 'text-blue-400' },
  { id: 'outline', name: 'Draft Outline', desc: 'Hierarchical outline mapped to evidence chunks for writing.', format: 'Word (.docx)', icon: Download, color: 'text-green-400' },
  { id: 'zip', name: 'Portable Corpus', desc: 'Package entire project (documents + embeddings) for sharing.', format: 'Archive (.zip)', icon: Archive, color: 'text-orange-400' },
  { id: 'jsonl', name: 'Raw Data', desc: 'Export all chunks and metadata in machine-readable format.', format: 'JSONL', icon: FileJson, color: 'text-purple-400' },
];

export default function ExportPage() {
  const { showToast } = useToast();
  const [loadingId, setLoadingId] = React.useState<string | null>(null);

  const handleExport = async (id: string) => {
    setLoadingId(id);
    // Simulate generation and download
    setTimeout(() => {
      setLoadingId(null);
      showToast(`${id.toUpperCase()} export ready for download`, 'success');
    }, 1500);
  };

  return (
    <div className="max-w-5xl mx-auto space-y-12 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold tracking-tight text-white">Export & Portability</h1>
        <p className="text-gray-400 mt-2">Take your research anywhere. Format your findings for writing or sharing.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {EXPORT_TYPES.map((type) => (
          <div key={type.id} className="forge-card flex flex-col p-8 group hover:bg-forge-blue/20 transition-all border-gray-800 relative overflow-hidden">
            <div className="flex justify-between items-start relative z-10">
              <div className={`p-4 rounded-2xl bg-gray-800/50 border border-gray-700 group-hover:scale-110 transition-transform ${type.color}`}>
                <type.icon size={28} />
              </div>
              <span className="text-[10px] font-black text-gray-500 uppercase tracking-widest bg-gray-800 px-3 py-1 rounded-full">
                {type.format}
              </span>
            </div>

            <div className="mt-8 space-y-2 relative z-10">
              <h3 className="text-xl font-bold text-white">{type.name}</h3>
              <p className="text-sm text-gray-500 leading-relaxed max-w-xs">{type.desc}</p>
            </div>

            <div className="mt-10 pt-6 border-t border-gray-800 flex justify-between items-center relative z-10">
              <div className="flex items-center gap-2 text-[10px] font-bold text-gray-500 uppercase">
                <ShieldCheck size={14} className="text-forge-accent" />
                Verified Metadata
              </div>
              <button 
                onClick={() => handleExport(type.id)}
                disabled={loadingId !== null}
                className="flex items-center gap-2 text-xs font-bold text-forge-crimson group-hover:text-white transition-colors"
              >
                {loadingId === type.id ? <Loader2 size={16} className="animate-spin" /> : (
                  <>
                    Prepare Export
                    <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
                  </>
                )}
              </button>
            </div>
            
            {/* Background Accent */}
            <div className="absolute bottom-[-20%] right-[-10%] w-32 h-32 bg-forge-crimson opacity-0 group-hover:opacity-5 blur-3xl transition-opacity rounded-full" />
          </div>
        ))}
      </div>

      <div className="forge-card border-dashed border-2 border-gray-800 p-12 flex flex-col items-center justify-center text-center space-y-6">
        <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center text-forge-accent border border-gray-700">
          <Share2 size={32} />
        </div>
        <div className="space-y-2">
          <h3 className="text-lg font-bold text-white">Cloud Sync (Beta)</h3>
          <p className="text-sm text-gray-500 max-w-sm">
            Sync your research corpus to a shared PostgreSQL instance for multi-user collaboration.
          </p>
        </div>
        <button className="text-[10px] font-bold text-forge-accent border border-forge-accent/20 px-8 py-3 rounded-2xl hover:bg-forge-accent hover:text-black transition-all uppercase tracking-[0.2em]">
          Connect Shared Database
        </button>
      </div>
    </div>
  );
}
