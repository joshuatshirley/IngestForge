'use client';

import React, { useState } from 'react';
import { FileText, Copy, ExternalLink, ChevronRight, BarChart3, CheckCircle2 } from 'lucide-react';
import { formatCitation, CitationStyle } from '@/utils/citationFormatter';
import { useToast } from '@/components/ToastProvider';
import { useAddVerifiedExampleMutation } from '@/store/api/ingestforgeApi';

/**
 * US-152: SRC-UI SearchResultCard Sub-components.
 * Rule #4: Extracted from main component.
 */

export const CardHeader: React.FC<{ source: string; section?: string; page?: number; score: number }> = ({ 
  source, section, page, score 
}) => (
  <div className="flex justify-between items-start">
    <div className="flex items-center gap-3">
      <div className="p-2.5 rounded-xl bg-gray-800 border border-gray-700 text-forge-crimson group-hover:bg-forge-crimson group-hover:text-white transition-all duration-500 shadow-inner">
        <FileText size={18} />
      </div>
      <div>
        <h4 className="font-bold text-sm text-gray-200 group-hover:text-white transition-colors">{source}</h4>
        <div className="flex items-center gap-2 mt-1">
          <span className="text-[10px] font-black text-gray-500 uppercase tracking-widest">{section || 'General'}</span>
          {page && <span className="text-[10px] font-bold text-forge-accent bg-forge-accent/10 px-1.5 py-0.5 rounded">PAGE {page}</span>}
        </div>
      </div>
    </div>
    
    <div className="flex flex-col items-end gap-1">
      <div className="flex items-center gap-1.5 text-forge-accent font-mono text-xs">
        <BarChart3 size={12} />
        <span className="font-black">{(score * 100).toFixed(1)}%</span>
      </div>
      <div className="w-16 bg-gray-800 h-1 rounded-full overflow-hidden">
        <div className="h-full bg-forge-accent" style={{ width: `${score * 100}%` }} />
      </div>
    </div>
  </div>
);

export const CardFooter: React.FC<{
  id: string; content: string; source: string; section?: string; page?: number; author?: string; entities: any[]; domain: string; onViewSource: (id: string) => void;
}> = (props) => {
  const { showToast } = useToast();
  const [citationStyle, setStyle] = useState<CitationStyle>('APA');
  const [addVerifiedExample, { isLoading: isVerifying }] = useAddVerifiedExampleMutation();

  const handleCopy = () => {
    const text = formatCitation({ author: props.author, sourceTitle: props.source, section: props.section, pageStart: props.page }, citationStyle);
    navigator.clipboard.writeText(text);
    showToast(`Citation copied in ${citationStyle} format`, 'success');
  };

  const handleVerify = async () => {
    try {
      await addVerifiedExample({ id: `v_${Date.now()}`, input_text: props.content, output_json: { entities: props.entities }, domain: props.domain, verified_by: 'manual' }).unwrap();
      showToast('Extraction verified', 'success');
    } catch (err) { showToast('Verification failed', 'error'); }
  };

  return (
    <div className="pt-6 border-t border-gray-800 flex flex-wrap justify-between items-center gap-4">
      <div className="flex items-center gap-2">
        <select value={citationStyle} onChange={(e) => setStyle(e.target.value as CitationStyle)} className="bg-gray-800 border border-gray-700 rounded-lg px-2 py-1.5 text-[10px] font-bold text-gray-400 outline-none focus:border-forge-crimson transition-colors">
          <option value="APA">APA</option><option value="MLA">MLA</option><option value="Chicago">Chicago</option>
        </select>
        <button onClick={handleCopy} className="flex items-center gap-2 px-3 py-1.5 text-[10px] font-black text-gray-400 hover:text-white hover:bg-gray-800 rounded-lg transition-all uppercase tracking-tighter">
          <Copy size={12} /> Copy
        </button>
      </div>
      <div className="flex items-center gap-3">
        <button onClick={handleVerify} disabled={isVerifying || props.entities.length === 0} className="flex items-center gap-2 px-3 py-2 bg-emerald-500/10 border border-emerald-500/30 rounded-xl text-[10px] font-bold text-emerald-400 hover:bg-emerald-500 hover:text-white transition-all uppercase tracking-widest disabled:opacity-50">
          <CheckCircle2 size={12} /> Verify
        </button>
        <button onClick={() => props.onViewSource(props.id)} className="flex items-center gap-2 px-4 py-2 bg-forge-crimson/10 border border-forge-crimson/30 rounded-xl text-[10px] font-bold text-forge-crimson hover:bg-forge-crimson hover:text-white transition-all uppercase tracking-widest shadow-lg">
          <ExternalLink size={12} /> View <ChevronRight size={12} />
        </button>
      </div>
    </div>
  );
};
