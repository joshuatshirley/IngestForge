"use client";

import React from 'react';
import { BookMarked, Bookmark, Tag, Maximize2 } from 'lucide-react';
import { SourcedResult } from './types';

interface EvidenceSidebarProps {
  results: SourcedResult[];
  onToggleBookmark: (id: string) => void;
  onAddTag: (id: string) => void;
  onPreview: (id: string) => void;
}

export const EvidenceSidebar: React.FC<EvidenceSidebarProps> = ({
  results,
  onToggleBookmark,
  onAddTag,
  onPreview,
}) => {
  return (
    <div className="w-80 space-y-6 flex flex-col">
      <div className="forge-card flex-1 overflow-y-auto custom-scrollbar p-0">
        <div className="p-6 border-b border-gray-800 bg-gray-800 bg-opacity-20">
          <h3 className="font-bold flex items-center gap-2 text-sm">
            <BookMarked size={18} className="text-forge-crimson" />
            Sourced Evidence
          </h3>
        </div>
        <div className="p-4 space-y-4">
          {results.map((result, i) => (
            <div key={i} className="bg-forge-navy bg-opacity-40 border border-gray-800 rounded-2xl p-4 space-y-4 hover:border-gray-600 transition-all group">
              <div className="flex justify-between items-center">
                <span className="text-[10px] font-mono bg-gray-800 px-2 py-0.5 rounded-lg text-gray-500">CHUNK_{i+1}</span>
                <div className="flex items-center gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button 
                    onClick={() => onToggleBookmark(result.id)} 
                    className={`p-1.5 hover:bg-gray-800 rounded-md ${result.metadata.bookmarked ? 'text-yellow-400' : 'text-gray-400'}`}
                  >
                    <Bookmark size={14} />
                  </button>
                  <button 
                    onClick={() => onAddTag(result.id)} 
                    className="p-1.5 hover:bg-gray-800 rounded-md text-gray-400 hover:text-forge-accent"
                  >
                    <Tag size={14} />
                  </button>
                  <button 
                    onClick={() => onPreview(result.id)} 
                    className="p-1.5 hover:bg-gray-800 rounded-md text-gray-400 hover:text-white"
                  >
                    <Maximize2 size={14} />
                  </button>
                </div>
              </div>
              <p className="text-xs text-gray-400 line-clamp-4 italic leading-relaxed">"{result.content}"</p>
              <div className="pt-3 border-t border-gray-800 flex justify-between items-center text-[10px]">
                <span className="font-bold text-gray-500 truncate max-w-[120px]">{result.metadata.source}</span>
                <span className="text-forge-accent font-bold">{(result.score * 100).toFixed(1)}%</span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
