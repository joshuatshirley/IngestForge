'use client';

import React, { useState } from 'react';
import { Copy, ExternalLink, ChevronRight, CheckCircle2 } from 'lucide-react';
import { TextHighlighter } from './TextHighlighter';
import { EntityBadge } from './EntityBadge';
import { formatCitation, CitationStyle } from '@/utils/citationFormatter';
import { useToast } from '@/components/ToastProvider';
import { useAddVerifiedExampleMutation } from '@/store/api/ingestforgeApi';
import { useWorkbenchContext } from '@/context/WorkbenchContext';
import { CardHeader, CardFooter } from './SearchResultCardParts';

/**
 * US-152: SRC-UI Optimized Result Card.
 * JPL Rule #4: Refactored logic to sub-components.
 */

export interface SearchResultCardProps {
  id: string;
  content: string;
  score: number;
  source: string;
  section?: string;
  page?: number;
  author?: string;
  entities?: Array<{ type: string, text: string }>;
  query: string;
  onViewSource: (id: string) => void;
  domain?: string;
}

export const SearchResultCard: React.FC<SearchResultCardProps> = ({
  id, content, score, source, section, page, author, entities = [], query, onViewSource, domain = 'general'
}) => {
  const { activeNodeId } = useWorkbenchContext();
  const isActive = activeNodeId === id || entities.some(e => e.text === activeNodeId);

  return (
    <div className={`forge-card group p-6 space-y-6 transition-all duration-300 backdrop-blur-xl border shadow-xl ${
      isActive ? 'border-forge-cyan ring-4 ring-forge-cyan/10 bg-forge-cyan/5 scale-[1.02]' : 'hover:border-forge-crimson/40 bg-gradient-to-br from-gray-900/80 to-transparent border-gray-800'
    }`}>
      <CardHeader source={source} section={section} page={page} score={score} />

      <div className="relative">
        <div className={`absolute -left-2 top-0 bottom-0 w-0.5 rounded-full ${isActive ? 'bg-forge-cyan' : 'bg-forge-crimson/20'}`} />
        <p className="text-sm text-gray-300 leading-relaxed pl-4 italic line-clamp-6">
          <TextHighlighter text={content} query={query} />
        </p>
      </div>

      {entities.length > 0 && (
        <div className="flex flex-wrap gap-2 pt-2">
          {entities.slice(0, 8).map((ent, i) => (
            <EntityBadge key={i} type={ent.type} text={ent.text} active={ent.text === activeNodeId} />
          ))}
        </div>
      )}

      <CardFooter id={id} content={content} source={source} section={section} page={page} author={author} entities={entities} domain={domain} onViewSource={onViewSource} />
    </div>
  );
};
