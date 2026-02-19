/**
 * TextHighlighter Component
 *
 * Efficiently wraps search terms in highlight spans.
 * Fulfills US-0203 AC for query term highlighting.
 *
 * JPL Compliance:
 * - Rule #2: Bounded regex execution.
 * - Rule #9: 100% type hints.
 */

import React, { useMemo } from 'react';

interface TextHighlighterProps {
  text: string;
  query: string;
  highlightClassName?: string;
}

export const TextHighlighter: React.FC<TextHighlighterProps> = ({
  text,
  query,
  highlightClassName = "bg-forge-crimson/30 text-forge-crimson font-bold rounded px-0.5"
}) => {
  const highlightedContent = useMemo(() => {
    if (!query.trim() || !text) return text;

    // JPL Rule #2: Limit query terms to prevent regex catastrophic backtracking
    const terms = query.trim().split(/\s+/).slice(0, 5);
    const pattern = new RegExp(`(${terms.join('|')})`, 'gi');
    
    const parts = text.split(pattern);

    return parts.map((part, i) => 
      pattern.test(part) ? (
        <mark key={i} className={highlightClassName}>
          {part}
        </mark>
      ) : (
        <span key={i}>{part}</span>
      )
    );
  }, [text, query, highlightClassName]);

  return <>{highlightedContent}</>;
};
