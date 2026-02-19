"use client";

import React from 'react';
import { SearchResultCard } from '../SearchResultCard';
import { TimelineView } from '../TimelineView';
import { EvidenceOverlay } from '../../pdf/EvidenceOverlay';
import { SourcedResult } from '../../chat/types';
import { useWorkbenchContext } from '../../../context/WorkbenchContext';

/**
 * US-1401.2.1: Document Viewer Pane
 * Center column of the research workbench.
 * JPL Rule #2: Performance safety bound.
 */
const MAX_LOCAL_RESULTS = 100;

interface DocumentViewerPaneProps {
  viewMode: 'grid' | 'timeline';
  currentResults: SourcedResult[];
  timelineData: any;
  isLoadingTimeline: boolean;
  query: string;
  onViewSource: (id: string) => void;
}

export const DocumentViewerPane: React.FC<DocumentViewerPaneProps> = ({
  viewMode,
  currentResults,
  timelineData,
  isLoadingTimeline,
  query,
  onViewSource,
}) => {
  const { currentDocumentId, activeNodeId, setActiveNode } = useWorkbenchContext();

  return (
    <div className="h-full flex flex-col overflow-hidden bg-black bg-opacity-10 border-r border-gray-800">
      <div className="p-4 border-b border-gray-800 flex items-center justify-between bg-forge-navy bg-opacity-30">
        <h2 className="text-xs font-bold uppercase tracking-widest text-gray-400">
          {currentDocumentId ? 'Source Evidence' : 'Search Results'}
        </h2>
        {currentDocumentId && (
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-forge-cyan font-mono truncate max-w-[150px]">
              Doc: {currentDocumentId}
            </span>
            <button 
              onClick={() => onViewSource('')} // Clear selection or similar
              className="text-[10px] text-gray-500 hover:text-white"
            >
              Close
            </button>
          </div>
        )}
      </div>
      <div className="flex-1 overflow-y-auto custom-scrollbar p-0">
        {currentDocumentId ? (
          <EvidenceOverlay 
            documentId={currentDocumentId} 
            selectedEntityId={activeNodeId}
            onHighlightClick={setActiveNode}
          />
        ) : (
          <div className="p-6">
            {viewMode === 'grid' ? (
              currentResults.length > 0 ? (
                <div className="flex flex-col gap-6 animate-in slide-in-from-bottom-4 duration-700">
                  {currentResults.slice(0, MAX_LOCAL_RESULTS).map((result) => (
                    <SearchResultCard 
                      key={result.id}
                      id={result.id}
                      content={result.content}
                      score={result.score}
                      source={result.metadata.source}
                      section={result.metadata.section_title}
                      page={result.metadata.page_start}
                      author={result.metadata.author_name}
                      entities={result.metadata.entities}
                      query={query}
                      onViewSource={() => onViewSource(result.id)}
                      domain={result.metadata.domain}
                    />
                  ))}
                </div>
              ) : (
                <div className="h-full flex items-center justify-center text-gray-600 italic text-sm mt-20">
                  No results to display. Use the search bar above.
                </div>
              )
            ) : (
              <TimelineView 
                events={timelineData?.events || []} 
                onViewSource={onViewSource}
                isLoading={isLoadingTimeline}
              />
            )}
          </div>
        )}
      </div>
    </div>
  );
};
