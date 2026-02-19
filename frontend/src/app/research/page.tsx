"use client";

import React, { useMemo, useCallback } from 'react';
import { useSelector } from 'react-redux';
import { PreviewModal } from '@/components/research/PreviewModal';
import { SourcedResult } from '@/components/chat/types';
import { SearchInput } from '@/components/research/SearchInput';
import { FilterBar } from '@/components/research/FilterBar';
import { VerticalBadge } from '@/components/research/VerticalBadge';
import { FoundryShell } from '@/components/research/FoundryShell';
import { useGetTimelineQuery } from '@/store/api/ingestforgeApi';
import { useResearchActions } from '@/hooks/useResearchActions';
import { RootState } from '@/store';
import { WorkbenchProvider, useWorkbenchContext } from '@/context/WorkbenchContext';

/**
 * Maximum search results to render.
 * JPL Rule #2: Performance safety bound.
 */
const MAX_RENDERED_RESULTS = 100;

function ResearchContent() {
  const { query, conversation, isSearching } = useSelector((state: RootState) => state.search);
  const [selectedPreview, setSelectedPreview] = React.useState<string | null>(null);
  const [viewMode, setViewMode] = React.useState<'grid' | 'timeline'>('grid');

  const { handleSearch, isChatting } = useResearchActions();
  const { setCurrentDocument } = useWorkbenchContext();

  // US-1103.1: Fetch timeline data
  const { data: timelineData, isLoading: isLoadingTimeline } = useGetTimelineQuery(
    { library: 'default' },
    { skip: viewMode !== 'timeline' }
  );

  const currentResults = useMemo(() => {
    const lastWithResults = [...conversation].reverse().find(m => m.results?.length);
    const results = (lastWithResults?.results || []) as SourcedResult[];
    return results.slice(0, MAX_RENDERED_RESULTS);
  }, [conversation]);

  const handleViewSource = useCallback((id: string) => {
    if (id.startsWith('doc_')) {
      setCurrentDocument(id);
    } else {
      setSelectedPreview(id);
    }
  }, [setCurrentDocument]);

  return (
    <div className="flex flex-col h-[calc(100vh-160px)] animate-in fade-in duration-500 gap-6">
      <PreviewModal chunkId={selectedPreview} onClose={() => setSelectedPreview(null)} />

      <div className="flex flex-col gap-2 flex-shrink-0">
        <div className="flex items-center justify-between px-4 max-w-4xl mx-auto w-full mb-2">
          <VerticalBadge />
          <div className="flex-1" />
        </div>
        <SearchInput onSearch={handleSearch} isSearching={isSearching || isChatting} />
        <div className="max-w-4xl mx-auto w-full px-4">
          <FilterBar />
        </div>
      </div>

      {/* US-1401.2.2: Synchronized Foundry Shell */}
      <FoundryShell 
        viewMode={viewMode}
        setViewMode={setViewMode}
        conversation={conversation}
        isSearching={isSearching}
        isChatting={isChatting}
        currentResults={currentResults}
        timelineData={timelineData}
        isLoadingTimeline={isLoadingTimeline}
        query={query}
        onViewSource={handleViewSource}
      />
    </div>
  );
}

export default function ResearchPage() {
  return (
    <WorkbenchProvider>
      <ResearchContent />
    </WorkbenchProvider>
  );
}
