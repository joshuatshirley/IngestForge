"use client";

import React from 'react';
import { 
  Library, 
  FileText, 
  Tag, 
  Bookmark, 
  Search, 
  Filter, 
  MoreVertical,
  Calendar,
  Layers,
  ChevronRight,
  Loader2
} from 'lucide-react';
import { 
  useGetDocumentsQuery, 
  useGetTagsQuery, 
  useGetBookmarksQuery,
  useGetStatusQuery 
} from '@/store/api/ingestforgeApi';
import { SyncStatusCard } from '@/components/library/SyncStatusCard';

export default function LibraryPage() {
  const [activeTab, setActiveTab] = React.useState<'documents' | 'bookmarks'>('documents');
  const { data: docData, isLoading: loadingDocs } = useGetDocumentsQuery();
  const { data: tagData } = useGetTagsQuery();
  const { data: bookmarkData } = useGetBookmarksQuery();
  const { data: status } = useGetStatusQuery();

  return (
    <div className="max-w-7xl mx-auto space-y-8 animate-in fade-in duration-500">
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Library size={32} className="text-forge-crimson" />
            Research Archive
          </h1>
          <p className="text-gray-400 mt-2">Browse and manage your ingested documents, annotations, and bookmarks.</p>
        </div>
      </div>

      <SyncStatusCard 
        localCount={status?.stats.totalChunks || 0} 
        remoteCount={1200} // Mocked until remote stats API ready
      />

      <div className="flex gap-8">
        {/* Sidebar: Tags & Filters */}
        <div className="w-64 space-y-6">
          <div className="forge-card p-6 space-y-6">
            <div className="flex items-center gap-2 text-forge-crimson font-bold text-xs uppercase tracking-widest">
              <Filter size={16} />
              Filter by Tag
            </div>
            <div className="flex flex-wrap gap-2">
              {tagData?.tags.map((tag: string) => (
                <span key={tag} className="text-[10px] font-bold px-3 py-1 bg-gray-800 text-gray-400 rounded-full border border-gray-700 hover:border-forge-crimson hover:text-white cursor-pointer transition-all">
                  #{tag}
                </span>
              ))}
            </div>
          </div>

          <div className="forge-card p-6">
            <div className="flex items-center gap-2 text-gray-500 font-bold text-xs uppercase tracking-widest mb-4">
              <Layers size={16} />
              Libraries
            </div>
            <div className="space-y-2">
              {['default_library', 'physics_2024', 'math_archive'].map((lib) => (
                <div key={lib} className="flex items-center justify-between text-xs text-gray-400 p-2 hover:bg-gray-800 rounded-lg cursor-pointer transition-all">
                  <span>{lib}</span>
                  <ChevronRight size={14} />
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 space-y-6">
          <div className="flex border-b border-gray-800 gap-8 px-2">
            <button 
              onClick={() => setActiveTab('documents')}
              className={`pb-4 text-sm font-bold uppercase tracking-widest transition-all relative ${
                activeTab === 'documents' ? 'text-white' : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              Documents
              {activeTab === 'documents' && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-forge-crimson" />}
            </button>
            <button 
              onClick={() => setActiveTab('bookmarks')}
              className={`pb-4 text-sm font-bold uppercase tracking-widest transition-all relative ${
                activeTab === 'bookmarks' ? 'text-white' : 'text-gray-500 hover:text-gray-300'
              }`}
            >
              Bookmarks
              {activeTab === 'bookmarks' && <div className="absolute bottom-0 left-0 right-0 h-0.5 bg-forge-crimson" />}
            </button>
          </div>

          <div className="grid grid-cols-1 gap-4">
            {activeTab === 'documents' ? (
              loadingDocs ? (
                <div className="h-64 flex items-center justify-center text-gray-500"><Loader2 className="animate-spin" /></div>
              ) : (
                docData?.documents.map((doc: any) => (
                  <div key={doc.id} className="forge-card flex items-center justify-between p-6 group hover:bg-forge-blue/20 transition-all border-gray-800">
                    <div className="flex items-center gap-6">
                      <div className="p-3 rounded-2xl bg-gray-800 text-blue-400 border border-gray-700">
                        <FileText size={24} />
                      </div>
                      <div>
                        <h3 className="font-bold text-gray-200 group-hover:text-forge-crimson transition-colors">{doc.name}</h3>
                        <div className="flex items-center gap-4 mt-1">
                          <span className="text-[10px] text-gray-500 font-bold uppercase">{doc.type} • {doc.chunks} Chunks</span>
                          <span className="text-[10px] text-gray-600 flex items-center gap-1">
                            <Calendar size={10} />
                            {doc.date}
                          </span>
                        </div>
                      </div>
                    </div>
                    <button className="p-2 text-gray-600 hover:text-white transition-colors">
                      <MoreVertical size={20} />
                    </button>
                  </div>
                ))
              )
            ) : (
              bookmarkData?.results.map((b: any) => (
                <div key={b.id} className="forge-card p-6 space-y-4 border-l-4 border-l-yellow-500 bg-yellow-500/5">
                  <div className="flex justify-between items-start">
                    <p className="text-sm text-gray-300 italic leading-relaxed">"{b.content}"</p>
                    <Bookmark size={18} className="text-yellow-500 fill-yellow-500" />
                  </div>
                  <div className="flex justify-between items-center text-[10px] font-bold uppercase tracking-widest text-gray-500">
                    <span>{b.source} • p. {b.page}</span>
                    <button className="text-forge-crimson hover:text-white transition-colors">View Context</button>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
