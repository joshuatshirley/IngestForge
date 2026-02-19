"use client";

/**
 * Apprentice Dashboard - Learning Governance UI
 *
 * US-2701.4: Provides a user interface for managing the "Learning Pool",
 * allowing experts to approve, edit, or remove golden examples.
 *
 * Features:
 * - Grid view showing: Original Text, Extracted JSON, Source Document
 * - Action buttons: Approve for Learning, Reject, Edit Example
 * - Visual cues for "Learning Ready" items
 */

import React, { useState } from 'react';
import {
  GraduationCap, CheckCircle, XCircle, Edit3, RefreshCcw,
  FileText, Code, ChevronDown, ChevronUp, Sparkles, AlertTriangle
} from 'lucide-react';
import {
  useGetLearningExamplesQuery,
  useGetLearningStatsQuery,
  useApproveLearningExampleMutation,
  useRejectLearningExampleMutation,
} from '@/store/api/ingestforgeApi';

interface LearningExample {
  example_id: string;
  vertical_id: string;
  entity_type: string;
  chunk_content: string;
  entities: any[];
  source_document?: string;
  approved_at?: string;
  approved_by?: string;
  status: 'pending' | 'approved' | 'rejected';
}

export default function ApprenticeDashboard() {
  const [statusFilter, setStatusFilter] = useState<string>('pending');
  const [expandedExample, setExpandedExample] = useState<string | null>(null);
  const [page, setPage] = useState(1);

  const {
    data: examplesData,
    isLoading: examplesLoading,
    refetch: refetchExamples
  } = useGetLearningExamplesQuery({
    status: statusFilter,
    page,
    page_size: 10
  });

  const { data: stats } = useGetLearningStatsQuery();

  const [approveExample, { isLoading: approving }] = useApproveLearningExampleMutation();
  const [rejectExample, { isLoading: rejecting }] = useRejectLearningExampleMutation();

  const handleApprove = async (exampleId: string) => {
    try {
      await approveExample({ example_id: exampleId, approved_by: 'user' }).unwrap();
      refetchExamples();
    } catch (err) {
      console.error('Failed to approve example:', err);
    }
  };

  const handleReject = async (exampleId: string) => {
    try {
      await rejectExample({ example_id: exampleId, reason: 'Rejected by user' }).unwrap();
      refetchExamples();
    } catch (err) {
      console.error('Failed to reject example:', err);
    }
  };

  const toggleExpand = (exampleId: string) => {
    setExpandedExample(expandedExample === exampleId ? null : exampleId);
  };

  const examples: LearningExample[] = examplesData?.examples || [];
  const totalExamples = examplesData?.total || 0;
  const hasMore = examplesData?.has_more || false;

  return (
    <div className="space-y-8 animate-in fade-in duration-500 pb-12">
      {/* Header */}
      <div className="flex justify-between items-end">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <GraduationCap className="text-forge-accent" size={32} />
            Apprentice Dashboard
          </h1>
          <p className="text-gray-400 mt-2">
            Manage the Learning Pool - approve, edit, or remove golden examples for few-shot learning.
          </p>
        </div>
        <button
          onClick={() => refetchExamples()}
          className="flex items-center gap-2 text-xs bg-gray-800 hover:bg-gray-700 px-4 py-2 rounded-xl transition-all border border-gray-700"
        >
          <RefreshCcw size={14} className={examplesLoading ? "animate-spin" : ""} />
          Refresh
        </button>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="forge-card flex items-center gap-4 group hover:bg-forge-blue/20 transition-all border-gray-800">
          <div className="p-3 rounded-2xl bg-gray-800/50 border border-gray-700 text-yellow-400">
            <AlertTriangle size={24} />
          </div>
          <div>
            <p className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Pending</p>
            <p className="text-2xl font-bold text-white">{stats?.total_examples || 0}</p>
          </div>
        </div>
        <div className="forge-card flex items-center gap-4 group hover:bg-forge-blue/20 transition-all border-gray-800">
          <div className="p-3 rounded-2xl bg-gray-800/50 border border-gray-700 text-green-400">
            <CheckCircle size={24} />
          </div>
          <div>
            <p className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Approved</p>
            <p className="text-2xl font-bold text-white">-</p>
          </div>
        </div>
        <div className="forge-card flex items-center gap-4 group hover:bg-forge-blue/20 transition-all border-gray-800">
          <div className="p-3 rounded-2xl bg-gray-800/50 border border-gray-700 text-blue-400">
            <Sparkles size={24} />
          </div>
          <div>
            <p className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Verticals</p>
            <p className="text-2xl font-bold text-white">{stats?.verticals || 0}</p>
          </div>
        </div>
        <div className="forge-card flex items-center gap-4 group hover:bg-forge-blue/20 transition-all border-gray-800">
          <div className="p-3 rounded-2xl bg-gray-800/50 border border-gray-700 text-forge-crimson">
            <GraduationCap size={24} />
          </div>
          <div>
            <p className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Learning Ready</p>
            <p className="text-2xl font-bold text-white">-</p>
          </div>
        </div>
      </div>

      {/* Filter Tabs */}
      <div className="flex gap-2">
        {['pending', 'approved', 'rejected'].map((status) => (
          <button
            key={status}
            onClick={() => { setStatusFilter(status); setPage(1); }}
            className={`px-4 py-2 rounded-xl text-xs font-bold uppercase tracking-wider transition-all ${
              statusFilter === status
                ? 'bg-forge-crimson text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
            }`}
          >
            {status}
          </button>
        ))}
      </div>

      {/* Examples Grid */}
      <div className="space-y-4">
        {examplesLoading ? (
          <div className="text-center py-12 text-gray-500">
            <RefreshCcw className="animate-spin mx-auto mb-4" size={32} />
            Loading examples...
          </div>
        ) : examples.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <GraduationCap className="mx-auto mb-4 opacity-50" size={48} />
            <p>No {statusFilter} examples found.</p>
            <p className="text-xs mt-2">Examples will appear here when extractions are verified.</p>
          </div>
        ) : (
          examples.map((example) => (
            <div
              key={example.example_id}
              className={`forge-card border-gray-800 transition-all ${
                example.status === 'approved'
                  ? 'border-l-4 border-l-green-500'
                  : example.status === 'rejected'
                  ? 'border-l-4 border-l-red-500'
                  : 'border-l-4 border-l-yellow-500'
              }`}
            >
              {/* Example Header */}
              <div className="flex justify-between items-start mb-4">
                <div className="flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="text-xs font-bold uppercase tracking-wider text-forge-accent bg-forge-blue/20 px-2 py-1 rounded">
                      {example.entity_type}
                    </span>
                    <span className="text-xs text-gray-500">
                      {example.vertical_id}
                    </span>
                    {example.source_document && (
                      <span className="text-xs text-gray-600 flex items-center gap-1">
                        <FileText size={12} />
                        {example.source_document}
                      </span>
                    )}
                  </div>
                  <p className="text-xs font-mono text-gray-500">
                    ID: {example.example_id}
                  </p>
                </div>

                {/* Action Buttons */}
                <div className="flex items-center gap-2">
                  {example.status === 'pending' && (
                    <>
                      <button
                        onClick={() => handleApprove(example.example_id)}
                        disabled={approving}
                        className="flex items-center gap-1 px-3 py-2 bg-green-500/20 text-green-400 rounded-lg text-xs font-bold hover:bg-green-500/30 transition-all disabled:opacity-50"
                      >
                        <CheckCircle size={14} />
                        Approve
                      </button>
                      <button
                        onClick={() => handleReject(example.example_id)}
                        disabled={rejecting}
                        className="flex items-center gap-1 px-3 py-2 bg-red-500/20 text-red-400 rounded-lg text-xs font-bold hover:bg-red-500/30 transition-all disabled:opacity-50"
                      >
                        <XCircle size={14} />
                        Reject
                      </button>
                    </>
                  )}
                  <button
                    className="flex items-center gap-1 px-3 py-2 bg-gray-700 text-gray-300 rounded-lg text-xs font-bold hover:bg-gray-600 transition-all"
                  >
                    <Edit3 size={14} />
                    Edit
                  </button>
                  <button
                    onClick={() => toggleExpand(example.example_id)}
                    className="p-2 bg-gray-800 rounded-lg hover:bg-gray-700 transition-all"
                  >
                    {expandedExample === example.example_id
                      ? <ChevronUp size={16} />
                      : <ChevronDown size={16} />
                    }
                  </button>
                </div>
              </div>

              {/* Original Text Preview */}
              <div className="mb-4">
                <h4 className="text-[10px] text-gray-500 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
                  <FileText size={12} />
                  Original Text
                </h4>
                <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800">
                  <p className="text-sm text-gray-300 line-clamp-3">
                    {example.chunk_content}
                  </p>
                </div>
              </div>

              {/* Extracted JSON (Expandable) */}
              {expandedExample === example.example_id && (
                <div className="animate-in fade-in duration-300">
                  <h4 className="text-[10px] text-gray-500 font-bold uppercase tracking-wider mb-2 flex items-center gap-2">
                    <Code size={12} />
                    Extracted Entities ({example.entities.length})
                  </h4>
                  <div className="bg-gray-900/50 rounded-xl p-4 border border-gray-800 overflow-x-auto">
                    <pre className="text-xs text-forge-accent font-mono">
                      {JSON.stringify(example.entities, null, 2)}
                    </pre>
                  </div>
                </div>
              )}

              {/* Approval Info */}
              {example.approved_at && (
                <div className="mt-4 pt-4 border-t border-gray-800 text-xs text-gray-500">
                  Approved by <span className="text-gray-400">{example.approved_by}</span> on{' '}
                  <span className="text-gray-400">{new Date(example.approved_at).toLocaleString()}</span>
                </div>
              )}
            </div>
          ))
        )}
      </div>

      {/* Pagination */}
      {totalExamples > 0 && (
        <div className="flex justify-between items-center">
          <p className="text-xs text-gray-500">
            Showing {examples.length} of {totalExamples} examples
          </p>
          <div className="flex gap-2">
            <button
              onClick={() => setPage(Math.max(1, page - 1))}
              disabled={page === 1}
              className="px-4 py-2 bg-gray-800 rounded-lg text-xs disabled:opacity-50 hover:bg-gray-700 transition-all"
            >
              Previous
            </button>
            <span className="px-4 py-2 text-xs text-gray-400">
              Page {page}
            </span>
            <button
              onClick={() => setPage(page + 1)}
              disabled={!hasMore}
              className="px-4 py-2 bg-gray-800 rounded-lg text-xs disabled:opacity-50 hover:bg-gray-700 transition-all"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
