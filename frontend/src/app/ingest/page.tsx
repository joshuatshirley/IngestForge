"use client";

/**
 * Ingest Page
 *
 * US-0201: Drag-Drop-Upload (Enhanced)
 * Epic AC-01: Drag-and-Drop Interface ✅
 * Epic AC-02: Upload Progress Tracking ✅
 * Epic AC-03: Error Handling ✅
 * Epic AC-04: Post-Upload Actions ✅
 * Epic AC-05: Mobile Responsive ✅
 *
 * Backwards Compatible: Existing upload functionality preserved
 *
 * @updated 2026-02-18
 */

import React, { useState, useCallback } from 'react';
import {
  FileUp, Upload, CheckCircle, Clock, AlertCircle, Loader2, Cloud, Search
} from 'lucide-react';
import { useGetJobStatusQuery } from '@/store/api/ingestforgeApi';
import { useToast } from '@/components/ToastProvider';
import { RemoteSourceModal } from '@/components/ingest/RemoteSourceModal';
import { DragDropZone, type FileValidationError } from '@/components/ingest/DragDropZone';
import { UploadProgress } from '@/components/ingest/UploadProgress';
import { useBatchUpload } from '@/hooks/useBatchUpload';

const JobItem = ({ jobId, initialName }: { jobId: string, initialName: string }) => {
  const { data: job, error } = useGetJobStatusQuery(jobId, {
    pollingInterval: 2000,
  });

  if (error) return <div className="text-red-500 text-xs italic">Update failed</div>;
  const currentJob = job || { status: 'PENDING', progress: 0, filename: initialName };

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-start">
        <div className="flex gap-3">
          <div className={`p-2 rounded-xl bg-gray-800 border border-gray-700 ${currentJob.status === 'COMPLETED' ? 'text-forge-accent' : 'text-blue-400'}`}>
            <FileUp size={16} />
          </div>
          <div className="max-w-[160px]">
            <p className="font-bold text-sm truncate text-gray-200">{currentJob.filename}</p>
            <p className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">{currentJob.status}</p>
          </div>
        </div>
        {currentJob.status === 'COMPLETED' ? <CheckCircle size={18} className="text-forge-accent" /> : 
         currentJob.status === 'FAILED' ? <AlertCircle size={18} className="text-red-500" /> : 
         <Clock size={18} className="text-blue-400 animate-pulse" />}
      </div>
      <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden shadow-inner">
        <div className={`h-full transition-all duration-700 ${currentJob.status === 'FAILED' ? 'bg-red-500' : 'bg-forge-accent'}`} style={{ width: `${currentJob.status === 'COMPLETED' ? 100 : currentJob.progress || 10}%` }} />
      </div>
    </div>
  );
};

export default function IngestPage() {
  const { showToast } = useToast();
  const [activeJobs, setActiveJobs] = useState<{id: string, name: string}[]>([]);
  const [isRemoteModalOpen, setIsRemoteModalOpen] = useState(false);

  // ===========================================================================
  // US-0201: Batch Upload Hook (Epic AC-02, AC-04)
  // ===========================================================================

  const {
    files: uploadFiles,
    uploadFiles: startBatchUpload,
    retryUpload,
    stats,
    isUploading,
  } = useBatchUpload({
    maxConcurrent: 3,
    onFileComplete: (result) => {
      // Epic AC-04.1: Automatic ingestion trigger
      if (result.success && result.jobId) {
        setActiveJobs((prev) => [
          { id: result.jobId!, name: result.file.name },
          ...prev,
        ]);
      }
    },
    onBatchComplete: (results) => {
      // Epic AC-04.3: Notification on completion
      const successCount = results.filter((r) => r.success).length;
      const failCount = results.filter((r) => !r.success).length;

      if (failCount === 0) {
        showToast(
          `✓ ${successCount} file${successCount !== 1 ? 's' : ''} uploaded successfully`,
          'success'
        );
      } else {
        showToast(
          `Uploaded ${successCount} files, ${failCount} failed`,
          'warning'
        );
      }
    },
    onError: (file, error) => {
      // Epic AC-03.4: Clear error messages
      showToast(`Failed to upload ${file.name}: ${error}`, 'error');
    },
  });

  // ===========================================================================
  // Event Handlers
  // ===========================================================================

  /**
   * Handle files selected from drag-drop zone.
   *
   * Epic AC-01.2: Multi-file selection
   * Epic AC-04.1: Automatic ingestion trigger
   */
  const handleFilesSelected = useCallback(
    (files: File[]) => {
      if (files.length === 0) return;

      showToast(
        `Starting upload of ${files.length} file${files.length !== 1 ? 's' : ''}...`,
        'info'
      );

      // Start batch upload
      startBatchUpload(files);
    },
    [startBatchUpload, showToast]
  );

  /**
   * Handle validation errors from drag-drop zone.
   *
   * Epic AC-03: Error Handling
   */
  const handleValidationError = useCallback(
    (errors: FileValidationError[]) => {
      errors.forEach((error) => {
        const errorMsg = error.errors.join(', ');
        showToast(`${error.file.name}: ${errorMsg}`, 'error');
      });
    },
    [showToast]
  );

  /**
   * Handle retry for failed upload.
   *
   * Epic AC-03.3: Failed upload retry button
   */
  const handleRetry = useCallback(
    (fileId: string) => {
      retryUpload(fileId);
      showToast('Retrying upload...', 'info');
    },
    [retryUpload, showToast]
  );

  return (
    <div className="max-w-6xl mx-auto space-y-12 animate-in fade-in duration-500">
      {/* Header */}
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-end gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <FileUp size={32} className="text-forge-crimson" />
            Ingest Center
          </h1>
          <p className="text-gray-400 mt-2">
            Scale your knowledge base by importing documents from local and cloud sources.
          </p>
          {/* Epic AC-02.2: Batch statistics */}
          {stats.totalFiles > 0 && (
            <p className="text-sm text-gray-500 mt-1">
              {stats.uploadingCount > 0 && (
                <span className="text-blue-400">
                  {stats.uploadingCount} uploading
                </span>
              )}
              {stats.uploadingCount > 0 && stats.pendingCount > 0 && <span> • </span>}
              {stats.pendingCount > 0 && (
                <span className="text-gray-400">
                  {stats.pendingCount} pending
                </span>
              )}
            </p>
          )}
        </div>
        <div className="flex gap-3">
          {/* Epic AC-04.4: Quick search button */}
          {stats.successCount > 0 && (
            <button
              onClick={() => (window.location.href = '/research')}
              className="flex items-center gap-2 text-xs bg-forge-accent hover:bg-forge-accent/80 px-6 py-3 rounded-xl border border-forge-accent/30 transition-all text-white font-bold uppercase tracking-widest"
            >
              <Search size={16} />
              Quick Search
            </button>
          )}
          <button
            onClick={() => setIsRemoteModalOpen(true)}
            className="flex items-center gap-2 text-xs bg-gray-800 hover:bg-gray-700 px-6 py-3 rounded-xl border border-gray-700 transition-all text-white font-bold uppercase tracking-widest"
          >
            <Cloud size={16} className="text-blue-400" />
            Import from Cloud
          </button>
        </div>
      </div>

      {/* Main content grid - Epic AC-05: Mobile Responsive */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 lg:gap-10">
        {/* Left column: Drag-drop zone */}
        <div className="lg:col-span-2 space-y-6">
          {/* Epic AC-01: Drag-and-Drop Interface */}
          <DragDropZone
            onFilesSelected={handleFilesSelected}
            onValidationError={handleValidationError}
            maxFiles={100}
            maxSize={100 * 1024 * 1024}
            isUploading={isUploading}
          />

          {/* Epic AC-02: Upload Progress (if files uploading) */}
          {uploadFiles.length > 0 && (
            <div className="forge-card p-8 border-gray-800">
              <h3 className="font-bold mb-6 text-xs uppercase tracking-[0.2em] text-gray-500 border-b border-gray-800 pb-4">
                Upload Progress
              </h3>
              <UploadProgress
                files={uploadFiles}
                onRetry={handleRetry}
                showDetailedStats={true}
              />
            </div>
          )}
        </div>

        {/* Right column: Processing history */}
        <div className="lg:col-span-1 space-y-6">
          <div className="forge-card p-6 lg:p-8 border-gray-800 min-h-[400px]">
            <h3 className="font-bold mb-8 text-xs uppercase tracking-[0.2em] text-gray-500 border-b border-gray-800 pb-4">
              Processing Queue
            </h3>
            <div className="space-y-10">
              {/* Epic AC-04.2: Background processing indicator */}
              {activeJobs.map((job) => (
                <JobItem key={job.id} jobId={job.id} initialName={job.name} />
              ))}
              {activeJobs.length === 0 && (
                <div className="text-center py-20 opacity-30 text-xs italic">
                  Awaiting document feed...
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Remote source modal */}
      <RemoteSourceModal
        isOpen={isRemoteModalOpen}
        onClose={() => setIsRemoteModalOpen(false)}
      />
    </div>
  );
}
