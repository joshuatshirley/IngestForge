/**
 * UploadProgress Component
 *
 * US-0201: Drag-Drop-Upload
 * Epic AC-02: Upload Progress Tracking
 * Epic AC-03: Error Handling
 *
 * Displays detailed progress for batch file uploads with individual file tracking.
 *
 * JPL Compliance:
 * - Rule #2: Bounded loops (iterate over files array)
 * - Rule #7: Error handling for failed uploads
 * - Rule #9: 100% TypeScript types
 *
 * @created 2026-02-18
 */

'use client';

import React, { useMemo } from 'react';
import {
  CheckCircle,
  XCircle,
  Loader2,
  Clock,
  AlertCircle,
  RotateCcw,
  Pause,
  Play,
} from 'lucide-react';
import { FileTypeIcon } from './FileTypeIcon';

// =============================================================================
// Type Definitions (JPL Rule #9)
// =============================================================================

/**
 * Upload status for individual files.
 *
 * Epic AC-02: Upload Progress Tracking
 */
export type UploadStatus = 'pending' | 'uploading' | 'success' | 'error' | 'paused';

/**
 * Individual file upload state.
 *
 * Epic AC-02: Upload Progress Tracking
 */
export interface UploadFile {
  /** Unique file identifier */
  id: string;
  /** Original File object */
  file: File;
  /** Upload progress (0-100) */
  progress: number;
  /** Current upload status */
  status: UploadStatus;
  /** Error message if status is 'error' */
  errorMessage?: string;
  /** Upload speed in bytes/second */
  speed?: number;
  /** Estimated time remaining in seconds */
  eta?: number;
}

/**
 * Overall batch progress statistics.
 *
 * Epic AC-02.2: Overall batch progress indicator
 */
export interface BatchProgress {
  /** Total number of files */
  totalFiles: number;
  /** Number of completed files */
  completedFiles: number;
  /** Number of failed files */
  failedFiles: number;
  /** Overall progress percentage (0-100) */
  overallProgress: number;
  /** Average upload speed in bytes/second */
  avgSpeed: number;
  /** Total bytes uploaded */
  uploadedBytes: number;
  /** Total bytes to upload */
  totalBytes: number;
}

/**
 * Props for UploadProgress component.
 *
 * Epic AC-02: Upload Progress Tracking
 */
export interface UploadProgressProps {
  /** Array of files being uploaded */
  files: UploadFile[];
  /** Callback to retry failed upload */
  onRetry?: (fileId: string) => void;
  /** Callback to pause upload (optional) */
  onPause?: (fileId: string) => void;
  /** Callback to resume paused upload (optional) */
  onResume?: (fileId: string) => void;
  /** Callback to cancel upload */
  onCancel?: (fileId: string) => void;
  /** Show detailed stats */
  showDetailedStats?: boolean;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Format file size for display.
 *
 * @param bytes - Size in bytes
 * @returns Formatted size string
 */
function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B';
  const k = 1024;
  const sizes = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${(bytes / Math.pow(k, i)).toFixed(1)} ${sizes[i]}`;
}

/**
 * Format upload speed for display.
 *
 * Epic AC-02.3: Upload speed display (MB/s)
 *
 * @param bytesPerSecond - Speed in bytes/second
 * @returns Formatted speed string
 */
function formatSpeed(bytesPerSecond: number): string {
  return `${formatBytes(bytesPerSecond)}/s`;
}

/**
 * Format estimated time remaining.
 *
 * Epic AC-02.4: Estimated time remaining
 *
 * @param seconds - Time in seconds
 * @returns Formatted time string
 */
function formatETA(seconds: number): string {
  if (seconds < 60) return `${Math.ceil(seconds)}s`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)}m`;
  return `${Math.ceil(seconds / 3600)}h`;
}

/**
 * Calculate batch progress statistics.
 *
 * Epic AC-02.2: Overall batch progress indicator
 * JPL Rule #2: Bounded iteration over files array
 *
 * @param files - Array of upload files
 * @returns Batch progress statistics
 */
function calculateBatchProgress(files: UploadFile[]): BatchProgress {
  const totalFiles = files.length;
  const completedFiles = files.filter((f) => f.status === 'success').length;
  const failedFiles = files.filter((f) => f.status === 'error').length;

  const totalBytes = files.reduce((sum, f) => sum + f.file.size, 0);
  const uploadedBytes = files.reduce(
    (sum, f) => sum + (f.file.size * f.progress) / 100,
    0
  );

  const overallProgress = totalBytes > 0 ? (uploadedBytes / totalBytes) * 100 : 0;

  // Calculate average speed from files currently uploading
  const uploadingFiles = files.filter((f) => f.status === 'uploading' && f.speed);
  const avgSpeed =
    uploadingFiles.length > 0
      ? uploadingFiles.reduce((sum, f) => sum + (f.speed || 0), 0) / uploadingFiles.length
      : 0;

  return {
    totalFiles,
    completedFiles,
    failedFiles,
    overallProgress,
    avgSpeed,
    uploadedBytes,
    totalBytes,
  };
}

/**
 * Get status icon for upload status.
 *
 * @param status - Upload status
 * @returns Icon component
 */
function getStatusIcon(status: UploadStatus): JSX.Element {
  switch (status) {
    case 'success':
      return <CheckCircle size={18} className="text-forge-accent" />;
    case 'error':
      return <XCircle size={18} className="text-red-500" />;
    case 'uploading':
      return <Loader2 size={18} className="text-blue-400 animate-spin" />;
    case 'paused':
      return <Pause size={18} className="text-yellow-400" />;
    case 'pending':
    default:
      return <Clock size={18} className="text-gray-500" />;
  }
}

// =============================================================================
// Sub-Components
// =============================================================================

/**
 * Individual file upload item.
 *
 * Epic AC-02.1: Individual file progress bars
 */
interface FileItemProps {
  file: UploadFile;
  onRetry?: (fileId: string) => void;
  onPause?: (fileId: string) => void;
  onResume?: (fileId: string) => void;
  onCancel?: (fileId: string) => void;
}

function FileItem({ file, onRetry, onPause, onResume, onCancel }: FileItemProps): JSX.Element {
  return (
    <div className="space-y-3">
      {/* File info and status */}
      <div className="flex justify-between items-start gap-3">
        <div className="flex gap-3 flex-1 min-w-0">
          {/* File type icon */}
          <div className="p-2 rounded-xl bg-gray-800 border border-gray-700 flex-shrink-0">
            <FileTypeIcon fileName={file.file.name} size={16} />
          </div>

          {/* File details */}
          <div className="flex-1 min-w-0">
            <p className="font-bold text-sm truncate text-gray-200" title={file.file.name}>
              {file.file.name}
            </p>
            <div className="flex items-center gap-2 mt-1">
              <p className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">
                {file.status}
              </p>
              {file.speed && file.status === 'uploading' && (
                <>
                  <span className="text-gray-700">•</span>
                  <p className="text-[10px] text-gray-500">
                    {formatSpeed(file.speed)}
                  </p>
                </>
              )}
              {file.eta && file.status === 'uploading' && (
                <>
                  <span className="text-gray-700">•</span>
                  <p className="text-[10px] text-gray-500">
                    {formatETA(file.eta)} remaining
                  </p>
                </>
              )}
            </div>
            {/* Error message */}
            {file.errorMessage && (
              <p className="text-xs text-red-400 mt-1 flex items-center gap-1">
                <AlertCircle size={12} />
                {file.errorMessage}
              </p>
            )}
          </div>
        </div>

        {/* Status icon and actions */}
        <div className="flex items-center gap-2 flex-shrink-0">
          {/* Retry button for failed uploads */}
          {file.status === 'error' && onRetry && (
            <button
              onClick={() => onRetry(file.id)}
              className="p-1.5 rounded-lg hover:bg-gray-800 transition-colors"
              aria-label="Retry upload"
            >
              <RotateCcw size={16} className="text-gray-400 hover:text-forge-accent" />
            </button>
          )}

          {/* Pause/Resume buttons (optional) */}
          {file.status === 'uploading' && onPause && (
            <button
              onClick={() => onPause(file.id)}
              className="p-1.5 rounded-lg hover:bg-gray-800 transition-colors"
              aria-label="Pause upload"
            >
              <Pause size={16} className="text-gray-400 hover:text-yellow-400" />
            </button>
          )}
          {file.status === 'paused' && onResume && (
            <button
              onClick={() => onResume(file.id)}
              className="p-1.5 rounded-lg hover:bg-gray-800 transition-colors"
              aria-label="Resume upload"
            >
              <Play size={16} className="text-gray-400 hover:text-green-400" />
            </button>
          )}

          {/* Status icon */}
          {getStatusIcon(file.status)}
        </div>
      </div>

      {/* Progress bar */}
      <div className="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden shadow-inner">
        <div
          className={`h-full transition-all duration-700 ${
            file.status === 'error' ? 'bg-red-500' : 'bg-forge-accent'
          }`}
          style={{ width: `${file.progress}%` }}
        />
      </div>
    </div>
  );
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * UploadProgress Component.
 *
 * Epic AC-02: Upload Progress Tracking
 * - AC-02.1: Individual file progress bars
 * - AC-02.2: Overall batch progress indicator
 * - AC-02.3: Upload speed display (MB/s)
 * - AC-02.4: Estimated time remaining
 * - AC-02.5: Pause/Resume capability (optional)
 *
 * Epic AC-03: Error Handling
 * - AC-03.3: Failed upload retry button
 * - AC-03.4: Clear error messages
 *
 * @example
 * ```tsx
 * <UploadProgress
 *   files={uploadFiles}
 *   onRetry={(id) => retryUpload(id)}
 *   showDetailedStats={true}
 * />
 * ```
 */
export function UploadProgress({
  files,
  onRetry,
  onPause,
  onResume,
  onCancel,
  showDetailedStats = true,
}: UploadProgressProps): JSX.Element {
  // Calculate batch statistics
  const batchProgress = useMemo(() => calculateBatchProgress(files), [files]);

  // No files to display
  if (files.length === 0) {
    return (
      <div className="text-center py-20 opacity-30 text-xs italic">
        No active uploads...
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Overall batch progress (Epic AC-02.2) */}
      {showDetailedStats && files.length > 1 && (
        <div className="p-4 bg-gray-800/50 rounded-lg border border-gray-700">
          <div className="flex justify-between items-center mb-3">
            <h4 className="text-xs font-bold uppercase tracking-widest text-gray-400">
              Batch Progress
            </h4>
            <span className="text-sm font-bold text-gray-300">
              {Math.round(batchProgress.overallProgress)}%
            </span>
          </div>

          {/* Overall progress bar */}
          <div className="w-full bg-gray-900 h-2 rounded-full overflow-hidden mb-3">
            <div
              className="h-full bg-gradient-to-r from-forge-crimson to-forge-accent transition-all duration-500"
              style={{ width: `${batchProgress.overallProgress}%` }}
            />
          </div>

          {/* Batch statistics */}
          <div className="flex items-center justify-between text-xs text-gray-500">
            <div className="flex items-center gap-4">
              <span>
                {batchProgress.completedFiles}/{batchProgress.totalFiles} files
              </span>
              {batchProgress.failedFiles > 0 && (
                <span className="text-red-400">{batchProgress.failedFiles} failed</span>
              )}
            </div>
            <div className="flex items-center gap-4">
              {batchProgress.avgSpeed > 0 && (
                <span>{formatSpeed(batchProgress.avgSpeed)}</span>
              )}
              <span>
                {formatBytes(batchProgress.uploadedBytes)} / {formatBytes(batchProgress.totalBytes)}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Individual file items (JPL Rule #2: bounded iteration) */}
      <div className="space-y-10">
        {files.map((file) => (
          <FileItem
            key={file.id}
            file={file}
            onRetry={onRetry}
            onPause={onPause}
            onResume={onResume}
            onCancel={onCancel}
          />
        ))}
      </div>
    </div>
  );
}
