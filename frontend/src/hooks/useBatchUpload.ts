/**
 * useBatchUpload Hook
 *
 * US-0201: Drag-Drop-Upload
 * AC-02.1: Individual file progress bars (via status/progress state)
 * AC-02.2: Overall batch progress indicator (via stats object)
 * AC-03.3: Failed upload retry button (via retryUpload function)
 * AC-04.1: Automatic ingestion trigger (via onFileComplete callback)
 * AC-04.3: Notification on completion (via onBatchComplete callback)
 *
 * Manages batch file uploads with concurrent upload limiting, progress tracking,
 * and automatic retry logic.
 *
 * JPL Compliance:
 * - Rule #2: Bounded concurrency (MAX_CONCURRENT_UPLOADS)
 * - Rule #7: Error handling with retry logic
 * - Rule #9: 100% TypeScript types
 *
 * @created 2026-02-18
 */

'use client';

import { useState, useCallback, useRef, useEffect } from 'react';
import { uploadFileWithProgress, type UploadResponse } from '@/services/uploadService';
import type { UploadFile, UploadStatus } from '@/components/ingest/UploadProgress';

// =============================================================================
// JPL Rule #2: Bounded Constants
// =============================================================================

/** Maximum files per batch (JPL Rule #2) */
const MAX_BATCH_FILES = 100;

/** Maximum concurrent uploads (JPL Rule #2) */
const MAX_CONCURRENT_UPLOADS = 3;

/** Maximum retry attempts per file (JPL Rule #2) */
const MAX_RETRY_ATTEMPTS = 3;

/** Upload timeout in milliseconds (JPL Rule #7) */
const UPLOAD_TIMEOUT = 0; // Disable timeout for large files handled by Axios

// =============================================================================
// Type Definitions (JPL Rule #9)
// =============================================================================

/**
 * Upload result for a single file.
 *
 * Epic AC-04: Post-Upload Actions
 */
export interface UploadResult {
  /** File that was uploaded */
  file: File;
  /** Upload success status */
  success: boolean;
  /** Job ID if successful */
  jobId?: string;
  /** Error message if failed */
  error?: string;
}

/**
 * Batch upload statistics.
 *
 * Epic AC-02.2: Overall batch progress
 */
export interface BatchUploadStats {
  /** Total files in batch */
  totalFiles: number;
  /** Files successfully uploaded */
  successCount: number;
  /** Files that failed */
  failureCount: number;
  /** Files pending upload */
  pendingCount: number;
  /** Files currently uploading */
  uploadingCount: number;
}

/**
 * Options for useBatchUpload hook.
 */
export interface UseBatchUploadOptions {
  /** Maximum concurrent uploads (default: 3) */
  maxConcurrent?: number;
  /** Callback when batch upload completes */
  onBatchComplete?: (results: UploadResult[]) => void;
  /** Callback when individual file upload completes */
  onFileComplete?: (result: UploadResult) => void;
  /** Callback when upload error occurs */
  onError?: (file: File, error: string) => void;
}

/**
 * Internal upload queue item.
 */
interface QueueItem {
  id: string;
  file: File;
  retryCount: number;
  startTime?: number;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Generate unique ID for upload.
 *
 * @returns Unique upload ID
 */
function generateUploadId(): string {
  return `upload_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

// =============================================================================
// Main Hook
// =============================================================================

/**
 * useBatchUpload Hook
 *
 * Manages batch file uploads with:
 * - Concurrent upload limiting (Epic AC-02)
 * - Progress tracking (Epic AC-02)
 * - Automatic retry logic (Epic AC-03)
 * - Post-upload actions (Epic AC-04)
 *
 * @param options - Hook configuration options
 * @returns Upload state and control functions
 */
export function useBatchUpload({
  maxConcurrent = MAX_CONCURRENT_UPLOADS,
  onBatchComplete,
  onFileComplete,
  onError,
}: UseBatchUploadOptions = {}) {
  // ===========================================================================
  // State
  // ===========================================================================

  const [files, setFiles] = useState<UploadFile[]>([]);
  const [uploadQueue, setUploadQueue] = useState<QueueItem[]>([]);
  const [activeUploads, setActiveUploads] = useState<Set<string>>(new Set());
  const [uploadResults, setUploadResults] = useState<UploadResult[]>([]);

  // Refs for closures
  const filesRef = useRef<UploadFile[]>([]);
  const queueRef = useRef<QueueItem[]>([]);
  const activeRef = useRef<Set<string>>(new Set());

  // Keep refs in sync with state
  useEffect(() => {
    filesRef.current = files;
  }, [files]);

  useEffect(() => {
    queueRef.current = uploadQueue;
  }, [uploadQueue]);

  useEffect(() => {
    activeRef.current = activeUploads;
  }, [activeUploads]);

  // ===========================================================================
  // Upload Functions
  // ===========================================================================

  /**
   * Update file state.
   *
   * @param id - File ID
   * @param updates - Partial file updates
   */
  const updateFile = useCallback((id: string, updates: Partial<UploadFile>) => {
    setFiles((prev) =>
      prev.map((f) => (f.id === id ? { ...f, ...updates } : f))
    );
  }, []);

  /**
   * Process single file upload.
   *
   * Epic AC-02: Upload Progress Tracking (Real-time via Axios)
   * Epic AC-03: Error Handling with retry
   *
   * @param queueItem - Queue item to process
   */
  const processUpload = useCallback(
    async (queueItem: QueueItem) => {
      const { id, file, retryCount } = queueItem;

      try {
        // Update status to uploading
        updateFile(id, {
          status: 'uploading',
          progress: 0,
          speed: 0,
        });

        // Perform upload with real-time progress
        const result: UploadResponse = await uploadFileWithProgress(
          file,
          (progress, speed, eta) => {
            updateFile(id, { progress, speed, eta });
          }
        );

        // Update to success
        updateFile(id, {
          status: 'success',
          progress: 100,
          speed: 0,
          eta: 0,
        });

        // Record result
        const uploadResult: UploadResult = {
          file,
          success: true,
          jobId: result.job_id,
        };
        setUploadResults((prev) => [...prev, uploadResult]);

        // Callback
        if (onFileComplete) {
          onFileComplete(uploadResult);
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Upload failed';

        // Retry logic (JPL Rule #2: bounded retries)
        if (retryCount < MAX_RETRY_ATTEMPTS) {
          // Retry: add back to queue with incremented retry count
          setUploadQueue((prev) => [
            ...prev,
            { ...queueItem, retryCount: retryCount + 1 },
          ]);

          updateFile(id, {
            status: 'pending',
            errorMessage: `Retrying... (${retryCount + 1}/${MAX_RETRY_ATTEMPTS})`,
          });
        } else {
          // Max retries reached: mark as error
          updateFile(id, {
            status: 'error',
            errorMessage,
            progress: 0,
          });

          // Record result
          const uploadResult: UploadResult = {
            file,
            success: false,
            error: errorMessage,
          };
          setUploadResults((prev) => [...prev, uploadResult]);

          // Error callback
          if (onError) {
            onError(file, errorMessage);
          }
        }
      } finally {
        // Remove from active uploads
        setActiveUploads((prev) => {
          const next = new Set(prev);
          next.delete(id);
          return next;
        });
      }
    },
    [updateFile, onFileComplete, onError]
  );

  /**
   * Process upload queue.
   *
   * JPL Rule #2: Bounded concurrent uploads
   */
  const processQueue = useCallback(() => {
    const currentActive = activeRef.current.size;
    const queue = queueRef.current;

    // Check if we can process more uploads
    if (currentActive >= maxConcurrent || queue.length === 0) {
      return;
    }

    // Get next items to process (JPL Rule #2: bounded by maxConcurrent)
    const slotsAvailable = maxConcurrent - currentActive;
    const itemsToProcess = queue.slice(0, slotsAvailable);

    // Remove from queue
    setUploadQueue((prev) => prev.slice(slotsAvailable));

    // Add to active uploads and start processing
    itemsToProcess.forEach((item) => {
      setActiveUploads((prev) => new Set([...prev, item.id]));
      processUpload(item);
    });
  }, [maxConcurrent, processUpload]);

  // Process queue whenever it or active uploads change
  useEffect(() => {
    processQueue();
  }, [uploadQueue, activeUploads, processQueue]);

  // Check for batch completion
  useEffect(() => {
    const allFilesProcessed = files.length > 0 && files.every(
      (f) => f.status === 'success' || f.status === 'error'
    );

    if (allFilesProcessed && uploadResults.length === files.length && onBatchComplete) {
      onBatchComplete(uploadResults);
    }
  }, [files, uploadResults, onBatchComplete]);

  // ===========================================================================
  // Public API
  // ===========================================================================

  /**
   * Start uploading files.
   *
   * Epic AC-02: Upload Progress Tracking
   * JPL Rule #2: Bounded input processing
   *
   * @param filesToUpload - Files to upload
   */
  const uploadFiles = useCallback((filesToUpload: File[]) => {
    // JPL Rule #2: Enforce fixed upper bound on input processing
    const boundedFiles = filesToUpload.slice(0, MAX_BATCH_FILES);

    // Create upload file objects
    const newFiles: UploadFile[] = boundedFiles.map((file) => ({
      id: generateUploadId(),
      file,
      progress: 0,
      status: 'pending' as UploadStatus,
    }));

    // Add to files state
    setFiles((prev) => [...prev, ...newFiles]);

    // Add to queue
    const queueItems: QueueItem[] = newFiles.map((f) => ({
      id: f.id,
      file: f.file,
      retryCount: 0,
    }));
    setUploadQueue((prev) => [...prev, ...queueItems]);

    // Reset results
    setUploadResults([]);
  }, []);

  /**
   * Cancel upload.
   *
   * @param fileId - File ID to cancel
   */
  const cancelUpload = useCallback((fileId: string) => {
    // Remove from queue
    setUploadQueue((prev) => prev.filter((item) => item.id !== fileId));

    // Update file status
    updateFile(fileId, {
      status: 'error',
      errorMessage: 'Cancelled by user',
    });
  }, [updateFile]);

  /**
   * Retry failed upload.
   *
   * Epic AC-03.3: Failed upload retry button
   *
   * @param fileId - File ID to retry
   */
  const retryUpload = useCallback((fileId: string) => {
    const file = filesRef.current.find((f) => f.id === fileId);
    if (!file) return;

    // Reset file state
    updateFile(fileId, {
      status: 'pending',
      progress: 0,
      errorMessage: undefined,
    });

    // Add back to queue with reset retry count
    setUploadQueue((prev) => [
      ...prev,
      {
        id: file.id,
        file: file.file,
        retryCount: 0,
      },
    ]);
  }, [updateFile]);

  /**
   * Clear all completed/failed uploads.
   */
  const clearUploads = useCallback(() => {
    setFiles([]);
    setUploadQueue([]);
    setUploadResults([]);
  }, []);

  /**
   * Calculate batch statistics.
   *
   * Epic AC-02.2: Overall batch progress
   */
  const stats: BatchUploadStats = {
    totalFiles: files.length,
    successCount: files.filter((f) => f.status === 'success').length,
    failureCount: files.filter((f) => f.status === 'error').length,
    pendingCount: files.filter((f) => f.status === 'pending').length,
    uploadingCount: files.filter((f) => f.status === 'uploading').length,
  };

  return {
    /** Current upload files with progress */
    files,
    /** Batch upload statistics */
    stats,
    /** Upload results */
    results: uploadResults,
    /** Start uploading files */
    uploadFiles,
    /** Cancel specific upload */
    cancelUpload,
    /** Retry failed upload */
    retryUpload,
    /** Clear all uploads */
    clearUploads,
    /** Whether any uploads are in progress */
    isUploading: activeUploads.size > 0,
  };
}
