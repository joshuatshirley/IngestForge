import axios, { AxiosProgressEvent } from 'axios';

/**
 * Upload Service
 *
 * Provides real-time upload progress tracking using Axios (XHR).
 * Replaces simulated progress in useBatchUpload hook.
 *
 * US-0201: Drag-Drop-Upload
 * AC-02.1: Individual file progress (via percentCompleted)
 * AC-02.3: Upload speed display (via speed calculation)
 * AC-02.4: Estimated time remaining (via eta calculation)
 */

export interface UploadProgressCallback {
  (progress: number, speed: number, eta: number): void;
}

export interface UploadResponse {
  job_id: string;
  filename: string;
  status: string;
}

/**
 * Upload a file with real-time progress tracking.
 *
 * @param file - File to upload
 * @param onProgress - Callback for progress updates
 * @returns Promise resolving to upload response
 */
export async function uploadFileWithProgress(
  file: File,
  onProgress: UploadProgressCallback
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('file', file);

  const startTime = Date.now();

  const response = await axios.post<UploadResponse>(
    'http://localhost:8000/v1/ingest/upload',
    formData,
    {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      onUploadProgress: (progressEvent: AxiosProgressEvent) => {
        if (!progressEvent.total) return;

        const percentCompleted = Math.round(
          (progressEvent.loaded * 100) / progressEvent.total
        );

        // Calculate speed (bytes/sec)
        const elapsedSeconds = (Date.now() - startTime) / 1000;
        const speed = elapsedSeconds > 0 ? progressEvent.loaded / elapsedSeconds : 0;

        // Calculate ETA (seconds)
        const remainingBytes = progressEvent.total - progressEvent.loaded;
        const eta = speed > 0 ? remainingBytes / speed : 0;

        onProgress(percentCompleted, speed, eta);
      },
    }
  );

  return response.data;
}
