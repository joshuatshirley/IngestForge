import { renderHook, act, waitFor } from '@testing-library/react';
import { useBatchUpload } from '../useBatchUpload';
import * as uploadService from '@/services/uploadService';

// Mock the upload service
jest.mock('@/services/uploadService');

describe('useBatchUpload Hook', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  it('initializes with empty state', () => {
    const { result } = renderHook(() => useBatchUpload());
    
    expect(result.current.files).toEqual([]);
    expect(result.current.results).toEqual([]);
    expect(result.current.isUploading).toBe(false);
    expect(result.current.stats).toEqual({
      totalFiles: 0,
      successCount: 0,
      failureCount: 0,
      pendingCount: 0,
      uploadingCount: 0,
    });
  });

  it('adds files to upload queue', async () => {
    // Mock upload with a delay to ensure we catch the pending state if we were testing for it immediately,
    // but since the hook processes queue immediately in useEffect, it might flip to uploading fast.
    // However, we can check that it WAS added.
    (uploadService.uploadFileWithProgress as jest.Mock).mockReturnValue(new Promise(() => {})); // Never resolves for this test

    const { result } = renderHook(() => useBatchUpload());
    const file = new File(['content'], 'test.pdf', { type: 'application/pdf' });

    await act(async () => {
      result.current.uploadFiles([file]);
    });

    expect(result.current.files).toHaveLength(1);
    expect(result.current.files[0].file).toBe(file);
    // It might be 'uploading' because processQueue runs immediately
    expect(['pending', 'uploading']).toContain(result.current.files[0].status);
  });

  it('handles successful upload', async () => {
    // Mock successful upload
    (uploadService.uploadFileWithProgress as jest.Mock).mockResolvedValue({
      job_id: 'job-123',
      filename: 'test.pdf',
      status: 'success',
    });

    const { result } = renderHook(() => useBatchUpload());
    const file = new File(['content'], 'test.pdf', { type: 'application/pdf' });

    await act(async () => {
      result.current.uploadFiles([file]);
    });

    // Wait for upload to complete
    await waitFor(() => {
      expect(result.current.results).toHaveLength(1);
    });

    expect(result.current.results[0].success).toBe(true);
    expect(result.current.files[0].status).toBe('success');
  });

  it('handles failed upload and retries', async () => {
    // Mock failed upload then success
    (uploadService.uploadFileWithProgress as jest.Mock)
      .mockRejectedValueOnce(new Error('Network error'))
      .mockResolvedValueOnce({
        job_id: 'job-123',
        filename: 'test.pdf',
        status: 'success',
      });

    const { result } = renderHook(() => useBatchUpload());
    const file = new File(['content'], 'test.pdf', { type: 'application/pdf' });

    await act(async () => {
      result.current.uploadFiles([file]);
    });

    // Initial failure (retry 1) - wait for it to be pending again (retry scheduled)
    // or uploading again. The key is it shouldn't be 'error' yet.
    await waitFor(() => {
       const status = result.current.files[0].status;
       // It bounces between pending (queued for retry) and uploading (retrying)
       expect(status !== 'error').toBe(true);
    });

    // Eventually success
    await waitFor(() => {
      expect(result.current.files[0].status).toBe('success');
    }, { timeout: 3000 });
  });
});
