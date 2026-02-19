import axios from 'axios';
import { uploadFileWithProgress } from '../uploadService';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

describe('uploadService: uploadFileWithProgress', () => {
  const file = new File(['dummy'], 'test.pdf', { type: 'application/pdf' });
  const onProgress = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
  });

  describe('Given a valid file and a success response', () => {
    it('When uploadFileWithProgress is called, Then it should call axios.post with correct parameters', async () => {
      mockedAxios.post.mockResolvedValueOnce({
        data: { job_id: '123', filename: 'test.pdf', status: 'pending' },
      });

      const result = await uploadFileWithProgress(file, onProgress);

      expect(mockedAxios.post).toHaveBeenCalledWith(
        'http://localhost:8000/v1/ingest/upload',
        expect.any(FormData),
        expect.objectContaining({
          headers: { 'Content-Type': 'multipart/form-data' },
          onUploadProgress: expect.any(Function),
        })
      );
      expect(result.job_id).toBe('123');
    });
  });

  describe('Given progress updates from axios', () => {
    it('When onUploadProgress is triggered, Then it should call the onProgress callback with calculated values', async () => {
      mockedAxios.post.mockImplementationOnce((url, data, config) => {
        // Manually trigger the progress callback
        if (config?.onUploadProgress) {
          config.onUploadProgress({
            loaded: 50,
            total: 100,
            bytes: 50,
            lengthComputable: true,
          } as any);
        }
        return Promise.resolve({ data: { job_id: '123' } });
      });

      await uploadFileWithProgress(file, onProgress);

      expect(onProgress).toHaveBeenCalledWith(
        50, // 50%
        expect.any(Number), // speed
        expect.any(Number)  // eta
      );
    });
  });

  describe('Given a network error', () => {
    it('When axios.post fails, Then it should throw the error', async () => {
      mockedAxios.post.mockRejectedValueOnce(new Error('Network Error'));

      await expect(uploadFileWithProgress(file, onProgress)).rejects.toThrow('Network Error');
    });
  });
});
