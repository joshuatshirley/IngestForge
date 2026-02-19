import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import { UploadProgress } from '../UploadProgress';
import { UploadFile } from '../UploadProgress';

describe('UploadProgress Component', () => {
  const mockFiles: UploadFile[] = [
    {
      id: '1',
      file: new File([''], 'doc1.pdf'),
      progress: 45,
      status: 'uploading',
      speed: 1024 * 1024, // 1MB/s
      eta: 10,
    },
    {
      id: '2',
      file: new File([''], 'doc2.docx'),
      progress: 100,
      status: 'success',
    }
  ];

  describe('Given a list of uploading files', () => {
    it('When rendered, Then it should show individual file progress', () => {
      render(<UploadProgress files={mockFiles} />);
      
      expect(screen.getByText('doc1.pdf')).toBeInTheDocument();
      expect(screen.getByText('doc2.docx')).toBeInTheDocument();
      
      // Use getAllByText for overlapping labels (like speed which appears in multiple places)
      const speedLabels = screen.getAllByText(/1.0 MB\/s/);
      expect(speedLabels.length).toBeGreaterThan(0);
      
      const etaLabels = screen.getAllByText(/10s/);
      expect(etaLabels.length).toBeGreaterThan(0);
    });

    it('When a file has an error, Then it should show the error message and retry button', () => {
      const errorFiles: UploadFile[] = [{
        id: '3',
        file: new File([''], 'fail.txt'),
        progress: 0,
        status: 'error',
        errorMessage: 'Connection lost'
      }];
      
      const onRetry = jest.fn();
      render(<UploadProgress files={errorFiles} onRetry={onRetry} />);
      
      expect(screen.getByText('Connection lost')).toBeInTheDocument();
      const retryButton = screen.getByLabelText('Retry upload');
      fireEvent.click(retryButton);
      expect(onRetry).toHaveBeenCalledWith('3');
    });
  });

  describe('Given multiple files', () => {
    it('When showDetailedStats is true, Then it should show batch progress summary', () => {
      render(<UploadProgress files={mockFiles} showDetailedStats={true} />);
      
      expect(screen.getByText('Batch Progress')).toBeInTheDocument();
      // Match the specific string "1/2 files" which is rendered in a single span
      expect(screen.getByText(/1\/2 files/)).toBeInTheDocument();
    });
  });
});
