import React from 'react';
import { render, screen } from '@testing-library/react';
import { DragDropZone } from '../DragDropZone';

// Store the onDrop callback to trigger it manually
let lastOnDrop: any = null;

jest.mock('react-dropzone', () => ({
  useDropzone: ({ onDrop }: any) => {
    lastOnDrop = onDrop;
    return {
      getRootProps: () => ({ 'data-testid': 'dropzone-root' }),
      getInputProps: () => ({ 'data-testid': 'dropzone-input' }),
      isDragActive: false,
      isDragReject: false,
    };
  },
}));

describe('DragDropZone Component (Comprehensive)', () => {
  const mockOnFilesSelected = jest.fn();
  const mockOnValidationError = jest.fn();

  beforeEach(() => {
    jest.clearAllMocks();
    lastOnDrop = null;
  });

  describe('Given the component is rendered', () => {
    it('When visible, Then it should display the correct support text and limits', () => {
      render(
        <DragDropZone
          onFilesSelected={mockOnFilesSelected}
          onValidationError={mockOnValidationError}
          maxFiles={50}
        />
      );

      expect(screen.getByText(/Drag & drop files/i)).toBeInTheDocument();
      expect(screen.getByText(/Max 50 files/i)).toBeInTheDocument();
    });
  });

  describe('Given files are dropped', () => {
    it('When files are valid, Then it should call onFilesSelected', () => {
      render(
        <DragDropZone
          onFilesSelected={mockOnFilesSelected}
          onValidationError={mockOnValidationError}
        />
      );

      const files = [new File([''], 'test.pdf')];
      
      // Manually trigger the callback captured by the mock
      lastOnDrop(files, []);

      expect(mockOnFilesSelected).toHaveBeenCalledWith(files);
    });

    it('When files are rejected, Then it should call onValidationError', () => {
      render(
        <DragDropZone
          onFilesSelected={mockOnFilesSelected}
          onValidationError={mockOnValidationError}
        />
      );

      const rejections = [
        {
          file: new File([''], 'huge.pdf'),
          errors: [{ code: 'file-too-large', message: 'Too big' }]
        }
      ];
      
      lastOnDrop([], rejections);

      expect(mockOnValidationError).toHaveBeenCalled();
      const errorArg = mockOnValidationError.mock.calls[0][0][0];
      expect(errorArg.file.name).toBe('huge.pdf');
    });
  });
});
