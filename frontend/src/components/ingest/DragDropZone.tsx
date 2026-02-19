/**
 * DragDropZone Component
 *
 * US-0201: Drag-Drop-Upload
 * Epic AC-01: Drag-and-Drop Interface
 * Epic AC-03: Error Handling
 *
 * Modern drag-and-drop file upload zone with validation and visual feedback.
 *
 * JPL Compliance:
 * - Rule #2: Bounded loops (MAX_FILES, MAX_FILE_SIZE)
 * - Rule #7: Error handling with validation
 * - Rule #9: 100% TypeScript types
 *
 * @created 2026-02-18
 */

'use client';

import React, { useCallback, useMemo } from 'react';
import { useDropzone, type FileRejection, type DropzoneOptions } from 'react-dropzone';
import { Upload, AlertCircle, Folder } from 'lucide-react';
import { FileTypeIcon, getFileTypeLabel } from './FileTypeIcon';

// =============================================================================
// JPL Rule #2: Bounded Constants
// =============================================================================

/** Maximum number of files per upload batch (JPL Rule #2) */
const MAX_FILES = 100;

/** Maximum file size in bytes (100MB) (JPL Rule #2) */
const MAX_FILE_SIZE = 100 * 1024 * 1024; // 100MB

/**
 * Supported file extensions.
 *
 * Epic AC-03.1: Unsupported format warning
 */
const SUPPORTED_EXTENSIONS = [
  // Documents
  '.pdf',
  '.doc',
  '.docx',
  '.txt',
  '.md',
  '.markdown',
  '.epub',
  // Images
  '.png',
  '.jpg',
  '.jpeg',
  '.gif',
  '.svg',
  '.webp',
  // Audio
  '.mp3',
  '.wav',
  '.ogg',
  '.m4a',
  // Video
  '.mp4',
  '.webm',
  '.avi',
  // Code
  '.py',
  '.js',
  '.ts',
  '.tsx',
  '.jsx',
  '.json',
  '.html',
  '.css',
];

// =============================================================================
// Type Definitions (JPL Rule #9)
// =============================================================================

/**
 * File validation error type.
 *
 * Epic AC-03: Error Handling
 */
export interface FileValidationError {
  file: File;
  errors: string[];
}

/**
 * Props for DragDropZone component.
 *
 * Epic AC-01: Drag-and-Drop Interface
 */
export interface DragDropZoneProps {
  /** Callback when files are selected */
  onFilesSelected: (files: File[]) => void;
  /** Callback when validation errors occur */
  onValidationError?: (errors: FileValidationError[]) => void;
  /** Maximum number of files (default: 100) */
  maxFiles?: number;
  /** Maximum file size in bytes (default: 100MB) */
  maxSize?: number;
  /** Whether upload is currently in progress */
  isUploading?: boolean;
  /** Additional CSS classes */
  className?: string;
}

// =============================================================================
// Helper Functions
// =============================================================================

/**
 * Format file size for display.
 *
 * @param bytes - File size in bytes
 * @returns Formatted size string (e.g., "1.5 MB")
 */
function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`;
}

/**
 * Convert FileRejection to FileValidationError.
 *
 * Epic AC-03: Error Handling
 * JPL Rule #7: Input validation
 *
 * @param rejections - Rejected files from dropzone
 * @returns Validation errors
 */
function convertRejections(rejections: FileRejection[]): FileValidationError[] {
  return rejections.map((rejection) => ({
    file: rejection.file,
    errors: rejection.errors.map((err) => {
      // Convert error codes to user-friendly messages
      switch (err.code) {
        case 'file-too-large':
          return `File too large: ${formatFileSize(rejection.file.size)} (max ${formatFileSize(MAX_FILE_SIZE)})`;
        case 'file-invalid-type':
          return `Unsupported file type: ${rejection.file.name.split('.').pop()?.toUpperCase() || 'unknown'}`;
        case 'too-many-files':
          return `Too many files: maximum ${MAX_FILES} files per batch`;
        default:
          return err.message;
      }
    }),
  }));
}

// =============================================================================
// Main Component
// =============================================================================

/**
 * DragDropZone Component.
 *
 * Epic AC-01: Drag-and-Drop Interface
 * - AC-01.1: Drag-and-drop zone with visual feedback
 * - AC-01.2: Multi-file selection
 * - AC-01.3: Directory upload support
 * - AC-01.4: File type auto-detection
 *
 * Epic AC-03: Error Handling
 * - AC-03.1: Unsupported format warning
 * - AC-03.2: File too large warning
 *
 * @example
 * ```tsx
 * <DragDropZone
 *   onFilesSelected={(files) => handleUpload(files)}
 *   onValidationError={(errors) => showErrors(errors)}
 *   maxFiles={100}
 *   isUploading={false}
 * />
 * ```
 */
export function DragDropZone({
  onFilesSelected,
  onValidationError,
  maxFiles = MAX_FILES,
  maxSize = MAX_FILE_SIZE,
  isUploading = false,
  className = '',
}: DragDropZoneProps): JSX.Element {
  // ===========================================================================
  // Dropzone Configuration
  // ===========================================================================

  const onDrop = useCallback(
    (acceptedFiles: File[], fileRejections: FileRejection[]) => {
      // Epic AC-03: Error Handling
      if (fileRejections.length > 0 && onValidationError) {
        const errors = convertRejections(fileRejections);
        onValidationError(errors);
      }

      // Epic AC-01.2: Multi-file selection
      if (acceptedFiles.length > 0) {
        onFilesSelected(acceptedFiles);
      }
    },
    [onFilesSelected, onValidationError]
  );

  const dropzoneOptions: DropzoneOptions = useMemo(
    () => ({
      onDrop,
      maxFiles, // JPL Rule #2: Bounded file count
      maxSize, // JPL Rule #2: Bounded file size
      accept: {
        // Epic AC-03.1: Supported file types
        'application/pdf': ['.pdf'],
        'application/msword': ['.doc'],
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document': ['.docx'],
        'text/plain': ['.txt'],
        'text/markdown': ['.md', '.markdown'],
        'application/epub+zip': ['.epub'],
        'image/*': ['.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp'],
        'audio/*': ['.mp3', '.wav', '.ogg', '.m4a'],
        'video/*': ['.mp4', '.webm', '.avi'],
        'text/html': ['.html'],
        'text/css': ['.css'],
        'application/json': ['.json'],
        'text/javascript': ['.js'],
        'text/typescript': ['.ts', '.tsx'],
        'text/x-python': ['.py'],
      },
      disabled: isUploading,
      // Epic AC-01.3: Directory upload support (browser-dependent)
      // @ts-expect-error - directory is not in DropzoneOptions types but supported
      directory: true,
    }),
    [onDrop, maxFiles, maxSize, isUploading]
  );

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone(dropzoneOptions);

  // ===========================================================================
  // Render
  // ===========================================================================

  return (
    <div
      {...getRootProps()}
      className={`
        relative border-2 border-dashed rounded-xl transition-all duration-200 cursor-pointer
        ${isDragActive && !isDragReject ? 'border-forge-crimson bg-forge-crimson/10 scale-[1.02]' : ''}
        ${isDragReject ? 'border-red-500 bg-red-500/10' : 'border-gray-700'}
        ${isUploading ? 'opacity-50 cursor-not-allowed' : 'hover:border-forge-crimson/50'}
        ${className}
      `}
    >
      {/* Hidden file input (supports directory upload) */}
      <input
        {...getInputProps()}
        // @ts-expect-error - webkitdirectory is not in standard InputHTMLAttributes
        webkitdirectory=""
        // @ts-expect-error - directory is not in standard InputHTMLAttributes
        directory=""
      />

      {/* Drop zone content */}
      <div className="flex flex-col items-center justify-center py-20 px-6 text-center">
        {/* Icon */}
        <div
          className={`
            w-20 h-20 rounded-full flex items-center justify-center mb-6
            transition-all duration-200
            ${isDragActive && !isDragReject ? 'bg-forge-crimson/20 scale-110' : 'bg-gray-800/50'}
            ${isDragReject ? 'bg-red-500/20' : ''}
            ${isUploading ? 'animate-pulse' : ''}
          `}
        >
          {isDragReject ? (
            <AlertCircle size={40} className="text-red-500" />
          ) : (
            <Upload
              size={40}
              className={`
                transition-colors duration-200
                ${isDragActive ? 'text-forge-crimson' : 'text-gray-400'}
                ${isUploading ? 'text-blue-400' : ''}
              `}
            />
          )}
        </div>

        {/* Text */}
        <h3 className="text-xl font-bold text-gray-200 mb-2">
          {isUploading
            ? 'Upload in progress...'
            : isDragActive
            ? isDragReject
              ? 'Some files are not supported'
              : 'Drop files here'
            : 'Drag & drop files or click to browse'}
        </h3>

        <p className="text-sm text-gray-500 font-medium">
          {isDragReject
            ? 'Please check file types and sizes'
            : 'Supports: PDF, DOCX, MD, Images, Audio, Code'}
        </p>

        {/* File limits info */}
        <div className="mt-4 flex items-center gap-4 text-xs text-gray-600">
          <div className="flex items-center gap-1">
            <Folder size={14} />
            <span>Max {maxFiles} files</span>
          </div>
          <div className="h-3 w-px bg-gray-700" />
          <span>Max {formatFileSize(maxSize)} per file</span>
        </div>
      </div>

      {/* Upload overlay (when uploading) */}
      {isUploading && (
        <div className="absolute inset-0 bg-gray-900/80 backdrop-blur-sm rounded-xl flex items-center justify-center">
          <div className="text-center">
            <div className="w-12 h-12 border-4 border-forge-crimson border-t-transparent rounded-full animate-spin mx-auto mb-3" />
            <p className="text-sm text-gray-300 font-medium">Processing files...</p>
          </div>
        </div>
      )}
    </div>
  );
}
