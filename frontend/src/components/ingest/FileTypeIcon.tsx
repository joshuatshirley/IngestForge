/**
 * FileTypeIcon Component
 *
 * US-0201: Drag-Drop-Upload
 * Epic AC-01.5: File type auto-detection with icons
 *
 * Displays appropriate icon for each supported file type.
 *
 * JPL Compliance:
 * - Rule #9: 100% TypeScript types
 *
 * @created 2026-02-18
 */

import React from 'react';
import {
  FileText,
  FileImage,
  FileAudio,
  FileVideo,
  FileCode,
  File,
  type LucideIcon,
} from 'lucide-react';

/**
 * Props for FileTypeIcon component.
 *
 * Epic AC-01.5: File type auto-detection with icons
 */
export interface FileTypeIconProps {
  /** File name or extension */
  fileName: string;
  /** Icon size in pixels (default: 20) */
  size?: number;
  /** Additional CSS classes */
  className?: string;
}

/**
 * File type configuration.
 * Maps extensions to icons and colors.
 */
interface FileTypeConfig {
  icon: LucideIcon;
  color: string;
  label: string;
}

/**
 * Supported file type mappings.
 *
 * Epic AC-01.5: Auto-detect file types
 */
const FILE_TYPE_MAP: Record<string, FileTypeConfig> = {
  // Documents
  pdf: { icon: FileText, color: 'text-red-400', label: 'PDF' },
  doc: { icon: FileText, color: 'text-blue-400', label: 'Word' },
  docx: { icon: FileText, color: 'text-blue-400', label: 'Word' },
  txt: { icon: FileText, color: 'text-gray-400', label: 'Text' },
  md: { icon: FileText, color: 'text-gray-300', label: 'Markdown' },
  markdown: { icon: FileText, color: 'text-gray-300', label: 'Markdown' },
  epub: { icon: FileText, color: 'text-purple-400', label: 'EPUB' },

  // Images
  png: { icon: FileImage, color: 'text-green-400', label: 'PNG' },
  jpg: { icon: FileImage, color: 'text-green-400', label: 'JPEG' },
  jpeg: { icon: FileImage, color: 'text-green-400', label: 'JPEG' },
  gif: { icon: FileImage, color: 'text-green-400', label: 'GIF' },
  svg: { icon: FileImage, color: 'text-green-400', label: 'SVG' },
  webp: { icon: FileImage, color: 'text-green-400', label: 'WebP' },

  // Audio
  mp3: { icon: FileAudio, color: 'text-yellow-400', label: 'MP3' },
  wav: { icon: FileAudio, color: 'text-yellow-400', label: 'WAV' },
  ogg: { icon: FileAudio, color: 'text-yellow-400', label: 'OGG' },
  m4a: { icon: FileAudio, color: 'text-yellow-400', label: 'M4A' },

  // Video
  mp4: { icon: FileVideo, color: 'text-pink-400', label: 'MP4' },
  webm: { icon: FileVideo, color: 'text-pink-400', label: 'WebM' },
  avi: { icon: FileVideo, color: 'text-pink-400', label: 'AVI' },

  // Code
  py: { icon: FileCode, color: 'text-cyan-400', label: 'Python' },
  js: { icon: FileCode, color: 'text-cyan-400', label: 'JavaScript' },
  ts: { icon: FileCode, color: 'text-cyan-400', label: 'TypeScript' },
  tsx: { icon: FileCode, color: 'text-cyan-400', label: 'TypeScript' },
  jsx: { icon: FileCode, color: 'text-cyan-400', label: 'JavaScript' },
  json: { icon: FileCode, color: 'text-cyan-400', label: 'JSON' },
  html: { icon: FileCode, color: 'text-cyan-400', label: 'HTML' },
  css: { icon: FileCode, color: 'text-cyan-400', label: 'CSS' },
};

/**
 * Extract file extension from filename.
 *
 * JPL Rule #7: Input validation
 *
 * @param fileName - File name with extension
 * @returns Lowercase extension without dot
 */
function getFileExtension(fileName: string): string {
  const parts = fileName.split('.');
  if (parts.length < 2) return '';
  return parts[parts.length - 1].toLowerCase();
}

/**
 * Get file type configuration by extension.
 *
 * @param fileName - File name with extension
 * @returns File type configuration or default
 */
function getFileTypeConfig(fileName: string): FileTypeConfig {
  const ext = getFileExtension(fileName);
  return FILE_TYPE_MAP[ext] || {
    icon: File,
    color: 'text-gray-500',
    label: 'File',
  };
}

/**
 * FileTypeIcon Component.
 *
 * Displays an icon for the given file type.
 *
 * Epic AC-01.5: File type auto-detection with icons
 *
 * @example
 * ```tsx
 * <FileTypeIcon fileName="document.pdf" size={24} />
 * <FileTypeIcon fileName="image.png" className="text-green-500" />
 * ```
 */
export function FileTypeIcon({
  fileName,
  size = 20,
  className = '',
}: FileTypeIconProps): JSX.Element {
  const config = getFileTypeConfig(fileName);
  const Icon = config.icon;

  return (
    <Icon
      size={size}
      className={`${config.color} ${className}`}
      aria-label={`${config.label} file`}
    />
  );
}

/**
 * Get file type label for display.
 *
 * Utility function for showing file type text.
 *
 * @param fileName - File name with extension
 * @returns Human-readable file type label
 */
export function getFileTypeLabel(fileName: string): string {
  const config = getFileTypeConfig(fileName);
  return config.label;
}
