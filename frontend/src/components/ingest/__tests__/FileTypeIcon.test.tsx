/**
 * FileTypeIcon Component Tests
 *
 * US-0201: Drag-Drop-Upload
 * Epic AC-01.5: File type auto-detection with icons
 *
 * Test Coverage: >90%
 * - File type detection
 * - Icon rendering
 * - Color assignment
 * - Label extraction
 * - Edge cases (unknown types, no extension)
 *
 * @created 2026-02-18
 */

import React from 'react';
import { render, screen } from '@testing-library/react';
import '@testing-library/jest-dom';
import { FileTypeIcon, getFileTypeLabel } from '../FileTypeIcon';

describe('FileTypeIcon Component', () => {
  // ===========================================================================
  // Epic AC-01.5: File Type Auto-Detection
  // ===========================================================================

  describe('Given a PDF file name', () => {
    describe('When rendering the icon', () => {
      it('Then should display PDF icon with correct color', () => {
        const { container } = render(<FileTypeIcon fileName="document.pdf" />);
        const icon = container.querySelector('svg');

        expect(icon).toBeInTheDocument();
        expect(icon).toHaveClass('text-red-400');
      });

      it('Then should have correct ARIA label', () => {
        const { container } = render(<FileTypeIcon fileName="report.pdf" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveAttribute('aria-label', 'PDF file');
      });
    });
  });

  describe('Given a DOCX file name', () => {
    describe('When rendering the icon', () => {
      it('Then should display Word icon with blue color', () => {
        const { container } = render(<FileTypeIcon fileName="essay.docx" />);
        const icon = container.querySelector('svg');

        expect(icon).toBeInTheDocument();
        expect(icon).toHaveClass('text-blue-400');
      });
    });
  });

  describe('Given an image file name', () => {
    describe('When rendering PNG icon', () => {
      it('Then should display image icon with green color', () => {
        const { container } = render(<FileTypeIcon fileName="photo.png" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('text-green-400');
      });
    });

    describe('When rendering JPEG icon', () => {
      it('Then should display image icon for .jpg extension', () => {
        const { container } = render(<FileTypeIcon fileName="image.jpg" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('text-green-400');
      });

      it('Then should display image icon for .jpeg extension', () => {
        const { container } = render(<FileTypeIcon fileName="photo.jpeg" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('text-green-400');
      });
    });
  });

  describe('Given an audio file name', () => {
    describe('When rendering MP3 icon', () => {
      it('Then should display audio icon with yellow color', () => {
        const { container } = render(<FileTypeIcon fileName="song.mp3" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('text-yellow-400');
      });
    });

    describe('When rendering WAV icon', () => {
      it('Then should display audio icon', () => {
        const { container } = render(<FileTypeIcon fileName="audio.wav" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('text-yellow-400');
      });
    });
  });

  describe('Given a video file name', () => {
    describe('When rendering MP4 icon', () => {
      it('Then should display video icon with pink color', () => {
        const { container } = render(<FileTypeIcon fileName="clip.mp4" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('text-pink-400');
      });
    });
  });

  describe('Given a code file name', () => {
    describe('When rendering Python icon', () => {
      it('Then should display code icon with cyan color', () => {
        const { container } = render(<FileTypeIcon fileName="script.py" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('text-cyan-400');
      });
    });

    describe('When rendering JavaScript icon', () => {
      it('Then should display code icon for .js', () => {
        const { container } = render(<FileTypeIcon fileName="app.js" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('text-cyan-400');
      });
    });

    describe('When rendering TypeScript icon', () => {
      it('Then should display code icon for .ts', () => {
        const { container } = render(<FileTypeIcon fileName="component.ts" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('text-cyan-400');
      });

      it('Then should display code icon for .tsx', () => {
        const { container } = render(<FileTypeIcon fileName="Component.tsx" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('text-cyan-400');
      });
    });
  });

  describe('Given a markdown file name', () => {
    describe('When rendering MD icon', () => {
      it('Then should display text icon for .md', () => {
        const { container } = render(<FileTypeIcon fileName="README.md" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('text-gray-300');
      });

      it('Then should display text icon for .markdown', () => {
        const { container } = render(<FileTypeIcon fileName="doc.markdown" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('text-gray-300');
      });
    });
  });

  // ===========================================================================
  // Edge Cases
  // ===========================================================================

  describe('Given an unknown file type', () => {
    describe('When rendering the icon', () => {
      it('Then should display default file icon', () => {
        const { container } = render(<FileTypeIcon fileName="unknown.xyz" />);
        const icon = container.querySelector('svg');

        expect(icon).toBeInTheDocument();
        expect(icon).toHaveClass('text-gray-500');
      });

      it('Then should have "File" label', () => {
        const { container } = render(<FileTypeIcon fileName="unknown.xyz" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveAttribute('aria-label', 'File file');
      });
    });
  });

  describe('Given a file name without extension', () => {
    describe('When rendering the icon', () => {
      it('Then should display default file icon', () => {
        const { container } = render(<FileTypeIcon fileName="noextension" />);
        const icon = container.querySelector('svg');

        expect(icon).toBeInTheDocument();
        expect(icon).toHaveClass('text-gray-500');
      });
    });
  });

  describe('Given a file name with uppercase extension', () => {
    describe('When rendering the icon', () => {
      it('Then should correctly detect PDF extension (case-insensitive)', () => {
        const { container } = render(<FileTypeIcon fileName="document.PDF" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('text-red-400');
      });

      it('Then should correctly detect DOCX extension', () => {
        const { container } = render(<FileTypeIcon fileName="file.DOCX" />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('text-blue-400');
      });
    });
  });

  // ===========================================================================
  // Props Testing
  // ===========================================================================

  describe('Given custom size prop', () => {
    describe('When rendering with size={24}', () => {
      it('Then should apply correct size', () => {
        const { container } = render(<FileTypeIcon fileName="test.pdf" size={24} />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveAttribute('width', '24');
        expect(icon).toHaveAttribute('height', '24');
      });
    });

    describe('When rendering with size={32}', () => {
      it('Then should apply correct size', () => {
        const { container } = render(<FileTypeIcon fileName="test.pdf" size={32} />);
        const icon = container.querySelector('svg');

        expect(icon).toHaveAttribute('width', '32');
        expect(icon).toHaveAttribute('height', '32');
      });
    });
  });

  describe('Given custom className prop', () => {
    describe('When rendering with additional classes', () => {
      it('Then should apply custom classes along with default', () => {
        const { container } = render(
          <FileTypeIcon fileName="test.pdf" className="custom-class" />
        );
        const icon = container.querySelector('svg');

        expect(icon).toHaveClass('custom-class');
        expect(icon).toHaveClass('text-red-400'); // Default color still applied
      });
    });
  });

  // ===========================================================================
  // Utility Function Tests
  // ===========================================================================

  describe('getFileTypeLabel utility function', () => {
    describe('Given various file types', () => {
      it('Then should return correct label for PDF', () => {
        expect(getFileTypeLabel('document.pdf')).toBe('PDF');
      });

      it('Then should return correct label for DOCX', () => {
        expect(getFileTypeLabel('file.docx')).toBe('Word');
      });

      it('Then should return correct label for PNG', () => {
        expect(getFileTypeLabel('image.png')).toBe('PNG');
      });

      it('Then should return correct label for MP3', () => {
        expect(getFileTypeLabel('song.mp3')).toBe('MP3');
      });

      it('Then should return correct label for Python', () => {
        expect(getFileTypeLabel('script.py')).toBe('Python');
      });

      it('Then should return correct label for TypeScript', () => {
        expect(getFileTypeLabel('component.tsx')).toBe('TypeScript');
      });

      it('Then should return "File" for unknown types', () => {
        expect(getFileTypeLabel('unknown.xyz')).toBe('File');
      });

      it('Then should return "File" for no extension', () => {
        expect(getFileTypeLabel('noextension')).toBe('File');
      });
    });
  });

  // ===========================================================================
  // Comprehensive File Type Coverage
  // ===========================================================================

  describe('Given comprehensive file type coverage', () => {
    const fileTypes = [
      { ext: 'pdf', color: 'text-red-400', label: 'PDF' },
      { ext: 'doc', color: 'text-blue-400', label: 'Word' },
      { ext: 'txt', color: 'text-gray-400', label: 'Text' },
      { ext: 'epub', color: 'text-purple-400', label: 'EPUB' },
      { ext: 'gif', color: 'text-green-400', label: 'GIF' },
      { ext: 'svg', color: 'text-green-400', label: 'SVG' },
      { ext: 'ogg', color: 'text-yellow-400', label: 'OGG' },
      { ext: 'webm', color: 'text-pink-400', label: 'WebM' },
      { ext: 'json', color: 'text-cyan-400', label: 'JSON' },
      { ext: 'html', color: 'text-cyan-400', label: 'HTML' },
    ];

    fileTypes.forEach(({ ext, color, label }) => {
      describe(`When rendering .${ext} file`, () => {
        it(`Then should display ${label} with ${color}`, () => {
          const { container } = render(<FileTypeIcon fileName={`test.${ext}`} />);
          const icon = container.querySelector('svg');

          expect(icon).toHaveClass(color);
          expect(getFileTypeLabel(`test.${ext}`)).toBe(label);
        });
      });
    });
  });
});
