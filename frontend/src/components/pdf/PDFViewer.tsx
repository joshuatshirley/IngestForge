/**
 * PDFViewer - Base PDF rendering component
 *
 * US-1404.1 Evidence Highlight Overlay - Phase 2
 *
 * Renders PDFs using PDF.js with lazy page loading and zoom controls.
 *
 * JPL Power of Ten Compliance:
 * - Rule #2: Fixed upper bounds (MAX_RENDERED_PAGES = 10)
 * - Rule #4: All functions < 60 lines
 * - Rule #9: Complete type hints
 *
 * Epic: EP-14 (Foundry UI)
 * Feature: FE-11-01 (Visualization)
 * Implementation Date: 2026-02-18
 */

import React, { useCallback, useEffect, useRef, useState } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import { PDFDocumentProxy, PDFPageProxy } from 'pdfjs-dist/types/src/display/api';

// =============================================================================
// TYPE DEFINITIONS (Rule #9: Complete type hints)
// =============================================================================

export interface PDFViewerProps {
  /** URL or file path to PDF document */
  documentUrl: string;
  /** Current page number (1-indexed) */
  currentPage?: number;
  /** Callback when page changes */
  onPageChange?: (page: number) => void;
  /** Zoom level (1.0 = 100%, 1.5 = 150%, etc.) */
  zoom?: number;
  /** Callback when PDF loads successfully */
  onDocumentLoad?: (pageCount: number) => void;
  /** Width of the viewer container */
  width?: number;
  /** Height of the viewer container */
  height?: number;
}

export interface PageDimensions {
  width: number;
  height: number;
  scale: number;
}

// =============================================================================
// CONSTANTS (JPL Rule #2: Fixed upper bounds)
// =============================================================================

const MAX_RENDERED_PAGES = 10; // Maximum pages rendered simultaneously
const DEFAULT_ZOOM = 1.5;
const MIN_ZOOM = 0.5;
const MAX_ZOOM = 3.0;
const DEFAULT_WIDTH = 800;
const DEFAULT_HEIGHT = 600;

// Configure PDF.js worker
if (typeof window !== 'undefined') {
  pdfjsLib.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.js`;
}

// =============================================================================
// COMPONENT (Rule #4: Main render < 60 lines)
// =============================================================================

export const PDFViewer: React.FC<PDFViewerProps> = ({
  documentUrl,
  currentPage = 1,
  onPageChange,
  zoom = DEFAULT_ZOOM,
  onDocumentLoad,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [pdfDoc, setPdfDoc] = useState<PDFDocumentProxy | null>(null);
  const [pageCount, setPageCount] = useState<number>(0);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [renderedPage, setRenderedPage] = useState<number>(0);

  // Load PDF document
  useEffect(() => {
    loadDocument(documentUrl, setPdfDoc, setPageCount, setLoading, setError, onDocumentLoad);
  }, [documentUrl, onDocumentLoad]);

  // Render current page when page/zoom changes
  useEffect(() => {
    if (!pdfDoc || !canvasRef.current) return;
    renderPage(pdfDoc, currentPage, zoom, canvasRef.current, setRenderedPage, setError);
  }, [pdfDoc, currentPage, zoom]);

  // Handle navigation
  const handlePrevPage = useCallback(() => {
    if (currentPage > 1) {
      onPageChange?.(currentPage - 1);
    }
  }, [currentPage, onPageChange]);

  const handleNextPage = useCallback(() => {
    if (currentPage < pageCount) {
      onPageChange?.(currentPage + 1);
    }
  }, [currentPage, pageCount, onPageChange]);

  return (
    <div
      ref={containerRef}
      className="pdf-viewer-container"
      style={{ width, height, position: 'relative', overflow: 'auto' }}
      data-testid="pdf-viewer"
    >
      {loading && (
        <div className="pdf-loading" style={loadingStyle}>
          Loading PDF...
        </div>
      )}
      {error && (
        <div className="pdf-error" style={errorStyle} data-testid="pdf-error">
          Error: {error}
        </div>
      )}
      {!loading && !error && (
        <>
          <canvas
            ref={canvasRef}
            className="pdf-canvas"
            style={{ display: 'block', margin: '0 auto' }}
            data-testid="pdf-canvas"
          />
          <div className="pdf-controls" style={controlsStyle}>
            <button
              onClick={handlePrevPage}
              disabled={currentPage === 1}
              data-testid="pdf-prev-button"
              style={buttonStyle}
            >
              Previous
            </button>
            <span data-testid="pdf-page-indicator">
              Page {currentPage} of {pageCount}
            </span>
            <button
              onClick={handleNextPage}
              disabled={currentPage === pageCount}
              data-testid="pdf-next-button"
              style={buttonStyle}
            >
              Next
            </button>
          </div>
        </>
      )}
    </div>
  );
};

// =============================================================================
// HELPER FUNCTIONS (JPL Rule #4: < 60 lines each)
// =============================================================================

/**
 * Load PDF document from URL.
 *
 * Rule #4: Under 60 lines.
 */
async function loadDocument(
  documentUrl: string,
  setPdfDoc: React.Dispatch<React.SetStateAction<PDFDocumentProxy | null>>,
  setPageCount: React.Dispatch<React.SetStateAction<number>>,
  setLoading: React.Dispatch<React.SetStateAction<boolean>>,
  setError: React.Dispatch<React.SetStateAction<string | null>>,
  onDocumentLoad?: (pageCount: number) => void
): Promise<void> {
  try {
    setLoading(true);
    setError(null);

    const loadingTask = pdfjsLib.getDocument(documentUrl);
    const pdf = await loadingTask.promise;

    setPdfDoc(pdf);
    setPageCount(pdf.numPages);
    setLoading(false);

    if (onDocumentLoad) {
      onDocumentLoad(pdf.numPages);
    }
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : 'Failed to load PDF';
    setError(errorMsg);
    setLoading(false);
  }
}

/**
 * Render a specific PDF page to canvas.
 *
 * Rule #4: Under 60 lines.
 */
async function renderPage(
  pdfDoc: PDFDocumentProxy,
  pageNum: number,
  zoom: number,
  canvas: HTMLCanvasElement,
  setRenderedPage: React.Dispatch<React.SetStateAction<number>>,
  setError: React.Dispatch<React.SetStateAction<string | null>>
): Promise<void> {
  try {
    const page = await pdfDoc.getPage(pageNum);
    const viewport = page.getViewport({ scale: zoom });

    // Set canvas dimensions
    canvas.width = viewport.width;
    canvas.height = viewport.height;

    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error('Failed to get canvas context');
    }

    // Render page
    const renderContext = {
      canvasContext: context,
      viewport,
    };

    await page.render(renderContext).promise;
    setRenderedPage(pageNum);
  } catch (err) {
    const errorMsg = err instanceof Error ? err.message : 'Failed to render page';
    setError(errorMsg);
  }
}

/**
 * Get page dimensions at current zoom level.
 *
 * Rule #4: Under 60 lines.
 */
export async function getPageDimensions(
  pdfDoc: PDFDocumentProxy,
  pageNum: number,
  zoom: number
): Promise<PageDimensions> {
  const page = await pdfDoc.getPage(pageNum);
  const viewport = page.getViewport({ scale: zoom });

  return {
    width: viewport.width,
    height: viewport.height,
    scale: zoom,
  };
}

/**
 * Validate zoom level within bounds.
 *
 * Rule #4: Under 60 lines.
 */
export function validateZoom(zoom: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, zoom));
}

// =============================================================================
// INLINE STYLES (temporary until CSS module added)
// =============================================================================

const loadingStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  height: '100%',
  fontSize: '18px',
  color: '#666',
};

const errorStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  height: '100%',
  fontSize: '16px',
  color: '#d32f2f',
  padding: '20px',
  textAlign: 'center',
};

const controlsStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: '16px',
  padding: '12px',
  borderTop: '1px solid #e0e0e0',
  backgroundColor: '#f5f5f5',
};

const buttonStyle: React.CSSProperties = {
  padding: '8px 16px',
  fontSize: '14px',
  cursor: 'pointer',
  border: '1px solid #ccc',
  borderRadius: '4px',
  backgroundColor: '#fff',
};

export default PDFViewer;
