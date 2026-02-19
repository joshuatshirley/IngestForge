/**
 * HighlightCanvas - PDF viewer with highlight overlay
 *
 * US-1404.1 Evidence Highlight Overlay - Phase 3
 *
 * Combines PDF.js rendering with canvas-based highlight overlay.
 *
 * JPL Power of Ten Compliance:
 * - Rule #2: Fixed upper bounds (MAX_OVERLAYS = 3)
 * - Rule #4: All functions < 60 lines
 * - Rule #9: Complete type hints
 *
 * Epic: EP-14 (Foundry UI)
 * Feature: FE-11-01 (Visualization)
 * Implementation Date: 2026-02-18
 */

import React, { useRef, useEffect, useState, useCallback } from 'react';
import * as pdfjsLib from 'pdfjs-dist';
import { PDFDocumentProxy } from 'pdfjs-dist/types/src/display/api';
import { usePDFHighlight, Highlight } from '../../hooks/usePDFHighlight';
import {
  BoundingBox,
  getPageDimensions,
  convertBBoxToPixels,
} from '../../utils/pdfCoordinates';

// =============================================================================
// TYPE DEFINITIONS (Rule #9: Complete type hints)
// =============================================================================

export interface HighlightCanvasProps {
  /** URL to PDF document */
  documentUrl: string;
  /** Current page number (1-indexed) */
  currentPage?: number;
  /** Highlights to display */
  highlights?: Highlight[];
  /** Zoom level (1.0 = 100%) */
  zoom?: number;
  /** Width of canvas */
  width?: number;
  /** Height of canvas */
  height?: number;
  /** Callback when highlight is clicked */
  onHighlightClick?: (highlight: Highlight) => void;
}

// =============================================================================
// CONSTANTS (JPL Rule #2: Fixed upper bounds)
// =============================================================================

const MAX_OVERLAYS = 3; // Base PDF + Highlight + Selection overlays
const DEFAULT_ZOOM = 1.5;
const DEFAULT_WIDTH = 800;
const DEFAULT_HEIGHT = 1000;

// =============================================================================
// COMPONENT (Rule #4: Main render < 60 lines)
// =============================================================================

export const HighlightCanvas: React.FC<HighlightCanvasProps> = ({
  documentUrl,
  currentPage = 1,
  highlights = [],
  zoom = DEFAULT_ZOOM,
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
  onHighlightClick,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const pdfCanvasRef = useRef<HTMLCanvasElement>(null);
  const highlightCanvasRef = useRef<HTMLCanvasElement>(null);

  const [pdfDoc, setPdfDoc] = useState<PDFDocumentProxy | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  const { renderHighlights } = usePDFHighlight(containerRef);

  // Load PDF document
  useEffect(() => {
    loadPDF(documentUrl, setPdfDoc, setLoading);
  }, [documentUrl]);

  // Render PDF page
  useEffect(() => {
    if (!pdfDoc || !pdfCanvasRef.current) return;
    renderPDFPage(pdfDoc, currentPage, zoom, pdfCanvasRef.current);
  }, [pdfDoc, currentPage, zoom]);

  // Render highlights overlay
  useEffect(() => {
    if (!pdfDoc || !highlightCanvasRef.current) return;
    renderHighlightOverlay(
      pdfDoc,
      currentPage,
      zoom,
      highlights,
      highlightCanvasRef.current
    );
  }, [pdfDoc, currentPage, zoom, highlights]);

  // Handle canvas click for highlight selection
  const handleCanvasClick = useCallback(
    (event: React.MouseEvent<HTMLCanvasElement>): void => {
      if (!onHighlightClick || !highlightCanvasRef.current || !pdfDoc) return;

      const rect = highlightCanvasRef.current.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      // Find clicked highlight
      findHighlightAtPoint(highlights, x, y, pdfDoc, currentPage, zoom).then(
        (clickedHighlight) => {
          if (clickedHighlight) {
            onHighlightClick(clickedHighlight);
          }
        }
      );
    },
    [onHighlightClick, highlights, pdfDoc, currentPage, zoom]
  );

  return (
    <div
      ref={containerRef}
      className="highlight-canvas-container"
      style={{ position: 'relative', width, height }}
      data-testid="highlight-canvas"
    >
      {loading && <div style={loadingStyle}>Loading PDF...</div>}

      <canvas
        ref={pdfCanvasRef}
        className="pdf-canvas-layer"
        style={canvasLayerStyle(0)}
        data-testid="pdf-canvas-layer"
      />

      <canvas
        ref={highlightCanvasRef}
        className="highlight-canvas-layer"
        style={canvasLayerStyle(1)}
        onClick={handleCanvasClick}
        data-testid="highlight-canvas-layer"
      />
    </div>
  );
};

// =============================================================================
// HELPER FUNCTIONS (JPL Rule #4: < 60 lines each)
// =============================================================================

/**
 * Load PDF document.
 *
 * Rule #4: Under 60 lines.
 */
async function loadPDF(
  documentUrl: string,
  setPdfDoc: React.Dispatch<React.SetStateAction<PDFDocumentProxy | null>>,
  setLoading: React.Dispatch<React.SetStateAction<boolean>>
): Promise<void> {
  try {
    setLoading(true);
    const loadingTask = pdfjsLib.getDocument(documentUrl);
    const pdf = await loadingTask.promise;
    setPdfDoc(pdf);
    setLoading(false);
  } catch (err) {
    console.error('Failed to load PDF:', err);
    setLoading(false);
  }
}

/**
 * Render PDF page to canvas.
 *
 * Rule #4: Under 60 lines.
 */
async function renderPDFPage(
  pdfDoc: PDFDocumentProxy,
  pageNum: number,
  zoom: number,
  canvas: HTMLCanvasElement
): Promise<void> {
  try {
    const page = await pdfDoc.getPage(pageNum);
    const viewport = page.getViewport({ scale: zoom });

    canvas.width = viewport.width;
    canvas.height = viewport.height;

    const context = canvas.getContext('2d');
    if (!context) return;

    await page.render({ canvasContext: context, viewport }).promise;
  } catch (err) {
    console.error('Failed to render PDF page:', err);
  }
}

/**
 * Render highlight overlay on separate canvas.
 *
 * US-1404.1 AC: Draw highlights on transparent overlay.
 *
 * Rule #4: Under 60 lines.
 */
async function renderHighlightOverlay(
  pdfDoc: PDFDocumentProxy,
  pageNum: number,
  zoom: number,
  highlights: Highlight[],
  canvas: HTMLCanvasElement
): Promise<void> {
  try {
    // Get page dimensions
    const pageDimensions = await getPageDimensions(pdfDoc, pageNum, zoom);

    // Set canvas size to match PDF canvas
    canvas.width = pageDimensions.width;
    canvas.height = pageDimensions.height;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear previous highlights
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Filter highlights for current page
    const pageHighlights = highlights.filter((h) => h.page === pageNum);

    // Render each highlight
    for (const highlight of pageHighlights) {
      const pixelBBox = convertBBoxToPixels(highlight.bbox, pageDimensions);
      drawHighlightRect(ctx, pixelBBox, highlight.color, highlight.opacity);
    }
  } catch (err) {
    console.error('Failed to render highlights:', err);
  }
}

/**
 * Draw highlight rectangle with border.
 *
 * Rule #4: Under 60 lines.
 */
function drawHighlightRect(
  ctx: CanvasRenderingContext2D,
  bbox: { x: number; y: number; width: number; height: number },
  color: string,
  opacity: number
): void {
  ctx.save();

  // Fill
  ctx.fillStyle = color;
  ctx.globalAlpha = opacity;
  ctx.fillRect(bbox.x, bbox.y, bbox.width, bbox.height);

  // Border
  ctx.strokeStyle = color;
  ctx.globalAlpha = 1.0;
  ctx.lineWidth = 2;
  ctx.strokeRect(bbox.x, bbox.y, bbox.width, bbox.height);

  ctx.restore();
}

/**
 * Find highlight at click coordinates.
 *
 * US-1404.1 AC: Click detection for PDF-to-graph sync.
 *
 * Rule #4: Under 60 lines.
 */
async function findHighlightAtPoint(
  highlights: Highlight[],
  x: number,
  y: number,
  pdfDoc: PDFDocumentProxy,
  pageNum: number,
  zoom: number
): Promise<Highlight | null> {
  const pageDimensions = await getPageDimensions(pdfDoc, pageNum, zoom);
  const pageHighlights = highlights.filter((h) => h.page === pageNum);

  for (const highlight of pageHighlights) {
    const pixelBBox = convertBBoxToPixels(highlight.bbox, pageDimensions);

    if (
      x >= pixelBBox.x &&
      x <= pixelBBox.x + pixelBBox.width &&
      y >= pixelBBox.y &&
      y <= pixelBBox.y + pixelBBox.height
    ) {
      return highlight;
    }
  }

  return null;
}

// =============================================================================
// STYLES
// =============================================================================

const loadingStyle: React.CSSProperties = {
  position: 'absolute',
  top: '50%',
  left: '50%',
  transform: 'translate(-50%, -50%)',
  fontSize: '18px',
  color: '#666',
};

function canvasLayerStyle(zIndex: number): React.CSSProperties {
  return {
    position: 'absolute',
    top: 0,
    left: 0,
    zIndex,
    pointerEvents: zIndex > 0 ? 'auto' : 'none', // Only top layer receives clicks
  };
}

export default HighlightCanvas;
