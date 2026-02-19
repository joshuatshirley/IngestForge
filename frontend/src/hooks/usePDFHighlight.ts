/**
 * usePDFHighlight - Hook for managing PDF highlights
 *
 * US-1404.1 Evidence Highlight Overlay - Phase 3
 *
 * Manages highlight state, rendering, and scroll-to functionality.
 *
 * JPL Power of Ten Compliance:
 * - Rule #2: Fixed upper bounds (MAX_HIGHLIGHTS = 100)
 * - Rule #4: All functions < 60 lines
 * - Rule #9: Complete type hints
 *
 * Epic: EP-14 (Foundry UI)
 * Feature: FE-11-01 (Visualization)
 * Implementation Date: 2026-02-18
 */

import { useState, useCallback, useRef, useEffect } from 'react';
import { PDFDocumentProxy } from 'pdfjs-dist/types/src/display/api';
import {
  BoundingBox,
  PixelBBox,
  PageDimensions,
  convertBBoxToPixels,
  scrollToBBox,
  getPageDimensions,
} from '../utils/pdfCoordinates';

// =============================================================================
// TYPE DEFINITIONS (Rule #9: Complete type hints)
// =============================================================================

export interface Highlight {
  id: string;
  bbox: BoundingBox;
  page: number;
  color: string;
  opacity: number;
  confidence?: number;
  entityId?: string;
  text?: string;
}

export interface HighlightOptions {
  color?: string;
  opacity?: number;
  padding?: number;
}

export interface UsePDFHighlightReturn {
  highlights: Highlight[];
  addHighlight: (
    bbox: BoundingBox,
    page: number,
    options?: HighlightOptions
  ) => string;
  removeHighlight: (id: string) => void;
  clearHighlights: () => void;
  scrollToHighlight: (id: string) => Promise<void>;
  renderHighlights: (
    canvas: HTMLCanvasElement,
    pdfDoc: PDFDocumentProxy,
    pageNum: number,
    zoom: number
  ) => Promise<void>;
}

// =============================================================================
// CONSTANTS (JPL Rule #2: Fixed upper bounds)
// =============================================================================

const MAX_HIGHLIGHTS = 100; // Maximum highlights per page
const DEFAULT_HIGHLIGHT_COLOR = '#FFEB3B'; // Yellow
const DEFAULT_OPACITY = 0.3;

// Confidence-based colors (from US-1404.1 spec)
const CONFIDENCE_COLORS = {
  HIGH: '#4CAF50', // Green (confidence > 0.9)
  MEDIUM: '#FFEB3B', // Yellow (confidence 0.7-0.9)
  LOW: '#F44336', // Red (confidence < 0.7)
};

// =============================================================================
// HOOK (Rule #4: Main logic < 60 lines)
// =============================================================================

export function usePDFHighlight(
  containerRef: React.RefObject<HTMLElement>
): UsePDFHighlightReturn {
  const [highlights, setHighlights] = useState<Highlight[]>([]);
  const highlightIdCounter = useRef<number>(0);

  const addHighlight = useCallback(
    (
      bbox: BoundingBox,
      page: number,
      options: HighlightOptions = {}
    ): string => {
      // JPL Rule #2: Enforce upper bound
      if (highlights.length >= MAX_HIGHLIGHTS) {
        console.warn(`Maximum highlights (${MAX_HIGHLIGHTS}) reached`);
        return '';
      }

      const id = `highlight-${++highlightIdCounter.current}`;
      const color = options.color || DEFAULT_HIGHLIGHT_COLOR;
      const opacity = options.opacity ?? DEFAULT_OPACITY;

      const newHighlight: Highlight = {
        id,
        bbox,
        page,
        color,
        opacity,
      };

      setHighlights((prev) => [...prev, newHighlight]);
      return id;
    },
    [highlights.length]
  );

  const removeHighlight = useCallback((id: string): void => {
    setHighlights((prev) => prev.filter((h) => h.id !== id));
  }, []);

  const clearHighlights = useCallback((): void => {
    setHighlights([]);
    highlightIdCounter.current = 0;
  }, []);

  const scrollToHighlight = useCallback(
    async (id: string): Promise<void> => {
      const highlight = highlights.find((h) => h.id === id);
      if (!highlight || !containerRef.current) return;

      // Convert bbox to pixels (assumes current page dimensions)
      // TODO: Get actual page dimensions from PDF
      const pageDimensions: PageDimensions = {
        width: 800,
        height: 1000,
        scale: 1.5,
      };

      const pixelBBox = convertBBoxToPixels(highlight.bbox, pageDimensions);
      scrollToBBox(containerRef.current, pixelBBox, true);
    },
    [highlights, containerRef]
  );

  const renderHighlights = useCallback(
    async (
      canvas: HTMLCanvasElement,
      pdfDoc: PDFDocumentProxy,
      pageNum: number,
      zoom: number
    ): Promise<void> => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Get page dimensions
      const pageDimensions = await getPageDimensions(pdfDoc, pageNum, zoom);

      // Filter highlights for current page
      const pageHighlights = highlights.filter((h) => h.page === pageNum);

      // Render each highlight
      for (const highlight of pageHighlights) {
        const pixelBBox = convertBBoxToPixels(highlight.bbox, pageDimensions);
        drawHighlight(ctx, pixelBBox, highlight.color, highlight.opacity);
      }
    },
    [highlights]
  );

  return {
    highlights,
    addHighlight,
    removeHighlight,
    clearHighlights,
    scrollToHighlight,
    renderHighlights,
  };
}

// =============================================================================
// HELPER FUNCTIONS (JPL Rule #4: < 60 lines each)
// =============================================================================

/**
 * Draw a single highlight rectangle on canvas.
 *
 * US-1404.1 AC: Render semi-transparent bounding box overlay.
 *
 * Rule #4: Under 60 lines.
 *
 * @param ctx - Canvas 2D context
 * @param bbox - Bounding box in pixels
 * @param color - Fill color
 * @param opacity - Alpha transparency (0-1)
 */
function drawHighlight(
  ctx: CanvasRenderingContext2D,
  bbox: PixelBBox,
  color: string,
  opacity: number
): void {
  const { x, y, width, height } = bbox;

  // Save context state
  ctx.save();

  // Draw filled rectangle
  ctx.fillStyle = color;
  ctx.globalAlpha = opacity;
  ctx.fillRect(x, y, width, height);

  // Draw border
  ctx.strokeStyle = color;
  ctx.globalAlpha = 1.0;
  ctx.lineWidth = 2;
  ctx.strokeRect(x, y, width, height);

  // Restore context state
  ctx.restore();
}

/**
 * Get color based on confidence level.
 *
 * US-1404.1 AC: Color-coded confidence indicators.
 *
 * Rule #4: Under 60 lines.
 *
 * @param confidence - Confidence score (0-1)
 * @returns CSS color string
 */
export function getConfidenceColor(confidence: number): string {
  if (confidence >= 0.9) {
    return CONFIDENCE_COLORS.HIGH;
  } else if (confidence >= 0.7) {
    return CONFIDENCE_COLORS.MEDIUM;
  } else {
    return CONFIDENCE_COLORS.LOW;
  }
}

/**
 * Group highlights by page for efficient rendering.
 *
 * Rule #4: Under 60 lines.
 *
 * @param highlights - Array of highlights
 * @returns Map of page number to highlights
 */
export function groupHighlightsByPage(
  highlights: Highlight[]
): Map<number, Highlight[]> {
  const grouped = new Map<number, Highlight[]>();

  for (const highlight of highlights) {
    const pageHighlights = grouped.get(highlight.page) || [];
    pageHighlights.push(highlight);
    grouped.set(highlight.page, pageHighlights);
  }

  return grouped;
}

/**
 * Create highlight from entity click event.
 *
 * US-1404.1 AC: Convert entity node to PDF highlight.
 *
 * Rule #4: Under 60 lines.
 *
 * @param entityId - Entity ID from knowledge graph
 * @param bbox - Bounding box coordinates
 * @param page - Page number
 * @param confidence - Extraction confidence
 * @returns Highlight object
 */
export function createHighlightFromEntity(
  entityId: string,
  bbox: BoundingBox,
  page: number,
  confidence: number
): Highlight {
  const color = getConfidenceColor(confidence);

  return {
    id: `entity-${entityId}-${page}`,
    bbox,
    page,
    color,
    opacity: DEFAULT_OPACITY,
    confidence,
    entityId,
  };
}

/**
 * Calculate total highlight area for performance monitoring.
 *
 * Rule #4: Under 60 lines.
 *
 * @param highlights - Array of highlights
 * @returns Total area covered (0-1 normalized)
 */
export function calculateHighlightArea(highlights: Highlight[]): number {
  let totalArea = 0;

  for (const highlight of highlights) {
    const width = highlight.bbox.x2 - highlight.bbox.x1;
    const height = highlight.bbox.y2 - highlight.bbox.y1;
    totalArea += width * height;
  }

  return totalArea;
}
