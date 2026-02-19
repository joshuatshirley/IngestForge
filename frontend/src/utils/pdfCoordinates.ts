/**
 * PDF Coordinate Conversion Utilities - US-1404.1
 *
 * Converts between normalized (0-1) bbox coordinates and pixel coordinates.
 *
 * JPL Power of Ten Compliance:
 * - Rule #4: All functions < 60 lines
 * - Rule #5: Assertions for coordinate bounds
 * - Rule #9: Complete type hints
 *
 * Epic: EP-14 (Foundry UI)
 * Feature: FE-11-01 (Visualization)
 * Implementation Date: 2026-02-18
 */

import { PDFDocumentProxy } from 'pdfjs-dist/types/src/display/api';

// =============================================================================
// TYPE DEFINITIONS (Rule #9: Complete type hints)
// =============================================================================

/** Normalized bounding box (0-1 coordinates) */
export interface BoundingBox {
  x1: number; // Left edge (0-1)
  y1: number; // Top edge (0-1)
  x2: number; // Right edge (0-1)
  y2: number; // Bottom edge (0-1)
}

/** Pixel-based bounding box (screen coordinates) */
export interface PixelBBox {
  x: number; // Left edge (pixels)
  y: number; // Top edge (pixels)
  width: number; // Width (pixels)
  height: number; // Height (pixels)
}

/** Page dimensions in pixels */
export interface PageDimensions {
  width: number;
  height: number;
  scale: number;
}

// =============================================================================
// CONSTANTS (JPL Rule #2: Fixed upper bounds)
// =============================================================================

const MIN_COORDINATE = 0.0;
const MAX_COORDINATE = 1.0;
const MAX_PAGE_NUMBER = 10000;

// =============================================================================
// COORDINATE CONVERSION FUNCTIONS (JPL Rule #4: < 60 lines each)
// =============================================================================

/**
 * Convert normalized bbox to pixel coordinates.
 *
 * US-1404.1 AC: Transform 0-1 coordinates to screen pixels for rendering.
 *
 * Rule #4: Under 60 lines.
 * Rule #5: Assert coordinate bounds.
 *
 * @param bbox - Normalized bounding box (0-1)
 * @param pageDimensions - Page dimensions in pixels
 * @returns Pixel-based bounding box
 */
export function convertBBoxToPixels(
  bbox: BoundingBox,
  pageDimensions: PageDimensions
): PixelBBox {
  // JPL Rule #5: Assert preconditions
  assertValidBBox(bbox);

  const { width: pageWidth, height: pageHeight } = pageDimensions;

  // Convert normalized coordinates to pixels
  const x = bbox.x1 * pageWidth;
  const y = bbox.y1 * pageHeight;
  const width = (bbox.x2 - bbox.x1) * pageWidth;
  const height = (bbox.y2 - bbox.y1) * pageHeight;

  return { x, y, width, height };
}

/**
 * Convert pixel coordinates to normalized bbox.
 *
 * US-1404.1 AC: Transform screen clicks to 0-1 coordinates for storage.
 *
 * Rule #4: Under 60 lines.
 * Rule #5: Assert coordinate bounds.
 *
 * @param x - X pixel coordinate
 * @param y - Y pixel coordinate
 * @param pageDimensions - Page dimensions in pixels
 * @returns Normalized bounding box (point bbox)
 */
export function convertPixelsToBBox(
  x: number,
  y: number,
  pageDimensions: PageDimensions
): BoundingBox {
  const { width: pageWidth, height: pageHeight } = pageDimensions;

  // Normalize to 0-1 range
  const x1 = Math.max(MIN_COORDINATE, Math.min(MAX_COORDINATE, x / pageWidth));
  const y1 = Math.max(MIN_COORDINATE, Math.min(MAX_COORDINATE, y / pageHeight));

  // Return point bbox (zero width/height)
  return { x1, y1, x2: x1, y2: y1 };
}

/**
 * Get page dimensions at current zoom level.
 *
 * US-1404.1 AC: Retrieve page size for coordinate conversion.
 *
 * Rule #4: Under 60 lines.
 *
 * @param pdfDoc - PDF document proxy
 * @param pageNum - Page number (1-indexed)
 * @param scale - Zoom scale (1.0 = 100%)
 * @returns Page dimensions in pixels
 */
export async function getPageDimensions(
  pdfDoc: PDFDocumentProxy,
  pageNum: number,
  scale: number
): Promise<PageDimensions> {
  // JPL Rule #5: Assert preconditions
  if (pageNum < 1 || pageNum > pdfDoc.numPages) {
    throw new Error(`Invalid page number: ${pageNum} (max: ${pdfDoc.numPages})`);
  }

  const page = await pdfDoc.getPage(pageNum);
  const viewport = page.getViewport({ scale });

  return {
    width: viewport.width,
    height: viewport.height,
    scale,
  };
}

/**
 * Scroll container to show bounding box.
 *
 * US-1404.1 AC: Smooth scroll to entity location (<50ms latency).
 *
 * Rule #4: Under 60 lines.
 *
 * @param container - Scrollable container element
 * @param pixelBBox - Bounding box in pixel coordinates
 * @param smooth - Use smooth scrolling animation
 */
export function scrollToBBox(
  container: HTMLElement,
  pixelBBox: PixelBBox,
  smooth: boolean = true
): void {
  const targetY = pixelBBox.y - container.clientHeight / 2 + pixelBBox.height / 2;

  container.scrollTo({
    top: Math.max(0, targetY),
    behavior: smooth ? 'smooth' : 'auto',
  });
}

/**
 * Check if bounding box is within viewport.
 *
 * Rule #4: Under 60 lines.
 *
 * @param pixelBBox - Bounding box in pixel coordinates
 * @param container - Scrollable container element
 * @returns True if bbox is visible in viewport
 */
export function isInViewport(
  pixelBBox: PixelBBox,
  container: HTMLElement
): boolean {
  const scrollTop = container.scrollTop;
  const scrollBottom = scrollTop + container.clientHeight;

  const bboxTop = pixelBBox.y;
  const bboxBottom = pixelBBox.y + pixelBBox.height;

  return bboxBottom >= scrollTop && bboxTop <= scrollBottom;
}

/**
 * Expand bounding box by padding percentage.
 *
 * Rule #4: Under 60 lines.
 *
 * @param bbox - Normalized bounding box
 * @param padding - Padding as fraction of bbox size (0.1 = 10%)
 * @returns Expanded bounding box
 */
export function expandBBox(bbox: BoundingBox, padding: number): BoundingBox {
  const width = bbox.x2 - bbox.x1;
  const height = bbox.y2 - bbox.y1;

  const dx = width * padding;
  const dy = height * padding;

  return {
    x1: Math.max(MIN_COORDINATE, bbox.x1 - dx),
    y1: Math.max(MIN_COORDINATE, bbox.y1 - dy),
    x2: Math.min(MAX_COORDINATE, bbox.x2 + dx),
    y2: Math.min(MAX_COORDINATE, bbox.y2 + dy),
  };
}

/**
 * Calculate bbox center point.
 *
 * Rule #4: Under 60 lines.
 *
 * @param bbox - Normalized bounding box
 * @returns Center point coordinates
 */
export function getBBoxCenter(bbox: BoundingBox): { x: number; y: number } {
  return {
    x: (bbox.x1 + bbox.x2) / 2,
    y: (bbox.y1 + bbox.y2) / 2,
  };
}

// =============================================================================
// VALIDATION HELPERS (JPL Rule #5: Assert preconditions)
// =============================================================================

/**
 * Assert bounding box coordinates are valid.
 *
 * Rule #4: Under 60 lines.
 * Rule #5: Fail-fast validation.
 *
 * @param bbox - Bounding box to validate
 * @throws Error if coordinates are invalid
 */
function assertValidBBox(bbox: BoundingBox): void {
  if (
    bbox.x1 < MIN_COORDINATE ||
    bbox.x1 > MAX_COORDINATE ||
    bbox.x2 < MIN_COORDINATE ||
    bbox.x2 > MAX_COORDINATE ||
    bbox.y1 < MIN_COORDINATE ||
    bbox.y1 > MAX_COORDINATE ||
    bbox.y2 < MIN_COORDINATE ||
    bbox.y2 > MAX_COORDINATE
  ) {
    throw new Error(
      `Invalid bbox coordinates: x1=${bbox.x1}, y1=${bbox.y1}, x2=${bbox.x2}, y2=${bbox.y2} (must be 0-1)`
    );
  }

  if (bbox.x2 < bbox.x1) {
    throw new Error(`Invalid bbox: x2 (${bbox.x2}) < x1 (${bbox.x1})`);
  }

  if (bbox.y2 < bbox.y1) {
    throw new Error(`Invalid bbox: y2 (${bbox.y2}) < y1 (${bbox.y1})`);
  }
}

/**
 * Assert page number is valid.
 *
 * Rule #4: Under 60 lines.
 * Rule #5: Fail-fast validation.
 *
 * @param pageNum - Page number to validate (1-indexed)
 * @param totalPages - Total pages in document
 * @throws Error if page number is invalid
 */
export function assertValidPage(pageNum: number, totalPages: number): void {
  if (pageNum < 1 || pageNum > Math.min(totalPages, MAX_PAGE_NUMBER)) {
    throw new Error(
      `Invalid page number: ${pageNum} (must be 1-${totalPages})`
    );
  }
}
