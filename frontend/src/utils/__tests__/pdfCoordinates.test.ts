/**
 * Unit tests for PDF coordinate utilities - US-1404.1
 *
 * Tests coordinate conversion, scrolling, and validation.
 *
 * JPL Power of Ten Compliance:
 * - Rule #4: All test functions < 60 lines
 * - Rule #9: Complete type hints
 *
 * Epic: EP-14 (Foundry UI)
 * Feature: FE-11-01 (Visualization)
 * Test Date: 2026-02-18
 */

import {
  BoundingBox,
  PixelBBox,
  PageDimensions,
  convertBBoxToPixels,
  convertPixelsToBBox,
  scrollToBBox,
  isInViewport,
  expandBBox,
  getBBoxCenter,
  assertValidPage,
} from '../pdfCoordinates';

// =============================================================================
// TEST FIXTURES
// =============================================================================

const mockPageDimensions: PageDimensions = {
  width: 800,
  height: 1000,
  scale: 1.5,
};

const mockBBox: BoundingBox = {
  x1: 0.1,
  y1: 0.2,
  x2: 0.3,
  y2: 0.25,
};

// =============================================================================
// BBOX TO PIXELS TESTS (6 tests)
// =============================================================================

describe('pdfCoordinates - BBox to Pixels', () => {
  test('GIVEN normalized bbox WHEN convertBBoxToPixels called THEN returns correct pixel coordinates', () => {
    const pixelBBox = convertBBoxToPixels(mockBBox, mockPageDimensions);

    expect(pixelBBox.x).toBe(80); // 0.1 * 800
    expect(pixelBBox.y).toBe(200); // 0.2 * 1000
    expect(pixelBBox.width).toBe(160); // (0.3 - 0.1) * 800
    expect(pixelBBox.height).toBe(50); // (0.25 - 0.2) * 1000
  });

  test('GIVEN full-page bbox WHEN convertBBoxToPixels called THEN returns full page dimensions', () => {
    const fullPageBBox: BoundingBox = {
      x1: 0.0,
      y1: 0.0,
      x2: 1.0,
      y2: 1.0,
    };

    const pixelBBox = convertBBoxToPixels(fullPageBBox, mockPageDimensions);

    expect(pixelBBox.x).toBe(0);
    expect(pixelBBox.y).toBe(0);
    expect(pixelBBox.width).toBe(800);
    expect(pixelBBox.height).toBe(1000);
  });

  test('GIVEN point bbox (zero width/height) WHEN convertBBoxToPixels called THEN returns zero size', () => {
    const pointBBox: BoundingBox = {
      x1: 0.5,
      y1: 0.5,
      x2: 0.5,
      y2: 0.5,
    };

    const pixelBBox = convertBBoxToPixels(pointBBox, mockPageDimensions);

    expect(pixelBBox.width).toBe(0);
    expect(pixelBBox.height).toBe(0);
  });

  test('GIVEN different page scale WHEN convertBBoxToPixels called THEN scales correctly', () => {
    const largeDimensions: PageDimensions = {
      width: 1600,
      height: 2000,
      scale: 2.0,
    };

    const pixelBBox = convertBBoxToPixels(mockBBox, largeDimensions);

    expect(pixelBBox.x).toBe(160); // 0.1 * 1600
    expect(pixelBBox.y).toBe(400); // 0.2 * 2000
  });

  test('GIVEN invalid bbox coordinates WHEN convertBBoxToPixels called THEN throws error', () => {
    const invalidBBox: BoundingBox = {
      x1: -0.1,
      y1: 0.2,
      x2: 0.3,
      y2: 0.25,
    };

    expect(() => {
      convertBBoxToPixels(invalidBBox, mockPageDimensions);
    }).toThrow(/invalid bbox coordinates/i);
  });

  test('GIVEN bbox with x2 < x1 WHEN convertBBoxToPixels called THEN throws error', () => {
    const invalidBBox: BoundingBox = {
      x1: 0.5,
      y1: 0.2,
      x2: 0.3, // x2 < x1
      y2: 0.25,
    };

    expect(() => {
      convertBBoxToPixels(invalidBBox, mockPageDimensions);
    }).toThrow(/invalid bbox/i);
  });
});

// =============================================================================
// PIXELS TO BBOX TESTS (5 tests)
// =============================================================================

describe('pdfCoordinates - Pixels to BBox', () => {
  test('GIVEN pixel coordinates WHEN convertPixelsToBBox called THEN returns normalized bbox', () => {
    const bbox = convertPixelsToBBox(80, 200, mockPageDimensions);

    expect(bbox.x1).toBeCloseTo(0.1, 2);
    expect(bbox.y1).toBeCloseTo(0.2, 2);
    expect(bbox.x2).toBe(bbox.x1); // Point bbox
    expect(bbox.y2).toBe(bbox.y1);
  });

  test('GIVEN top-left corner WHEN convertPixelsToBBox called THEN returns 0,0', () => {
    const bbox = convertPixelsToBBox(0, 0, mockPageDimensions);

    expect(bbox.x1).toBe(0.0);
    expect(bbox.y1).toBe(0.0);
  });

  test('GIVEN bottom-right corner WHEN convertPixelsToBBox called THEN returns 1,1', () => {
    const bbox = convertPixelsToBBox(800, 1000, mockPageDimensions);

    expect(bbox.x1).toBe(1.0);
    expect(bbox.y1).toBe(1.0);
  });

  test('GIVEN coordinates outside page WHEN convertPixelsToBBox called THEN clamps to 0-1', () => {
    const bbox = convertPixelsToBBox(1600, 2000, mockPageDimensions);

    expect(bbox.x1).toBe(1.0); // Clamped to max
    expect(bbox.y1).toBe(1.0);
  });

  test('GIVEN negative coordinates WHEN convertPixelsToBBox called THEN clamps to 0', () => {
    const bbox = convertPixelsToBBox(-100, -200, mockPageDimensions);

    expect(bbox.x1).toBe(0.0); // Clamped to min
    expect(bbox.y1).toBe(0.0);
  });
});

// =============================================================================
// SCROLL TO BBOX TESTS (4 tests)
// =============================================================================

describe('pdfCoordinates - Scroll To BBox', () => {
  test('GIVEN container and bbox WHEN scrollToBBox called THEN scrolls to position', () => {
    const container = document.createElement('div');
    container.clientHeight = 600;

    const mockScrollTo = jest.fn();
    container.scrollTo = mockScrollTo;

    const pixelBBox: PixelBBox = {
      x: 80,
      y: 200,
      width: 160,
      height: 50,
    };

    scrollToBBox(container, pixelBBox, true);

    expect(mockScrollTo).toHaveBeenCalledWith({
      top: -75, // 200 - 600/2 + 50/2 = 200 - 300 + 25 = -75 (clamped to 0)
      behavior: 'smooth',
    });
  });

  test('GIVEN smooth=false WHEN scrollToBBox called THEN uses instant scroll', () => {
    const container = document.createElement('div');
    container.clientHeight = 600;

    const mockScrollTo = jest.fn();
    container.scrollTo = mockScrollTo;

    const pixelBBox: PixelBBox = {
      x: 80,
      y: 500,
      width: 160,
      height: 50,
    };

    scrollToBBox(container, pixelBBox, false);

    expect(mockScrollTo).toHaveBeenCalledWith(
      expect.objectContaining({
        behavior: 'auto',
      })
    );
  });

  test('GIVEN bbox below viewport WHEN scrollToBBox called THEN scrolls down', () => {
    const container = document.createElement('div');
    container.clientHeight = 600;

    const mockScrollTo = jest.fn();
    container.scrollTo = mockScrollTo;

    const pixelBBox: PixelBBox = {
      x: 80,
      y: 1500,
      width: 160,
      height: 50,
    };

    scrollToBBox(container, pixelBBox, true);

    const expectedTop = 1500 - 600 / 2 + 50 / 2; // 1525 - 300 = 1225
    expect(mockScrollTo).toHaveBeenCalledWith(
      expect.objectContaining({
        top: expectedTop,
      })
    );
  });

  test('GIVEN bbox at top of page WHEN scrollToBBox called THEN clamps to 0', () => {
    const container = document.createElement('div');
    container.clientHeight = 600;

    const mockScrollTo = jest.fn();
    container.scrollTo = mockScrollTo;

    const pixelBBox: PixelBBox = {
      x: 80,
      y: 50,
      width: 160,
      height: 50,
    };

    scrollToBBox(container, pixelBBox, true);

    // targetY would be negative, should clamp to 0
    expect(mockScrollTo).toHaveBeenCalledWith(
      expect.objectContaining({
        top: 0,
      })
    );
  });
});

// =============================================================================
// VIEWPORT TESTS (4 tests)
// =============================================================================

describe('pdfCoordinates - Viewport', () => {
  test('GIVEN bbox in viewport WHEN isInViewport called THEN returns true', () => {
    const container = document.createElement('div');
    Object.defineProperties(container, {
      scrollTop: { value: 200, writable: true },
      clientHeight: { value: 600 },
    });

    const pixelBBox: PixelBBox = {
      x: 80,
      y: 400,
      width: 160,
      height: 50,
    };

    expect(isInViewport(pixelBBox, container)).toBe(true);
  });

  test('GIVEN bbox below viewport WHEN isInViewport called THEN returns false', () => {
    const container = document.createElement('div');
    Object.defineProperties(container, {
      scrollTop: { value: 200, writable: true },
      clientHeight: { value: 600 },
    });

    const pixelBBox: PixelBBox = {
      x: 80,
      y: 1000, // Below viewport (200 + 600 = 800)
      width: 160,
      height: 50,
    };

    expect(isInViewport(pixelBBox, container)).toBe(false);
  });

  test('GIVEN bbox above viewport WHEN isInViewport called THEN returns false', () => {
    const container = document.createElement('div');
    Object.defineProperties(container, {
      scrollTop: { value: 500, writable: true },
      clientHeight: { value: 600 },
    });

    const pixelBBox: PixelBBox = {
      x: 80,
      y: 100, // Above viewport (scrollTop = 500)
      width: 160,
      height: 50,
    };

    expect(isInViewport(pixelBBox, container)).toBe(false);
  });

  test('GIVEN bbox partially in viewport WHEN isInViewport called THEN returns true', () => {
    const container = document.createElement('div');
    Object.defineProperties(container, {
      scrollTop: { value: 200, writable: true },
      clientHeight: { value: 600 },
    });

    const pixelBBox: PixelBBox = {
      x: 80,
      y: 750, // Partially visible (bottom edge at 800, viewport ends at 800)
      width: 160,
      height: 100, // Extends to 850
    };

    expect(isInViewport(pixelBBox, container)).toBe(true);
  });
});

// =============================================================================
// EXPAND BBOX TESTS (4 tests)
// =============================================================================

describe('pdfCoordinates - Expand BBox', () => {
  test('GIVEN bbox and padding WHEN expandBBox called THEN expands bbox', () => {
    const expanded = expandBBox(mockBBox, 0.1);

    const width = mockBBox.x2 - mockBBox.x1; // 0.2
    const height = mockBBox.y2 - mockBBox.y1; // 0.05
    const dx = width * 0.1; // 0.02
    const dy = height * 0.1; // 0.005

    expect(expanded.x1).toBeCloseTo(mockBBox.x1 - dx, 3);
    expect(expanded.y1).toBeCloseTo(mockBBox.y1 - dy, 3);
    expect(expanded.x2).toBeCloseTo(mockBBox.x2 + dx, 3);
    expect(expanded.y2).toBeCloseTo(mockBBox.y2 + dy, 3);
  });

  test('GIVEN bbox at edge WHEN expandBBox called THEN clamps to 0-1', () => {
    const edgeBBox: BoundingBox = {
      x1: 0.0,
      y1: 0.0,
      x2: 0.1,
      y2: 0.1,
    };

    const expanded = expandBBox(edgeBBox, 0.5);

    expect(expanded.x1).toBe(0.0); // Clamped to min
    expect(expanded.y1).toBe(0.0);
    expect(expanded.x2).toBeLessThanOrEqual(1.0);
    expect(expanded.y2).toBeLessThanOrEqual(1.0);
  });

  test('GIVEN zero padding WHEN expandBBox called THEN returns same bbox', () => {
    const expanded = expandBBox(mockBBox, 0.0);

    expect(expanded).toEqual(mockBBox);
  });

  test('GIVEN large padding WHEN expandBBox called THEN clamps to page bounds', () => {
    const centerBBox: BoundingBox = {
      x1: 0.4,
      y1: 0.4,
      x2: 0.6,
      y2: 0.6,
    };

    const expanded = expandBBox(centerBBox, 5.0); // Very large padding

    expect(expanded.x1).toBeGreaterThanOrEqual(0.0);
    expect(expanded.y1).toBeGreaterThanOrEqual(0.0);
    expect(expanded.x2).toBeLessThanOrEqual(1.0);
    expect(expanded.y2).toBeLessThanOrEqual(1.0);
  });
});

// =============================================================================
// BBOX CENTER TESTS (3 tests)
// =============================================================================

describe('pdfCoordinates - BBox Center', () => {
  test('GIVEN bbox WHEN getBBoxCenter called THEN returns center point', () => {
    const center = getBBoxCenter(mockBBox);

    expect(center.x).toBe(0.2); // (0.1 + 0.3) / 2
    expect(center.y).toBe(0.225); // (0.2 + 0.25) / 2
  });

  test('GIVEN full-page bbox WHEN getBBoxCenter called THEN returns 0.5,0.5', () => {
    const fullPageBBox: BoundingBox = {
      x1: 0.0,
      y1: 0.0,
      x2: 1.0,
      y2: 1.0,
    };

    const center = getBBoxCenter(fullPageBBox);

    expect(center.x).toBe(0.5);
    expect(center.y).toBe(0.5);
  });

  test('GIVEN point bbox WHEN getBBoxCenter called THEN returns same point', () => {
    const pointBBox: BoundingBox = {
      x1: 0.3,
      y1: 0.4,
      x2: 0.3,
      y2: 0.4,
    };

    const center = getBBoxCenter(pointBBox);

    expect(center.x).toBe(0.3);
    expect(center.y).toBe(0.4);
  });
});

// =============================================================================
// VALIDATION TESTS (4 tests)
// =============================================================================

describe('pdfCoordinates - Validation', () => {
  test('GIVEN valid page number WHEN assertValidPage called THEN does not throw', () => {
    expect(() => {
      assertValidPage(5, 10);
    }).not.toThrow();
  });

  test('GIVEN page < 1 WHEN assertValidPage called THEN throws error', () => {
    expect(() => {
      assertValidPage(0, 10);
    }).toThrow(/invalid page number/i);
  });

  test('GIVEN page > totalPages WHEN assertValidPage called THEN throws error', () => {
    expect(() => {
      assertValidPage(15, 10);
    }).toThrow(/invalid page number/i);
  });

  test('GIVEN page = totalPages WHEN assertValidPage called THEN does not throw', () => {
    expect(() => {
      assertValidPage(10, 10);
    }).not.toThrow();
  });
});
