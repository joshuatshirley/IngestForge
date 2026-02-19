/**
 * Unit tests for PDFViewer component - US-1404.1
 *
 * Tests PDF.js integration, page rendering, and navigation.
 *
 * JPL Power of Ten Compliance:
 * - Rule #4: All test functions < 60 lines
 * - Rule #9: Complete type hints
 *
 * Epic: EP-14 (Foundry UI)
 * Feature: FE-11-01 (Visualization)
 * Test Date: 2026-02-18
 */

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { PDFViewer } from '../PDFViewer';
import * as pdfjsLib from 'pdfjs-dist';

// =============================================================================
// MOCKS
// =============================================================================

const mockPDFPage = {
  getViewport: jest.fn(() => ({
    width: 800,
    height: 1000,
    scale: 1.5,
  })),
  render: jest.fn(() => ({
    promise: Promise.resolve(),
  })),
};

const mockPDFDoc = {
  numPages: 10,
  getPage: jest.fn(() => Promise.resolve(mockPDFPage)),
};

jest.mock('pdfjs-dist', () => ({
  GlobalWorkerOptions: {
    workerSrc: '',
  },
  version: '2.14.305',
  getDocument: jest.fn(() => ({
    promise: Promise.resolve(mockPDFDoc),
  })),
}));

// =============================================================================
// BASIC RENDERING TESTS (5 tests)
// =============================================================================

describe('PDFViewer - Rendering', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('GIVEN valid PDF URL WHEN component mounts THEN loads PDF document', async () => {
    render(<PDFViewer documentUrl="/test.pdf" />);

    await waitFor(() => {
      expect(pdfjsLib.getDocument).toHaveBeenCalledWith('/test.pdf');
    });
  });

  test('GIVEN PDF loading WHEN component renders THEN shows loading message', () => {
    (pdfjsLib.getDocument as jest.Mock).mockReturnValue({
      promise: new Promise(() => {}), // Never resolves
    });

    render(<PDFViewer documentUrl="/test.pdf" />);

    expect(screen.getByText(/loading pdf/i)).toBeInTheDocument();
  });

  test('GIVEN PDF loaded WHEN rendered THEN displays canvas element', async () => {
    render(<PDFViewer documentUrl="/test.pdf" currentPage={1} />);

    await waitFor(() => {
      const canvas = screen.getByTestId('pdf-canvas');
      expect(canvas).toBeInTheDocument();
    });
  });

  test('GIVEN PDF loaded WHEN rendered THEN displays page navigation controls', async () => {
    render(<PDFViewer documentUrl="/test.pdf" currentPage={1} />);

    await waitFor(() => {
      expect(screen.getByTestId('pdf-prev-button')).toBeInTheDocument();
      expect(screen.getByTestId('pdf-next-button')).toBeInTheDocument();
      expect(screen.getByTestId('pdf-page-indicator')).toBeInTheDocument();
    });
  });

  test('GIVEN PDF with 10 pages WHEN loaded THEN displays correct page count', async () => {
    render(<PDFViewer documentUrl="/test.pdf" currentPage={1} />);

    await waitFor(() => {
      expect(screen.getByText(/page 1 of 10/i)).toBeInTheDocument();
    });
  });
});

// =============================================================================
// PAGE NAVIGATION TESTS (6 tests)
// =============================================================================

describe('PDFViewer - Page Navigation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('GIVEN on page 1 WHEN Previous button clicked THEN stays on page 1', async () => {
    const onPageChange = jest.fn();
    render(
      <PDFViewer
        documentUrl="/test.pdf"
        currentPage={1}
        onPageChange={onPageChange}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId('pdf-prev-button')).toBeInTheDocument();
    });

    const prevButton = screen.getByTestId('pdf-prev-button');
    expect(prevButton).toBeDisabled();

    fireEvent.click(prevButton);
    expect(onPageChange).not.toHaveBeenCalled();
  });

  test('GIVEN on page 1 WHEN Next button clicked THEN navigates to page 2', async () => {
    const onPageChange = jest.fn();
    render(
      <PDFViewer
        documentUrl="/test.pdf"
        currentPage={1}
        onPageChange={onPageChange}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId('pdf-next-button')).toBeInTheDocument();
    });

    const nextButton = screen.getByTestId('pdf-next-button');
    fireEvent.click(nextButton);

    expect(onPageChange).toHaveBeenCalledWith(2);
  });

  test('GIVEN on page 5 WHEN Previous button clicked THEN navigates to page 4', async () => {
    const onPageChange = jest.fn();
    render(
      <PDFViewer
        documentUrl="/test.pdf"
        currentPage={5}
        onPageChange={onPageChange}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId('pdf-prev-button')).toBeInTheDocument();
    });

    const prevButton = screen.getByTestId('pdf-prev-button');
    fireEvent.click(prevButton);

    expect(onPageChange).toHaveBeenCalledWith(4);
  });

  test('GIVEN on last page (10) WHEN Next button clicked THEN stays on page 10', async () => {
    const onPageChange = jest.fn();
    render(
      <PDFViewer
        documentUrl="/test.pdf"
        currentPage={10}
        onPageChange={onPageChange}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId('pdf-next-button')).toBeInTheDocument();
    });

    const nextButton = screen.getByTestId('pdf-next-button');
    expect(nextButton).toBeDisabled();

    fireEvent.click(nextButton);
    expect(onPageChange).not.toHaveBeenCalled();
  });

  test('GIVEN page changes WHEN currentPage prop updates THEN renders new page', async () => {
    const { rerender } = render(
      <PDFViewer documentUrl="/test.pdf" currentPage={1} />
    );

    await waitFor(() => {
      expect(screen.getByText(/page 1 of 10/i)).toBeInTheDocument();
    });

    // Change page
    rerender(<PDFViewer documentUrl="/test.pdf" currentPage={5} />);

    await waitFor(() => {
      expect(mockPDFDoc.getPage).toHaveBeenCalledWith(5);
    });
  });

  test('GIVEN current page indicator WHEN rendered THEN shows correct format', async () => {
    render(<PDFViewer documentUrl="/test.pdf" currentPage={3} />);

    await waitFor(() => {
      const indicator = screen.getByTestId('pdf-page-indicator');
      expect(indicator).toHaveTextContent('Page 3 of 10');
    });
  });
});

// =============================================================================
// ZOOM TESTS (4 tests)
// =============================================================================

describe('PDFViewer - Zoom', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('GIVEN default zoom WHEN component renders THEN uses 1.5x scale', async () => {
    render(<PDFViewer documentUrl="/test.pdf" currentPage={1} />);

    await waitFor(() => {
      expect(mockPDFPage.getViewport).toHaveBeenCalledWith({ scale: 1.5 });
    });
  });

  test('GIVEN custom zoom 2.0 WHEN component renders THEN uses 2.0x scale', async () => {
    render(<PDFViewer documentUrl="/test.pdf" currentPage={1} zoom={2.0} />);

    await waitFor(() => {
      expect(mockPDFPage.getViewport).toHaveBeenCalledWith({ scale: 2.0 });
    });
  });

  test('GIVEN zoom changes WHEN zoom prop updates THEN re-renders page at new zoom', async () => {
    const { rerender } = render(
      <PDFViewer documentUrl="/test.pdf" currentPage={1} zoom={1.0} />
    );

    await waitFor(() => {
      expect(mockPDFPage.getViewport).toHaveBeenCalledWith({ scale: 1.0 });
    });

    // Change zoom
    rerender(<PDFViewer documentUrl="/test.pdf" currentPage={1} zoom={2.0} />);

    await waitFor(() => {
      expect(mockPDFPage.getViewport).toHaveBeenCalledWith({ scale: 2.0 });
    });
  });

  test('GIVEN canvas WHEN page rendered THEN canvas dimensions match viewport', async () => {
    render(<PDFViewer documentUrl="/test.pdf" currentPage={1} zoom={1.5} />);

    await waitFor(() => {
      const canvas = screen.getByTestId('pdf-canvas') as HTMLCanvasElement;
      expect(canvas.width).toBe(800);
      expect(canvas.height).toBe(1000);
    });
  });
});

// =============================================================================
// ERROR HANDLING TESTS (3 tests)
// =============================================================================

describe('PDFViewer - Error Handling', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('GIVEN invalid PDF URL WHEN load fails THEN displays error message', async () => {
    (pdfjsLib.getDocument as jest.Mock).mockReturnValue({
      promise: Promise.reject(new Error('Failed to load PDF')),
    });

    render(<PDFViewer documentUrl="/invalid.pdf" />);

    await waitFor(() => {
      expect(screen.getByTestId('pdf-error')).toBeInTheDocument();
      expect(screen.getByText(/failed to load pdf/i)).toBeInTheDocument();
    });
  });

  test('GIVEN page render fails WHEN rendering THEN displays error message', async () => {
    mockPDFPage.render.mockReturnValue({
      promise: Promise.reject(new Error('Render failed')),
    });

    render(<PDFViewer documentUrl="/test.pdf" currentPage={1} />);

    await waitFor(() => {
      expect(screen.getByTestId('pdf-error')).toBeInTheDocument();
    });
  });

  test('GIVEN error state WHEN rendered THEN hides canvas and controls', async () => {
    (pdfjsLib.getDocument as jest.Mock).mockReturnValue({
      promise: Promise.reject(new Error('Load failed')),
    });

    render(<PDFViewer documentUrl="/test.pdf" />);

    await waitFor(() => {
      expect(screen.getByTestId('pdf-error')).toBeInTheDocument();
    });

    expect(screen.queryByTestId('pdf-canvas')).not.toBeInTheDocument();
    expect(screen.queryByTestId('pdf-prev-button')).not.toBeInTheDocument();
  });
});

// =============================================================================
// CALLBACK TESTS (2 tests)
// =============================================================================

describe('PDFViewer - Callbacks', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('GIVEN onDocumentLoad callback WHEN PDF loads THEN calls callback with page count', async () => {
    const onDocumentLoad = jest.fn();

    render(
      <PDFViewer
        documentUrl="/test.pdf"
        onDocumentLoad={onDocumentLoad}
      />
    );

    await waitFor(() => {
      expect(onDocumentLoad).toHaveBeenCalledWith(10);
    });
  });

  test('GIVEN onPageChange callback WHEN page changes THEN calls callback with new page', async () => {
    const onPageChange = jest.fn();

    render(
      <PDFViewer
        documentUrl="/test.pdf"
        currentPage={1}
        onPageChange={onPageChange}
      />
    );

    await waitFor(() => {
      expect(screen.getByTestId('pdf-next-button')).toBeInTheDocument();
    });

    const nextButton = screen.getByTestId('pdf-next-button');
    fireEvent.click(nextButton);

    expect(onPageChange).toHaveBeenCalledWith(2);
  });
});

// =============================================================================
// DOCUMENT CHANGE TESTS (2 tests)
// =============================================================================

describe('PDFViewer - Document Changes', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('GIVEN document URL changes WHEN re-rendered THEN loads new document', async () => {
    const { rerender } = render(
      <PDFViewer documentUrl="/test1.pdf" currentPage={1} />
    );

    await waitFor(() => {
      expect(pdfjsLib.getDocument).toHaveBeenCalledWith('/test1.pdf');
    });

    // Change document
    rerender(<PDFViewer documentUrl="/test2.pdf" currentPage={1} />);

    await waitFor(() => {
      expect(pdfjsLib.getDocument).toHaveBeenCalledWith('/test2.pdf');
    });
  });

  test('GIVEN new document loading WHEN URL changes THEN shows loading state', async () => {
    const { rerender } = render(
      <PDFViewer documentUrl="/test1.pdf" currentPage={1} />
    );

    await waitFor(() => {
      expect(screen.getByTestId('pdf-canvas')).toBeInTheDocument();
    });

    // Change to slow-loading document
    (pdfjsLib.getDocument as jest.Mock).mockReturnValue({
      promise: new Promise(() => {}), // Never resolves
    });

    rerender(<PDFViewer documentUrl="/test2.pdf" currentPage={1} />);

    await waitFor(() => {
      expect(screen.getByText(/loading pdf/i)).toBeInTheDocument();
    });
  });
});

// =============================================================================
// DIMENSIONS TESTS (2 tests)
// =============================================================================

describe('PDFViewer - Dimensions', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('GIVEN custom width/height WHEN rendered THEN container has correct dimensions', () => {
    render(
      <PDFViewer
        documentUrl="/test.pdf"
        width={1200}
        height={800}
      />
    );

    const container = screen.getByTestId('pdf-viewer');
    expect(container).toHaveStyle({ width: '1200px', height: '800px' });
  });

  test('GIVEN default dimensions WHEN rendered THEN uses 800x600', () => {
    render(<PDFViewer documentUrl="/test.pdf" />);

    const container = screen.getByTestId('pdf-viewer');
    expect(container).toHaveStyle({ width: '800px', height: '600px' });
  });
});
