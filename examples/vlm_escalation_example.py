"""Example: VLM Escalation for Low-Confidence OCR.

Demonstrates using vision language models to improve OCR accuracy
in handwritten or degraded text regions.

Usage:
    python examples/vlm_escalation_example.py image.png
"""

from pathlib import Path
import sys

from ingestforge.ingest.ocr import (
    # Spatial parsing
    parse_ocr_file,
    create_escalator,
    escalate_low_confidence_ocr,
    identify_low_confidence_elements,
)


def main():
    """Run VLM escalation example."""
    if len(sys.argv) < 2:
        print("Usage: python vlm_escalation_example.py image.png")
        sys.exit(1)

    image_path = Path(sys.argv[1])

    # Step 1: Parse OCR output (assumes hOCR file exists)
    hocr_path = image_path.with_suffix(".hocr")
    if not hocr_path.exists():
        print(f"Error: hOCR file not found: {hocr_path}")
        print("Run Tesseract first: tesseract image.png output -l eng hocr")
        sys.exit(1)

    print(f"Parsing OCR from: {hocr_path}")
    ocr_doc = parse_ocr_file(hocr_path)

    if not ocr_doc.pages:
        print("No pages found in OCR output")
        sys.exit(1)

    # Get first page
    page = ocr_doc.pages[0]
    print(f"Found {len(page.elements)} elements on page")

    # Step 2: Identify low-confidence regions
    low_confidence = identify_low_confidence_elements(
        page.elements, confidence_threshold=0.7
    )

    print(f"Found {len(low_confidence)} low-confidence regions")

    if not low_confidence:
        print("All OCR results have high confidence!")
        return

    # Show some low-confidence regions
    print("\nLow-confidence regions:")
    for i, elem in enumerate(low_confidence[:5]):
        print(f"  {i+1}. '{elem.text}' (confidence: {elem.confidence:.2f})")

    # Step 3: Create VLM escalator
    # Note: Requires OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable
    print("\nCreating VLM escalator...")
    escalator = create_escalator(
        provider="openai",  # or "anthropic"
        consent_granted=True,  # User consent for cloud API usage
    )

    if not escalator.config.api_key:
        print("Error: No API key configured")
        print("Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        sys.exit(1)

    # Step 4: Escalate low-confidence regions to VLM
    print(f"\nEscalating {len(low_confidence)} regions to VLM...")
    results = escalate_low_confidence_ocr(
        image_path, page.elements, escalator, confidence_threshold=0.7
    )

    # Step 5: Show results
    print("\nEscalation results:")
    for i, result in enumerate(results):
        if result.success:
            original = result.request.original_text
            corrected = result.vlm_text
            improved = "✓" if result.improved else "✗"
            print(f"  {i+1}. [{improved}] '{original}' → '{corrected}'")
            print(
                f"      Confidence: {result.request.original_confidence:.2f} → {result.vlm_confidence:.2f}"
            )
        else:
            print(f"  {i+1}. [ERROR] {result.error}")

    # Summary statistics
    successful = sum(1 for r in results if r.success)
    improved = sum(1 for r in results if r.improved)

    print("\nSummary:")
    print(f"  Total regions: {len(results)}")
    print(f"  Successful: {successful}")
    print(f"  Improved: {improved}")
    print(f"  Success rate: {successful / len(results) * 100:.1f}%")
    print(f"  Improvement rate: {improved / len(results) * 100:.1f}%")


if __name__ == "__main__":
    main()
