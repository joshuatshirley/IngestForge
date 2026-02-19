# VLM Escalation for OCR Enhancement

Vision Language Model (VLM) escalation improves OCR accuracy by sending low-confidence text regions to advanced vision models like GPT-4o-mini or Claude Vision.

## Overview

Traditional OCR engines struggle with:
- Handwritten text
- Degraded or low-quality scans
- Unusual fonts or styles
- Complex layouts

VLM escalation addresses these issues by:
1. Identifying low-confidence OCR regions
2. Cropping image regions around problem areas
3. Sending crops to vision language models
4. Merging improved results back into OCR output

## Architecture

```
┌─────────────────┐
│  Original Image │
└────────┬────────┘
         │
         ↓
┌─────────────────┐
│ Tesseract/OCR   │ ← Initial OCR with confidence scores
└────────┬────────┘
         │
         ↓
┌─────────────────────────┐
│ Confidence Evaluator    │ ← Identify low-confidence regions
│ (threshold: 0.7)        │
└────────┬────────────────┘
         │
         ↓
┌─────────────────────────┐
│ Image Cropper           │ ← Crop problem regions with padding
└────────┬────────────────┘
         │
         ↓
┌─────────────────────────┐
│ VLM Escalator           │
│ - OpenAI GPT-4o-mini    │ ← Send to vision model
│ - Claude Vision         │
└────────┬────────────────┘
         │
         ↓
┌─────────────────────────┐
│ Result Merger           │ ← Merge improved text back
└────────┬────────────────┘
         │
         ↓
┌─────────────────────────┐
│ Enhanced OCR Output     │
└─────────────────────────┘
```

## Usage

### Basic Usage

```python
from pathlib import Path
from ingestforge.ingest.ocr import (
    parse_ocr_file,
    create_escalator,
    escalate_low_confidence_ocr,
)

# Parse OCR output
ocr_doc = parse_ocr_file(Path("document.hocr"))
page = ocr_doc.pages[0]

# Create escalator
escalator = create_escalator(
    provider="openai",  # or "anthropic"
    api_key="your-api-key",  # or set OPENAI_API_KEY env var
    consent_granted=True
)

# Escalate low-confidence regions
results = escalate_low_confidence_ocr(
    image_path=Path("document.png"),
    ocr_elements=page.elements,
    escalator=escalator,
    confidence_threshold=0.7  # Escalate if < 70% confidence
)

# Process results
for result in results:
    if result.improved:
        print(f"Improved: {result.request.original_text} → {result.vlm_text}")
```

### Manual Control

```python
from ingestforge.ingest.ocr import (
    identify_low_confidence_elements,
    escalate_elements,
)

# Manually identify low-confidence regions
low_conf = identify_low_confidence_elements(
    page.elements,
    confidence_threshold=0.7
)

print(f"Found {len(low_conf)} low-confidence regions")

# Escalate specific elements
results = escalate_elements(low_conf, image_path, escalator)
```

### Provider Configuration

```python
# OpenAI GPT-4o-mini (fast, cost-effective)
escalator = create_escalator(
    provider="openai",
    api_key=os.environ["OPENAI_API_KEY"],
    consent_granted=True
)

# Anthropic Claude Vision (high quality)
escalator = create_escalator(
    provider="anthropic",
    api_key=os.environ["ANTHROPIC_API_KEY"],
    consent_granted=True
)

# Custom configuration
from ingestforge.ingest.ocr import EscalatorConfig, VLMProvider, VLMEscalator

config = EscalatorConfig(
    provider=VLMProvider.OPENAI,
    api_key="your-key",
    model_name="gpt-4o",  # Use full GPT-4o for highest quality
    max_tokens=500,
    temperature=0.0,  # Deterministic output
    crop_padding=30,  # Extra context around region
)

escalator = VLMEscalator(config=config)
```

## Safety Features

### User Consent

VLM escalation requires explicit user consent before sending images to cloud APIs:

```python
# Deny by default (safe)
escalator = VLMEscalator
assert escalator.check_consent is False

# Explicit consent required
escalator = create_escalator(consent_granted=True)
assert escalator.check_consent is True

# Interactive consent callback
def ask_user_consent -> bool:
    response = input("Send image crops to OpenAI? (y/n): ")
    return response.lower == "y"

escalator.set_consent_callback(ask_user_consent)
escalator.request_consent  # Calls callback
```

### Fixed Bounds (JPL Rule #2)

All operations have fixed upper bounds to prevent resource exhaustion:

```python
MAX_ESCALATIONS_PER_BATCH = 50      # Max regions per batch
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10MB max
MAX_CROP_DIMENSION = 4096            # 4K max dimension
DEFAULT_PADDING_PIXELS = 20          # Crop padding
```

### Input Validation (JPL Rule #7)

All inputs are validated before processing:

```python
# Invalid threshold clamped to safe value
low_conf = identify_low_confidence_elements(
    elements,
    confidence_threshold=1.5  # > 1.0, will use 0.7 default
)

# Missing image data handled gracefully
result = escalator.escalate_single(request)
# Returns EscalationResult with success=False, error message
```

## Performance Considerations

### Cost Optimization

- **Threshold selection**: Higher threshold (e.g., 0.8) = fewer API calls
- **Batch processing**: Process multiple pages together
- **Provider selection**: GPT-4o-mini is ~10x cheaper than GPT-4o

```python
# Conservative: Only escalate very low confidence
results = escalate_low_confidence_ocr(
    image_path, elements, escalator,
    confidence_threshold=0.5  # Only if < 50%
)

# Aggressive: Escalate anything uncertain
results = escalate_low_confidence_ocr(
    image_path, elements, escalator,
    confidence_threshold=0.9  # If < 90%
)
```

### Latency

- GPT-4o-mini: ~500-1000ms per region
- Claude Haiku: ~300-800ms per region
- Batch processing recommended for >10 regions

## Testing

Run the test suite:

```bash
pytest tests/unit/ingest/ocr/test_vlm_escalator.py -v
pytest tests/unit/ingest/ocr/test_vlm_clients.py -v
pytest tests/unit/ingest/ocr/test_image_cropper.py -v
```

Test with stub client (no API calls):

```python
from ingestforge.ingest.ocr import VLMClientStub, VLMEscalator, EscalatorConfig

stub = VLMClientStub(
    response_text="Corrected text",
    response_confidence=0.95
)

escalator = VLMEscalator(
    config=EscalatorConfig(consent_status=ConsentStatus.GRANTED),
    client=stub
)

# No actual API calls made
results = escalator.escalate_batch(requests)
```

## Compliance

This implementation follows NASA JPL Power of 10 Commandments:

- **Rule #1**: Max 3 nesting levels, early returns
- **Rule #2**: Fixed upper bounds on all operations
- **Rule #4**: All functions <60 lines
- **Rule #7**: Input validation before operations
- **Rule #9**: Complete type hints

Example compliance check:

```python
def escalate_single(self, request: EscalationRequest) -> EscalationResult:
    # Level 1: Early return on consent check
    if not self.check_consent:
        return EscalationResult(request=request, success=False, error="...")
    
    # Level 2: Early return on validation
    if not self._validate_request(request):
        return EscalationResult(request=request, success=False, error="...")
    
    # Level 3: Try/except for API call
    try:
        text, confidence = client.extract_text(image_data, prompt)
        return EscalationResult(request=request, success=True, ...)
    except Exception as e:
        return EscalationResult(request=request, success=False, error=str(e))
requests[:MAX_ESCALATIONS_PER_BATCH]  # Always limited
if not 0.0 <= confidence_threshold <= 1.0:
    logger.warning(f"Invalid threshold, using 0.7")
    confidence_threshold = 0.7
```

## References

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [OpenAI Vision API](https://platform.openai.com/docs/guides/vision)
- [Anthropic Claude Vision](https://docs.anthropic.com/claude/docs/vision)
- [hOCR Format](https://github.com/kba/hocr-spec)
