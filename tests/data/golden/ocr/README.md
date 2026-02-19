# OCR Golden Dataset

## Purpose

This directory contains "golden" reference outputs for OCR testing. These files represent the expected markdown output when processing specific test PDF documents through IngestForge's OCR pipeline.

## Dataset Structure

- **sample_1col.md**: Expected output for single-column academic paper format
- **sample_2col.md**: Expected output for two-column journal article format
- **sample_table.md**: Expected output for document containing tabular data

## Usage

Golden datasets are used in regression testing to verify that OCR output quality remains consistent across code changes. Tests compare actual OCR output against these reference files using structural and content similarity metrics.

## Maintenance

When updating OCR algorithms or dependencies:
1. Verify that changes improve output quality
2. Update golden files if new output is objectively better
3. Document why golden files were updated in commit messages

## Test Methodology

Tests use golden files with these validation approaches:
- **Exact matching**: For critical structured content (tables, headers)
- **Similarity threshold**: For prose content (allows minor OCR variations)
- **Structure validation**: Verify expected markdown elements are present

## Note on Source PDFs

The actual PDF files corresponding to these golden outputs are not stored in this repository due to size and licensing constraints. Contact the test infrastructure maintainer for access to source test files.
