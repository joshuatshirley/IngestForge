"""Tests for image cropping utilities.

Tests image region cropping for VLM escalation."""

from __future__ import annotations

from pathlib import Path

import pytest

from ingestforge.ingest.ocr.spatial_parser import BoundingBox
from ingestforge.ingest.ocr.image_cropper import (
    crop_image_region,
    crop_pil_image,
    load_and_crop_regions,
)

pytest.importorskip("PIL", reason="PIL/Pillow required for image cropping")


class TestCropImageRegion:
    """Tests for crop_image_region function."""

    def test_crop_nonexistent_file(self, tmp_path: Path) -> None:
        """Test cropping from non-existent file returns None."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        result = crop_image_region(tmp_path / "missing.png", bbox)
        assert result is None

    def test_crop_valid_region(self, tmp_path: Path) -> None:
        """Test cropping a valid region."""
        from PIL import Image

        # Create test image
        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (200, 200), color="white")
        img.save(img_path)

        bbox = BoundingBox(x1=50, y1=50, x2=150, y2=150)
        result = crop_image_region(img_path, bbox, padding=10)

        assert result is not None
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_crop_with_padding(self, tmp_path: Path) -> None:
        """Test crop respects padding parameter."""
        from PIL import Image

        img_path = tmp_path / "test.png"
        img = Image.new("RGB", (200, 200), color="red")
        img.save(img_path)

        bbox = BoundingBox(x1=100, y1=100, x2=120, y2=120)
        result = crop_image_region(img_path, bbox, padding=20)

        assert result is not None


class TestCropPilImage:
    """Tests for crop_pil_image function."""

    def test_crop_none_image(self) -> None:
        """Test cropping None image returns None."""
        bbox = BoundingBox(x1=0, y1=0, x2=100, y2=50)
        result = crop_pil_image(None, bbox)
        assert result is None

    def test_crop_at_image_edge(self) -> None:
        """Test crop at edge doesn't go negative."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="blue")
        bbox = BoundingBox(x1=5, y1=5, x2=50, y2=50)
        result = crop_pil_image(img, bbox, padding=20)

        assert result is not None

    def test_crop_invalid_box(self) -> None:
        """Test invalid crop box returns None."""
        from PIL import Image

        img = Image.new("RGB", (100, 100), color="green")
        # Invalid: x1 >= x2
        bbox = BoundingBox(x1=100, y1=0, x2=50, y2=50)
        result = crop_pil_image(img, bbox)

        assert result is None


class TestLoadAndCropRegions:
    """Tests for load_and_crop_regions function."""

    def test_crop_multiple_regions(self, tmp_path: Path) -> None:
        """Test cropping multiple regions at once."""
        from PIL import Image

        img_path = tmp_path / "multi.png"
        img = Image.new("RGB", (300, 300), color="yellow")
        img.save(img_path)

        regions = [
            BoundingBox(x1=10, y1=10, x2=50, y2=50),
            BoundingBox(x1=100, y1=100, x2=150, y2=150),
            BoundingBox(x1=200, y1=200, x2=250, y2=250),
        ]

        results = load_and_crop_regions(img_path, regions)

        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_crop_missing_file(self, tmp_path: Path) -> None:
        """Test cropping from missing file returns Nones."""
        regions = [BoundingBox(x1=0, y1=0, x2=100, y2=50)]
        results = load_and_crop_regions(tmp_path / "missing.png", regions)

        assert len(results) == 1
        assert results[0] is None
