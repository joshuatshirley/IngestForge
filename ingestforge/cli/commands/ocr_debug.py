"""OCR preprocessing debug command (OCR-002.2).

Provides visual verification of image cleanup with --debug-images flag
that saves processed intermediate files.

NASA JPL Commandments compliance:
- Rule #1: Simple control flow, no deep nesting
- Rule #4: Functions <60 lines
- Rule #7: Input validation
- Rule #9: Full type hints

Usage:
    ingestforge ocr-debug input.png --output-dir ./debug
    ingestforge ocr-debug input.png --debug-images
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from ingestforge.core.logging import get_logger
from ingestforge.ingest.ocr.cleanup_core import (
    ImagePreprocessor,
    PreprocessingConfig,
    PreprocessingResult,
    BinarizationMethod,
)

logger = get_logger(__name__)

app = typer.Typer(
    name="ocr-debug",
    help="Debug OCR preprocessing pipeline",
    no_args_is_help=True,
)


def _get_console() -> Console:
    """Get Rich console."""
    return Console()


def _format_size(size: tuple[int, int]) -> str:
    """Format image size for display."""
    return f"{size[0]}x{size[1]}"


def _create_result_panel(
    result: PreprocessingResult,
    console: Console,
) -> Panel:
    """Create a Rich panel for preprocessing result."""
    if result.success:
        content = Text()
        content.append("Status: ", style="dim")
        content.append("SUCCESS", style="bold green")

        content.append("\n\nOriginal Size: ", style="dim")
        content.append(_format_size(result.original_size))

        content.append("\nProcessed Size: ", style="dim")
        content.append(_format_size(result.processed_size))

        if result.rotation_angle != 0.0:
            content.append("\n\nRotation: ", style="dim")
            content.append(f"{result.rotation_angle:.2f}°", style="yellow")

        content.append("\n\nOperations Applied:", style="dim")
        if result.was_binarized:
            content.append("\n  • Binarization", style="green")
        if result.was_denoised:
            content.append("\n  • Denoising", style="green")

        border_style = "green"
    else:
        content = Text()
        content.append("Status: ", style="dim")
        content.append("FAILED", style="bold red")
        content.append(f"\n\nError: {result.error}", style="red")
        border_style = "red"

    return Panel(
        content,
        title=f"[bold]{result.original_path.name}[/bold]",
        border_style=border_style,
        padding=(1, 2),
    )


def _save_debug_images(
    result: PreprocessingResult,
    output_dir: Path,
    preprocessor: ImagePreprocessor,
) -> List[Path]:
    """Save intermediate debug images.

    Returns list of saved file paths.
    """
    saved: List[Path] = []

    if not result.success or result.processed_image is None:
        return saved

    output_dir.mkdir(parents=True, exist_ok=True)

    stem = result.original_path.stem
    suffix = result.original_path.suffix or ".png"

    # Save final processed image
    final_path = output_dir / f"{stem}_processed{suffix}"
    if preprocessor.save_result(result, final_path):
        saved.append(final_path)

    return saved


@app.command("process")
def process_image(
    input_path: Path = typer.Argument(
        ...,
        help="Path to input image file",
        exists=True,
        readable=True,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save processed images",
    ),
    debug_images: bool = typer.Option(
        False,
        "--debug-images",
        "-d",
        help="Save intermediate processed images for debugging",
    ),
    no_deskew: bool = typer.Option(
        False,
        "--no-deskew",
        help="Disable automatic deskew",
    ),
    no_binarize: bool = typer.Option(
        False,
        "--no-binarize",
        help="Disable binarization",
    ),
    no_denoise: bool = typer.Option(
        False,
        "--no-denoise",
        help="Disable denoising",
    ),
    binarization: str = typer.Option(
        "otsu",
        "--binarization",
        "-b",
        help="Binarization method: otsu, adaptive, fixed",
    ),
    threshold: int = typer.Option(
        128,
        "--threshold",
        "-t",
        help="Fixed threshold value (for fixed method)",
        min=0,
        max=255,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-v",
        help="Show detailed output",
    ),
) -> None:
    """Process a single image and show preprocessing results.

    This command helps debug OCR preprocessing by showing what
    operations were applied and optionally saving intermediate
    images.
    """
    console = _get_console()

    # Build config
    try:
        method = BinarizationMethod(binarization.lower())
    except ValueError:
        console.print(f"[red]Invalid binarization method: {binarization}[/red]")
        raise typer.Exit(1)

    config = PreprocessingConfig(
        enable_deskew=not no_deskew,
        enable_binarize=not no_binarize,
        enable_denoise=not no_denoise,
        binarization_method=method,
        threshold=threshold,
    )

    # Process image
    preprocessor = ImagePreprocessor(config)

    if not preprocessor.is_available:
        console.print(
            "[red]OpenCV not available. Install with:[/red]\n"
            "  pip install opencv-python"
        )
        raise typer.Exit(1)

    console.print(f"\nProcessing: [bold]{input_path}[/bold]\n")

    result = preprocessor.process(input_path)

    # Display result
    panel = _create_result_panel(result, console)
    console.print(panel)

    # Save debug images if requested
    if debug_images and result.success:
        out_dir = output_dir or input_path.parent / "ocr_debug"
        saved = _save_debug_images(result, out_dir, preprocessor)

        if saved:
            console.print("\n[bold]Saved Debug Images:[/bold]")
            for path in saved:
                console.print(f"  • {path}")

    if verbose and result.success:
        console.print("\n[dim]Configuration:[/dim]")
        console.print(f"  Deskew: {'enabled' if config.enable_deskew else 'disabled'}")
        console.print(
            f"  Binarize: {'enabled' if config.enable_binarize else 'disabled'}"
        )
        console.print(
            f"  Denoise: {'enabled' if config.enable_denoise else 'disabled'}"
        )
        console.print(f"  Method: {config.binarization_method.value}")


@app.command("batch")
def process_batch(
    input_dir: Path = typer.Argument(
        ...,
        help="Directory containing images to process",
        exists=True,
        dir_okay=True,
        file_okay=False,
    ),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save processed images",
    ),
    pattern: str = typer.Option(
        "*.png",
        "--pattern",
        "-p",
        help="Glob pattern for input files",
    ),
    debug_images: bool = typer.Option(
        False,
        "--debug-images",
        "-d",
        help="Save intermediate processed images",
    ),
) -> None:
    """Process multiple images in a directory.

    Processes all images matching the pattern and shows a
    summary of results.
    """
    console = _get_console()

    # Find images
    images = list(input_dir.glob(pattern))
    if not images:
        console.print(f"[yellow]No images found matching: {pattern}[/yellow]")
        raise typer.Exit(0)

    console.print(f"\nFound {len(images)} images to process\n")

    # Process all
    preprocessor = ImagePreprocessor()

    if not preprocessor.is_available:
        console.print("[red]OpenCV not available.[/red]")
        raise typer.Exit(1)

    results: List[PreprocessingResult] = []

    with console.status("[bold]Processing images..."):
        for image_path in images:
            result = preprocessor.process(image_path)
            results.append(result)

            if debug_images and result.success:
                out_dir = output_dir or input_dir / "ocr_debug"
                _save_debug_images(result, out_dir, preprocessor)

    # Summary table
    table = Table(title="Processing Results")
    table.add_column("File", style="bold")
    table.add_column("Status")
    table.add_column("Size")
    table.add_column("Rotation")

    success_count = 0
    for result in results:
        if result.success:
            success_count += 1
            status = Text("OK", style="green")
            size = _format_size(result.processed_size)
            rotation = f"{result.rotation_angle:.1f}°" if result.rotation_angle else "-"
        else:
            status = Text("FAIL", style="red")
            size = "-"
            rotation = "-"

        table.add_row(
            result.original_path.name,
            status,
            size,
            rotation,
        )

    console.print(table)
    console.print(
        f"\n[bold]Summary:[/bold] {success_count}/{len(results)} processed successfully"
    )


@app.command("info")
def show_info() -> None:
    """Show information about available preprocessing features."""
    console = _get_console()

    preprocessor = ImagePreprocessor()

    table = Table(title="OCR Preprocessing Features")
    table.add_column("Feature", style="bold")
    table.add_column("Status")
    table.add_column("Description")

    # Check OpenCV
    cv2_status = "Available" if preprocessor.cv2 else "Not installed"
    cv2_style = "green" if preprocessor.cv2 else "red"

    table.add_row(
        "OpenCV",
        Text(cv2_status, style=cv2_style),
        "Core image processing library",
    )

    table.add_row(
        "Deskew",
        Text("Available", style="green")
        if preprocessor.is_available
        else Text("Unavailable", style="red"),
        "Automatic rotation correction via Hough transform",
    )

    table.add_row(
        "Binarization",
        Text("Available", style="green")
        if preprocessor.is_available
        else Text("Unavailable", style="red"),
        "Otsu, Adaptive, and Fixed threshold methods",
    )

    table.add_row(
        "Denoising",
        Text("Available", style="green")
        if preprocessor.is_available
        else Text("Unavailable", style="red"),
        "Non-local means denoising",
    )

    console.print(table)

    if not preprocessor.is_available:
        console.print("\n[yellow]To enable all features, install OpenCV:[/yellow]")
        console.print("  pip install opencv-python")


# Register with main CLI if this is imported
def register_commands(parent_app: typer.Typer) -> None:
    """Register OCR debug commands with parent app."""
    parent_app.add_typer(app, name="ocr-debug")
