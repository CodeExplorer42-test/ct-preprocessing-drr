"""Isotropic resampling module for CT preprocessing."""

from typing import Optional, Tuple
import numpy as np
import SimpleITK as sitk
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

console = Console()


def resample_to_isotropic(
    image: sitk.Image,
    target_spacing: float = 0.5,
    interpolator: int = sitk.sitkBSpline,
    default_pixel_value: float = -1000.0,
    show_progress: bool = True
) -> Tuple[sitk.Image, dict]:
    """
    Resample a CT image to isotropic spacing.
    
    Args:
        image: Input SimpleITK image
        target_spacing: Target isotropic spacing in mm (default: 0.5)
        interpolator: SimpleITK interpolator type (default: B-spline)
        default_pixel_value: Value for pixels outside the image (default: -1000 HU for air)
        
    Returns:
        Tuple of (resampled image, resampling info dict)
    """
    # Get original properties
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    original_origin = image.GetOrigin()
    original_direction = image.GetDirection()
    
    # Calculate new size based on target spacing
    new_size = [
        int(round(orig_size * orig_spacing / target_spacing))
        for orig_size, orig_spacing in zip(original_size, original_spacing)
    ]
    
    # Create resampler
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([target_spacing] * image.GetDimension())
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(original_direction)
    resampler.SetOutputOrigin(original_origin)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_pixel_value)
    resampler.SetInterpolator(interpolator)
    
    # Perform resampling (SimpleITK operations are atomic, so progress bar isn't meaningful)
    if show_progress:
        console.print("  Resampling to isotropic spacing...")
    resampled_image = resampler.Execute(image)
    
    # Collect resampling information
    info = {
        "original_spacing": original_spacing,
        "original_size": original_size,
        "new_spacing": resampled_image.GetSpacing(),
        "new_size": resampled_image.GetSize(),
        "target_spacing": target_spacing,
        "interpolator": get_interpolator_name(interpolator),
        "memory_increase_factor": np.prod(new_size) / np.prod(original_size)
    }
    
    return resampled_image, info


def get_interpolator_name(interpolator: int) -> str:
    """Get human-readable name for interpolator."""
    interpolator_names = {
        sitk.sitkNearestNeighbor: "Nearest Neighbor",
        sitk.sitkLinear: "Linear",
        sitk.sitkBSpline: "B-Spline",
        sitk.sitkGaussian: "Gaussian",
        sitk.sitkLabelGaussian: "Label Gaussian",
        sitk.sitkHammingWindowedSinc: "Hamming Windowed Sinc",
        sitk.sitkCosineWindowedSinc: "Cosine Windowed Sinc",
        sitk.sitkWelchWindowedSinc: "Welch Windowed Sinc",
        sitk.sitkLanczosWindowedSinc: "Lanczos Windowed Sinc",
        sitk.sitkBlackmanWindowedSinc: "Blackman Windowed Sinc",
    }
    return interpolator_names.get(interpolator, "Unknown")


def print_resampling_info(info: dict) -> None:
    """Print formatted resampling information."""
    console.print("\n[bold cyan]Resampling Information[/bold cyan]")
    
    console.print(f"Interpolator: [green]{info['interpolator']}[/green]")
    console.print(f"Target spacing: [green]{info['target_spacing']} mm[/green]")
    
    console.print("\n[bold]Original:[/bold]")
    console.print(f"  Size: {info['original_size']}")
    console.print(f"  Spacing: ({info['original_spacing'][0]:.3f}, {info['original_spacing'][1]:.3f}, {info['original_spacing'][2]:.3f}) mm")
    
    console.print("\n[bold]Resampled:[/bold]")
    console.print(f"  Size: {info['new_size']}")
    console.print(f"  Spacing: ({info['new_spacing'][0]:.3f}, {info['new_spacing'][1]:.3f}, {info['new_spacing'][2]:.3f}) mm")
    
    console.print(f"\n[yellow]Memory increase factor: {info['memory_increase_factor']:.2f}x[/yellow]")


def compare_interpolators(
    image: sitk.Image,
    target_spacing: float = 0.5,
    slice_idx: Optional[int] = None
) -> dict:
    """
    Compare different interpolation methods on a sample slice.
    
    Args:
        image: Input SimpleITK image
        target_spacing: Target isotropic spacing
        slice_idx: Slice index to compare (default: middle slice)
        
    Returns:
        Dictionary of resampled images with different interpolators
    """
    interpolators = {
        "Linear": sitk.sitkLinear,
        "B-Spline": sitk.sitkBSpline,
        "Lanczos-3": sitk.sitkLanczosWindowedSinc,
    }
    
    results = {}
    
    console.print("\n[bold]Comparing interpolators...[/bold]")
    for name, interpolator in interpolators.items():
        console.print(f"  Processing {name}...")
        resampled, info = resample_to_isotropic(
            image, target_spacing, interpolator, show_progress=False
        )
        results[name] = {
            "image": resampled,
            "info": info
        }
    
    return results


if __name__ == "__main__":
    from pathlib import Path
    from dicom_loader import load_dicom_series, analyze_ct_characteristics
    
    # Load test data
    dicom_dir = Path("1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192")
    
    if dicom_dir.exists():
        console.print("[green]Loading DICOM series...[/green]")
        image, metadata = load_dicom_series(dicom_dir)
        
        # Test different target spacings
        for target_spacing in [0.5, 0.4]:
            console.print(f"\n[bold]Testing with target spacing: {target_spacing} mm[/bold]")
            
            # Resample with B-spline (recommended default)
            resampled, info = resample_to_isotropic(image, target_spacing)
            print_resampling_info(info)
            
            # Analyze resampled image
            stats = analyze_ct_characteristics(resampled)
            console.print(f"Resampled memory usage: [yellow]{stats['memory_mb']:.1f} MB[/yellow]")
        
        # Save a test slice for visualization
        console.print("\n[bold cyan]Saving test results[/bold cyan]")
        import numpy as np
        
        # Extract middle slice from original and resampled
        original_array = sitk.GetArrayFromImage(image)
        resampled_array = sitk.GetArrayFromImage(resampled)
        
        mid_slice_original = original_array[original_array.shape[0]//2]
        mid_slice_resampled = resampled_array[resampled_array.shape[0]//2]
        
        console.print(f"Original middle slice shape: {mid_slice_original.shape}")
        console.print(f"Resampled middle slice shape: {mid_slice_resampled.shape}")
        console.print("\n[green]Resampling complete![/green]")
    else:
        console.print("[red]DICOM directory not found![/red]")