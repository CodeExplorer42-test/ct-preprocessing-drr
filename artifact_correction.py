"""Artifact correction module for CT preprocessing."""

from typing import Tuple, Dict, Optional
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from scipy.signal import medfilt2d
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

console = Console()


def detect_metal_regions(image: sitk.Image, threshold: float = 3000.0) -> Tuple[sitk.Image, Dict]:
    """
    Detect metal regions in CT image using threshold-based segmentation.
    
    Args:
        image: Input SimpleITK image
        threshold: HU threshold for metal detection
        
    Returns:
        Tuple of (metal mask, info dict)
    """
    array = sitk.GetArrayFromImage(image)
    
    # Create metal mask
    metal_mask = array > threshold
    
    # Apply morphological operations to clean up mask
    metal_mask = ndimage.binary_erosion(metal_mask, iterations=1)
    metal_mask = ndimage.binary_dilation(metal_mask, iterations=2)
    metal_mask = ndimage.binary_erosion(metal_mask, iterations=1)
    
    # Count metal voxels
    num_metal_voxels = np.sum(metal_mask)
    metal_percentage = (num_metal_voxels / metal_mask.size) * 100
    
    info = {
        "threshold": threshold,
        "num_metal_voxels": int(num_metal_voxels),
        "metal_percentage": float(metal_percentage),
        "contains_metal": metal_percentage > 0.01  # More than 0.01% is significant
    }
    
    # Convert to SimpleITK image
    mask_image = sitk.GetImageFromArray(metal_mask.astype(np.uint8))
    mask_image.CopyInformation(image)
    
    return mask_image, info


def reduce_metal_artifacts_simple(image: sitk.Image, metal_threshold: float = 3000.0) -> Tuple[sitk.Image, Dict]:
    """
    Simple metal artifact reduction using interpolation.
    
    Args:
        image: Input SimpleITK image
        metal_threshold: HU threshold for metal detection
        
    Returns:
        Tuple of (corrected image, info dict)
    """
    # Detect metal regions
    metal_mask, mask_info = detect_metal_regions(image, metal_threshold)
    
    if not mask_info["contains_metal"]:
        console.print("[green]No significant metal artifacts detected[/green]")
        return image, {"method": "none", "reason": "no metal detected"}
    
    console.print(f"[yellow]Metal detected: {mask_info['metal_percentage']:.2f}% of volume[/yellow]")
    
    array = sitk.GetArrayFromImage(image)
    mask_array = sitk.GetArrayFromImage(metal_mask).astype(bool)
    
    # Create corrected array
    corrected = array.copy()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Reducing metal artifacts...", total=array.shape[0])
        
        # Process slice by slice
        for i in range(array.shape[0]):
            if np.any(mask_array[i]):
                # Interpolate metal regions using surrounding tissue
                slice_data = corrected[i].copy()
                metal_slice = mask_array[i]
                
                # Dilate mask slightly for interpolation
                dilated_mask = ndimage.binary_dilation(metal_slice, iterations=3)
                
                # Get surrounding tissue values
                surrounding = slice_data[dilated_mask & ~metal_slice]
                if len(surrounding) > 0:
                    # Replace metal with median of surrounding tissue
                    replacement_value = np.median(surrounding)
                else:
                    # Fallback to global median
                    replacement_value = np.median(slice_data[~metal_slice])
                
                corrected[i][metal_slice] = replacement_value
            
            progress.update(task, advance=1)
    
    # Create output image
    corrected_image = sitk.GetImageFromArray(corrected)
    corrected_image.CopyInformation(image)
    
    info = {
        "method": "Simple Metal Artifact Reduction",
        "metal_threshold": metal_threshold,
        "metal_percentage": mask_info["metal_percentage"],
        "voxels_corrected": mask_info["num_metal_voxels"]
    }
    
    return corrected_image, info


def correct_beam_hardening(image: sitk.Image, correction_strength: float = 0.5) -> Tuple[sitk.Image, Dict]:
    """
    Apply beam hardening correction using polynomial correction.
    
    Args:
        image: Input SimpleITK image
        correction_strength: Strength of correction (0-1)
        
    Returns:
        Tuple of (corrected image, info dict)
    """
    array = sitk.GetArrayFromImage(image)
    
    # Define water and bone HU values
    water_hu = 0
    bone_hu = 1000
    
    # Apply polynomial correction
    # This is a simplified version - real implementation would use calibration data
    corrected = array.copy()
    
    # Only correct positive HU values (water to bone range)
    mask = (array > water_hu) & (array < bone_hu)
    
    if np.any(mask):
        # Normalize to [0, 1] range
        norm_values = (array[mask] - water_hu) / (bone_hu - water_hu)
        
        # Apply quadratic correction
        # BH causes underestimation in middle ranges
        correction = correction_strength * norm_values * (1 - norm_values) * 4
        
        # Apply correction
        corrected[mask] = array[mask] + correction * (bone_hu - water_hu)
    
    # Create output image
    corrected_image = sitk.GetImageFromArray(corrected)
    corrected_image.CopyInformation(image)
    
    info = {
        "method": "Polynomial Beam Hardening Correction",
        "correction_strength": correction_strength,
        "voxels_corrected": int(np.sum(mask))
    }
    
    return corrected_image, info


def remove_ring_artifacts(image: sitk.Image, filter_size: int = 5) -> Tuple[sitk.Image, Dict]:
    """
    Remove ring artifacts using polar coordinate filtering.
    
    Args:
        image: Input SimpleITK image
        filter_size: Size of median filter kernel
        
    Returns:
        Tuple of (corrected image, info dict)
    """
    array = sitk.GetArrayFromImage(image)
    corrected = np.zeros_like(array)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Removing ring artifacts...", total=array.shape[0])
        
        for i in range(array.shape[0]):
            slice_data = array[i]
            
            # Find center of image
            center_y, center_x = np.array(slice_data.shape) // 2
            
            # Create coordinate grids
            y, x = np.ogrid[:slice_data.shape[0], :slice_data.shape[1]]
            
            # Convert to polar coordinates
            r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            theta = np.arctan2(y - center_y, x - center_x)
            
            # Create polar image
            max_radius = int(np.min([center_x, center_y]))
            polar_shape = (360, max_radius)
            polar_image = np.zeros(polar_shape)
            
            # Simple polar transformation
            for t_idx in range(polar_shape[0]):
                angle = t_idx * np.pi / 180
                for r_idx in range(polar_shape[1]):
                    x_coord = int(center_x + r_idx * np.cos(angle))
                    y_coord = int(center_y + r_idx * np.sin(angle))
                    
                    if 0 <= x_coord < slice_data.shape[1] and 0 <= y_coord < slice_data.shape[0]:
                        polar_image[t_idx, r_idx] = slice_data[y_coord, x_coord]
            
            # Apply median filter along angular direction
            filtered_polar = medfilt2d(polar_image, kernel_size=(filter_size, 1))
            
            # Convert back to Cartesian
            corrected_slice = slice_data.copy()
            
            # Simple inverse transformation
            for t_idx in range(polar_shape[0]):
                angle = t_idx * np.pi / 180
                for r_idx in range(polar_shape[1]):
                    x_coord = int(center_x + r_idx * np.cos(angle))
                    y_coord = int(center_y + r_idx * np.sin(angle))
                    
                    if 0 <= x_coord < slice_data.shape[1] and 0 <= y_coord < slice_data.shape[0]:
                        # Blend original and filtered based on ring artifact strength
                        diff = abs(polar_image[t_idx, r_idx] - filtered_polar[t_idx, r_idx])
                        if diff > 50:  # Threshold for ring artifact detection
                            corrected_slice[y_coord, x_coord] = filtered_polar[t_idx, r_idx]
            
            corrected[i] = corrected_slice
            progress.update(task, advance=1)
    
    # Create output image
    corrected_image = sitk.GetImageFromArray(corrected)
    corrected_image.CopyInformation(image)
    
    info = {
        "method": "Polar Coordinate Ring Artifact Removal",
        "filter_size": filter_size
    }
    
    return corrected_image, info


def apply_artifact_correction(
    image: sitk.Image,
    correct_metal: bool = True,
    correct_beam_hardening: bool = True,
    correct_rings: bool = False
) -> Tuple[sitk.Image, Dict]:
    """
    Apply multiple artifact correction methods.
    
    Args:
        image: Input SimpleITK image
        correct_metal: Apply metal artifact reduction
        correct_beam_hardening: Apply beam hardening correction
        correct_rings: Apply ring artifact removal
        
    Returns:
        Tuple of (corrected image, info dict)
    """
    corrected = image
    info = {"corrections_applied": []}
    
    if correct_metal:
        console.print("\n[bold]Checking for metal artifacts...[/bold]")
        corrected, metal_info = reduce_metal_artifacts_simple(corrected)
        if metal_info.get("method", "none") != "none":
            info["corrections_applied"].append("metal")
            info["metal_correction"] = metal_info
    
    if correct_beam_hardening:
        console.print("\n[bold]Applying beam hardening correction...[/bold]")
        corrected, bh_info = correct_beam_hardening(corrected)
        info["corrections_applied"].append("beam_hardening")
        info["beam_hardening_correction"] = bh_info
    
    if correct_rings:
        console.print("\n[bold]Checking for ring artifacts...[/bold]")
        corrected, ring_info = remove_ring_artifacts(corrected)
        info["corrections_applied"].append("rings")
        info["ring_correction"] = ring_info
    
    return corrected, info


if __name__ == "__main__":
    from pathlib import Path
    from dicom_loader import load_dicom_series
    
    # Load test data
    dicom_dir = Path("1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192")
    
    if dicom_dir.exists():
        console.print("[green]Loading DICOM series...[/green]")
        image, metadata = load_dicom_series(dicom_dir)
        
        # Apply artifact corrections
        console.print("\n[bold cyan]Applying Artifact Corrections[/bold cyan]")
        corrected, info = apply_artifact_correction(
            image,
            correct_metal=True,
            correct_beam_hardening=True,
            correct_rings=False  # Ring artifacts are less common
        )
        
        console.print(f"\n[green]Corrections applied: {', '.join(info['corrections_applied'])}[/green]")
        
        # Compare statistics
        original_array = sitk.GetArrayFromImage(image)
        corrected_array = sitk.GetArrayFromImage(corrected)
        
        console.print("\n[bold]Statistics Comparison:[/bold]")
        console.print(f"Original - Mean: {np.mean(original_array):.1f}, Std: {np.std(original_array):.1f}")
        console.print(f"Corrected - Mean: {np.mean(corrected_array):.1f}, Std: {np.std(corrected_array):.1f}")
    else:
        console.print("[red]DICOM directory not found![/red]")