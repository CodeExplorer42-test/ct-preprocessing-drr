"""Quality control metrics and reporting module for CT preprocessing."""

from typing import Dict, List, Tuple, Optional
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity as ssim
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import json
from pathlib import Path
from datetime import datetime

console = Console()


def calculate_edge_preservation_index(
    original: sitk.Image,
    processed: sitk.Image,
    edge_threshold: float = 100.0
) -> float:
    """
    Calculate edge preservation index between original and processed images.
    
    Args:
        original: Original image
        processed: Processed image
        edge_threshold: Gradient threshold for edge detection
        
    Returns:
        Edge preservation index (0-1, higher is better)
    """
    # Check if dimensions match
    if original.GetSize() != processed.GetSize():
        # Resample original to match processed dimensions
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(processed.GetSize())
        resampler.SetOutputSpacing(processed.GetSpacing())
        resampler.SetOutputOrigin(processed.GetOrigin())
        resampler.SetOutputDirection(processed.GetDirection())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(-1000)
        resampler.SetInterpolator(sitk.sitkLinear)
        original = resampler.Execute(original)
    
    # Convert to arrays
    orig_array = sitk.GetArrayFromImage(original)
    proc_array = sitk.GetArrayFromImage(processed)
    
    # Calculate gradients using Sobel filter
    orig_grad = sitk.GetArrayFromImage(sitk.SobelEdgeDetection(original))
    proc_grad = sitk.GetArrayFromImage(sitk.SobelEdgeDetection(processed))
    
    # Find edge regions (high gradient)
    edge_mask = orig_grad > edge_threshold
    
    if np.sum(edge_mask) == 0:
        return 1.0  # No edges found
    
    # Calculate mean gradient in edge regions
    orig_edge_mean = np.mean(orig_grad[edge_mask])
    proc_edge_mean = np.mean(proc_grad[edge_mask])
    
    # Edge preservation index
    epi = min(proc_edge_mean / orig_edge_mean, 1.0) if orig_edge_mean > 0 else 1.0
    
    return float(epi)


def calculate_ssim_3d(
    original: sitk.Image,
    processed: sitk.Image,
    sample_slices: int = 10
) -> float:
    """
    Calculate SSIM for 3D images by sampling slices.
    
    Args:
        original: Original image
        processed: Processed image
        sample_slices: Number of slices to sample
        
    Returns:
        Mean SSIM value
    """
    # Check if dimensions match
    if original.GetSize() != processed.GetSize():
        # Resample original to match processed dimensions
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(processed.GetSize())
        resampler.SetOutputSpacing(processed.GetSpacing())
        resampler.SetOutputOrigin(processed.GetOrigin())
        resampler.SetOutputDirection(processed.GetDirection())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(-1000)
        resampler.SetInterpolator(sitk.sitkLinear)  # Fast interpolation for QC
        original = resampler.Execute(original)
    
    orig_array = sitk.GetArrayFromImage(original)
    proc_array = sitk.GetArrayFromImage(processed)
    
    # Sample slices evenly
    num_slices = orig_array.shape[0]
    slice_indices = np.linspace(0, num_slices - 1, sample_slices, dtype=int)
    
    ssim_values = []
    for idx in slice_indices:
        # Normalize slices to [0, 1] range
        orig_slice = orig_array[idx].astype(np.float32)
        proc_slice = proc_array[idx].astype(np.float32)
        
        # Compute data range
        data_range = max(orig_slice.max() - orig_slice.min(), 
                        proc_slice.max() - proc_slice.min())
        
        if data_range > 0:
            ssim_val = ssim(orig_slice, proc_slice, data_range=data_range)
            ssim_values.append(ssim_val)
    
    return float(np.mean(ssim_values)) if ssim_values else 0.0


def calculate_nrmse(original: sitk.Image, processed: sitk.Image) -> float:
    """
    Calculate Normalized Root Mean Square Error.
    
    Args:
        original: Original image
        processed: Processed image
        
    Returns:
        NRMSE value (lower is better)
    """
    # Check if dimensions match
    if original.GetSize() != processed.GetSize():
        # Resample original to match processed dimensions
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(processed.GetSize())
        resampler.SetOutputSpacing(processed.GetSpacing())
        resampler.SetOutputOrigin(processed.GetOrigin())
        resampler.SetOutputDirection(processed.GetDirection())
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(-1000)
        resampler.SetInterpolator(sitk.sitkLinear)
        original = resampler.Execute(original)
    
    orig_array = sitk.GetArrayFromImage(original).astype(np.float32)
    proc_array = sitk.GetArrayFromImage(processed).astype(np.float32)
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((orig_array - proc_array) ** 2))
    
    # Normalize by range of original
    data_range = orig_array.max() - orig_array.min()
    nrmse = rmse / data_range if data_range > 0 else 0.0
    
    return float(nrmse)


def check_hu_accuracy(image: sitk.Image) -> Dict[str, float]:
    """
    Check HU accuracy in reference regions.
    
    Args:
        image: Input image in HU
        
    Returns:
        Dictionary of tissue HU measurements
    """
    array = sitk.GetArrayFromImage(image)
    
    results = {}
    
    # Check air
    air_mask = (array > -1100) & (array < -900)
    if np.any(air_mask):
        results["air_hu"] = float(np.mean(array[air_mask]))
        results["air_deviation"] = abs(results["air_hu"] - (-1000))
    
    # Check fat
    fat_mask = (array > -150) & (array < -50)
    if np.any(fat_mask):
        results["fat_hu"] = float(np.mean(array[fat_mask]))
        results["fat_deviation"] = abs(results["fat_hu"] - (-100))
    
    # Check water/blood
    water_mask = (array > -50) & (array < 50)
    if np.any(water_mask):
        results["water_hu"] = float(np.mean(array[water_mask]))
        results["water_deviation"] = abs(results["water_hu"] - 0)
    
    return results


def generate_qc_report(
    original_image: sitk.Image,
    processed_image: sitk.Image,
    processing_info: Dict,
    output_path: Optional[Path] = None
) -> Dict:
    """
    Generate comprehensive QC report.
    
    Args:
        original_image: Original input image
        processed_image: Final processed image
        processing_info: Dictionary with processing step information
        output_path: Optional path to save report
        
    Returns:
        QC metrics dictionary
    """
    console.print("\n[bold cyan]Generating Quality Control Report[/bold cyan]")
    
    # Calculate metrics
    console.print("Calculating QC metrics...")
    
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "processing_info": processing_info,
        "metrics": {}
    }
    
    # SSIM
    console.print("  Computing SSIM...")
    metrics["metrics"]["ssim"] = calculate_ssim_3d(original_image, processed_image)
    
    # Edge Preservation
    console.print("  Computing Edge Preservation Index...")
    metrics["metrics"]["edge_preservation"] = calculate_edge_preservation_index(
        original_image, processed_image
    )
    
    # NRMSE
    console.print("  Computing NRMSE...")
    metrics["metrics"]["nrmse"] = calculate_nrmse(original_image, processed_image)
    
    # HU Accuracy (if image is in HU)
    console.print("  Checking HU accuracy...")
    metrics["metrics"]["hu_accuracy"] = check_hu_accuracy(processed_image)
    
    # Data statistics
    orig_array = sitk.GetArrayFromImage(original_image)
    proc_array = sitk.GetArrayFromImage(processed_image)
    
    metrics["data_stats"] = {
        "original": {
            "shape": original_image.GetSize(),
            "spacing": original_image.GetSpacing(),
            "min": float(np.min(orig_array)),
            "max": float(np.max(orig_array)),
            "mean": float(np.mean(orig_array)),
            "std": float(np.std(orig_array))
        },
        "processed": {
            "shape": processed_image.GetSize(),
            "spacing": processed_image.GetSpacing(),
            "min": float(np.min(proc_array)),
            "max": float(np.max(proc_array)),
            "mean": float(np.mean(proc_array)),
            "std": float(np.std(proc_array))
        }
    }
    
    # Check against thresholds
    metrics["qc_status"] = evaluate_qc_thresholds(metrics["metrics"])
    
    # Save report if path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        console.print(f"\n[green]QC report saved to: {output_path}[/green]")
    
    # Print summary
    print_qc_summary(metrics)
    
    return metrics


def evaluate_qc_thresholds(metrics: Dict) -> Dict[str, str]:
    """
    Evaluate metrics against QC thresholds.
    
    Args:
        metrics: Dictionary of calculated metrics
        
    Returns:
        Dictionary of pass/fail status for each metric
    """
    thresholds = {
        "ssim": {"threshold": 0.95, "comparison": "greater"},
        "edge_preservation": {"threshold": 0.90, "comparison": "greater"},
        "nrmse": {"threshold": 0.10, "comparison": "less"},
        "air_hu_deviation": {"threshold": 5.0, "comparison": "less"},
        "fat_hu_deviation": {"threshold": 10.0, "comparison": "less"},
        "water_hu_deviation": {"threshold": 5.0, "comparison": "less"}
    }
    
    status = {}
    
    # Check SSIM
    if "ssim" in metrics:
        passed = metrics["ssim"] >= thresholds["ssim"]["threshold"]
        status["ssim"] = "PASS" if passed else "FAIL"
    
    # Check Edge Preservation
    if "edge_preservation" in metrics:
        passed = metrics["edge_preservation"] >= thresholds["edge_preservation"]["threshold"]
        status["edge_preservation"] = "PASS" if passed else "FAIL"
    
    # Check NRMSE
    if "nrmse" in metrics:
        passed = metrics["nrmse"] <= thresholds["nrmse"]["threshold"]
        status["nrmse"] = "PASS" if passed else "FAIL"
    
    # Check HU accuracy
    if "hu_accuracy" in metrics:
        hu_acc = metrics["hu_accuracy"]
        for tissue in ["air", "fat", "water"]:
            key = f"{tissue}_deviation"
            if key in hu_acc:
                threshold_key = f"{tissue}_hu_deviation"
                passed = hu_acc[key] <= thresholds[threshold_key]["threshold"]
                status[f"{tissue}_hu"] = "PASS" if passed else "FAIL"
    
    # Overall status
    all_passed = all(v == "PASS" for v in status.values())
    status["overall"] = "PASS" if all_passed else "FAIL"
    
    return status


def print_qc_summary(metrics: Dict) -> None:
    """Print formatted QC summary."""
    # Main metrics table
    table = Table(title="Quality Control Metrics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="yellow")
    table.add_column("Target", style="green")
    table.add_column("Status", style="bold")
    
    # Add rows
    m = metrics["metrics"]
    status = metrics["qc_status"]
    
    table.add_row(
        "SSIM",
        f"{m.get('ssim', 0):.3f}",
        "> 0.95",
        f"[{'green' if status.get('ssim') == 'PASS' else 'red'}]{status.get('ssim', 'N/A')}[/]"
    )
    
    table.add_row(
        "Edge Preservation",
        f"{m.get('edge_preservation', 0):.3f}",
        "≥ 0.90",
        f"[{'green' if status.get('edge_preservation') == 'PASS' else 'red'}]{status.get('edge_preservation', 'N/A')}[/]"
    )
    
    table.add_row(
        "NRMSE",
        f"{m.get('nrmse', 0):.3f}",
        "< 0.10",
        f"[{'green' if status.get('nrmse') == 'PASS' else 'red'}]{status.get('nrmse', 'N/A')}[/]"
    )
    
    console.print(table)
    
    # HU accuracy table if available
    if "hu_accuracy" in m and m["hu_accuracy"]:
        hu_table = Table(title="HU Accuracy Check")
        hu_table.add_column("Tissue", style="cyan")
        hu_table.add_column("Measured HU", style="yellow")
        hu_table.add_column("Expected HU", style="green")
        hu_table.add_column("Deviation", style="yellow")
        hu_table.add_column("Status", style="bold")
        
        hu = m["hu_accuracy"]
        expected = {"air": -1000, "fat": -100, "water": 0}
        
        for tissue in ["air", "fat", "water"]:
            if f"{tissue}_hu" in hu:
                hu_table.add_row(
                    tissue.capitalize(),
                    f"{hu[f'{tissue}_hu']:.1f}",
                    f"{expected[tissue]}",
                    f"{hu[f'{tissue}_deviation']:.1f}",
                    f"[{'green' if status.get(f'{tissue}_hu') == 'PASS' else 'red'}]{status.get(f'{tissue}_hu', 'N/A')}[/]"
                )
        
        console.print(hu_table)
    
    # Overall status
    overall_color = "green" if status.get("overall") == "PASS" else "red"
    console.print(
        Panel(
            f"[bold {overall_color}]Overall QC Status: {status.get('overall', 'N/A')}[/]",
            expand=False
        )
    )


if __name__ == "__main__":
    from dicom_loader import load_dicom_series
    from resampling import resample_to_isotropic
    from denoising import denoise_adaptive
    
    # Load test data
    dicom_dir = Path("1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192")
    
    if dicom_dir.exists():
        console.print("[green]Loading DICOM series...[/green]")
        original, metadata = load_dicom_series(dicom_dir)
        
        # Apply some processing
        console.print("\n[green]Applying test processing...[/green]")
        resampled, _ = resample_to_isotropic(original, target_spacing=0.5)
        denoised, _ = denoise_adaptive(resampled, method="bilateral")
        
        # Generate QC report
        processing_info = {
            "steps": ["resampling", "denoising"],
            "resampling": {"method": "B-spline", "target_spacing": 0.5},
            "denoising": {"method": "bilateral adaptive"}
        }
        
        report = generate_qc_report(
            original,
            denoised,
            processing_info,
            output_path=Path("qc_report.json")
        )
    else:
        console.print("[red]DICOM directory not found![/red]")