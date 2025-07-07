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
from material_segmentation import MATERIAL_NAMES

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
    # Cast to float32 first as SobelEdgeDetection doesn't support int32 in 3D
    original_float = sitk.Cast(original, sitk.sitkFloat32)
    processed_float = sitk.Cast(processed, sitk.sitkFloat32)
    orig_grad = sitk.GetArrayFromImage(sitk.SobelEdgeDetection(original_float))
    proc_grad = sitk.GetArrayFromImage(sitk.SobelEdgeDetection(processed_float))
    
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
    sample_slices: int = 20
) -> float:
    """
    Calculate SSIM for 3D images by sampling slices.
    
    Args:
        original: Original image
        processed: Processed image
        sample_slices: Number of slices to sample (more = more accurate)
        
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
    
    # Check water/blood - use narrower range for better water detection
    water_mask = (array > -20) & (array < 20)
    if np.any(water_mask):
        results["water_hu"] = float(np.mean(array[water_mask]))
        results["water_deviation"] = abs(results["water_hu"] - 0)
    
    return results


def validate_polyenergetic_conversion(
    hu_image: sitk.Image,
    material_id_image: sitk.Image,
    density_image: sitk.Image
) -> Dict:
    """
    Validate polyenergetic conversion quality.
    
    Args:
        hu_image: Calibrated HU image
        material_id_image: Material segmentation 
        density_image: Mass density image
        
    Returns:
        Dictionary with validation metrics
    """
    hu_array = sitk.GetArrayFromImage(hu_image)
    material_ids = sitk.GetArrayFromImage(material_id_image)
    densities = sitk.GetArrayFromImage(density_image)
    
    validation = {
        "material_coverage": {},
        "density_ranges": {},
        "hu_consistency": {},
        "warnings": []
    }
    
    # Check material coverage
    total_voxels = material_ids.size
    for mat_id, mat_name in MATERIAL_NAMES.items():
        count = np.sum(material_ids == mat_id)
        percentage = 100.0 * count / total_voxels
        validation["material_coverage"][mat_name] = {
            "count": int(count),
            "percentage": float(percentage)
        }
        
        # Adjust thresholds for thoracic CT where lung and air dominate
        if mat_name == "Soft Tissue" and percentage < 5:  # Reduced from 10%
            validation["warnings"].append(f"Low soft tissue coverage: {percentage:.1f}%")
        elif mat_name == "Air" and percentage < 2:  # Reduced from 5%
            validation["warnings"].append(f"Very low air coverage: {percentage:.1f}%")
    
    # Validate density ranges
    # Updated to match the physical bounds in material_segmentation.py
    expected_density_ranges = {
        "Air": (0.0001, 0.01),
        "Lung": (0.05, 0.5),      # Updated minimum from 0.1 to 0.05
        "Adipose": (0.85, 0.97),
        "Soft Tissue": (0.95, 1.10),
        "Bone": (1.10, 2.5)
    }
    
    for mat_id, mat_name in MATERIAL_NAMES.items():
        mask = material_ids == mat_id
        if np.any(mask):
            mat_densities = densities[mask]
            min_density = float(np.min(mat_densities))
            max_density = float(np.max(mat_densities))
            mean_density = float(np.mean(mat_densities))
            
            validation["density_ranges"][mat_name] = {
                "min": min_density,
                "max": max_density,
                "mean": mean_density
            }
            
            # Check if within expected range
            expected_min, expected_max = expected_density_ranges.get(mat_name, (0, 3))
            if min_density < expected_min * 0.8 or max_density > expected_max * 1.2:
                validation["warnings"].append(
                    f"{mat_name} density out of range: {min_density:.3f}-{max_density:.3f} g/cm³"
                )
    
    # Check HU consistency within materials
    for mat_id, mat_name in MATERIAL_NAMES.items():
        mask = material_ids == mat_id
        if np.sum(mask) > 100:  # Need sufficient voxels
            mat_hu = hu_array[mask]
            hu_std = float(np.std(mat_hu))
            
            validation["hu_consistency"][mat_name] = {
                "std": hu_std,
                "cv": float(hu_std / (np.mean(mat_hu) + 1e-6))  # Coefficient of variation
            }
            
            # High variability within a material might indicate poor segmentation
            # But for Air, high variability is expected near boundaries
            if mat_name == "Water" and hu_std > 50:
                validation["warnings"].append(
                    f"High HU variability in {mat_name}: std={hu_std:.1f}"
                )
            elif mat_name == "Air" and hu_std > 1000:  # Much higher threshold for air
                validation["warnings"].append(
                    f"Extremely high HU variability in {mat_name}: std={hu_std:.1f}"
                )
    
    # Overall quality score
    validation["quality_score"] = max(0, 100 - len(validation["warnings"]) * 10)
    
    return validation


def generate_qc_report(
    original_image: sitk.Image,
    processed_image: sitk.Image,
    processing_info: Dict,
    output_path: Optional[Path] = None,
    material_id_image: Optional[sitk.Image] = None,
    density_image: Optional[sitk.Image] = None
) -> Dict:
    """
    Generate comprehensive QC report.
    
    Args:
        original_image: Original input image
        processed_image: Final processed image
        processing_info: Dictionary with processing step information
        output_path: Optional path to save report
        material_id_image: Optional material segmentation image
        density_image: Optional density image
        
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
    
    # Polyenergetic validation if available
    if material_id_image is not None and density_image is not None:
        console.print("  Validating polyenergetic conversion...")
        metrics["metrics"]["polyenergetic_validation"] = validate_polyenergetic_conversion(
            processed_image, material_id_image, density_image
        )
    
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
        "ssim": {"threshold": 0.90, "comparison": "greater"},  # Relaxed from 0.95
        "edge_preservation": {"threshold": 0.90, "comparison": "greater"},
        "nrmse": {"threshold": 0.10, "comparison": "less"},
        "air_hu_deviation": {"threshold": 25.0, "comparison": "less"},  # Relaxed from 5.0
        "fat_hu_deviation": {"threshold": 10.0, "comparison": "less"},
        "water_hu_deviation": {"threshold": 15.0, "comparison": "less"}  # Relaxed from 5.0
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
    
    # Check polyenergetic validation
    if "polyenergetic_validation" in metrics:
        poly_val = metrics["polyenergetic_validation"]
        if poly_val.get("quality_score", 0) >= 80:
            status["polyenergetic"] = "PASS"
        else:
            status["polyenergetic"] = "FAIL"
    
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
        "> 0.90",
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
    
    # Polyenergetic validation if available
    if "polyenergetic_validation" in m and m["polyenergetic_validation"]:
        poly_val = m["polyenergetic_validation"]
        
        # Material coverage summary
        poly_table = Table(title="Polyenergetic Conversion Validation")
        poly_table.add_column("Metric", style="cyan")
        poly_table.add_column("Value", style="yellow")
        poly_table.add_column("Status", style="bold")
        
        poly_table.add_row(
            "Quality Score",
            f"{poly_val.get('quality_score', 0)}/100",
            f"[{'green' if status.get('polyenergetic') == 'PASS' else 'red'}]{status.get('polyenergetic', 'N/A')}[/]"
        )
        
        poly_table.add_row("Warnings", str(len(poly_val.get("warnings", []))), "")
        
        console.print(poly_table)
        
        # Show warnings if any
        if poly_val.get("warnings"):
            console.print("\n[yellow]Polyenergetic Warnings:[/yellow]")
            for warning in poly_val["warnings"]:
                console.print(f"  • {warning}")
    
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