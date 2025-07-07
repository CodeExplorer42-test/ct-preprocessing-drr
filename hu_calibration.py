"""HU calibration and conversion to attenuation coefficients module."""

from typing import Tuple, Dict, Optional, List
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from rich.console import Console
from rich.table import Table
from pathlib import Path

# Import polyenergetic modules
from material_segmentation import segment_materials_with_cleanup, print_segmentation_report
from polyenergetic_conversion import (
    polyenergetic_hu_to_mu, 
    export_polyenergetic_data,
    print_polyenergetic_report
)

console = Console()


def segment_tissue_roi(image: sitk.Image, tissue_type: str) -> Tuple[np.ndarray, Dict]:
    """
    Segment ROI for specific tissue type for calibration.
    
    Args:
        image: Input SimpleITK image
        tissue_type: "air", "fat", "water", or "bone"
        
    Returns:
        Tuple of (ROI mask array, info dict)
    """
    array = sitk.GetArrayFromImage(image)
    
    # Define HU ranges for different tissues
    tissue_ranges = {
        "air": (-1100, -900),
        "fat": (-150, -50),
        "water": (-50, 50),
        "bone": (300, 1500)
    }
    
    if tissue_type not in tissue_ranges:
        raise ValueError(f"Unknown tissue type: {tissue_type}")
    
    hu_min, hu_max = tissue_ranges[tissue_type]
    
    # Create initial mask
    mask = (array >= hu_min) & (array <= hu_max)
    
    # Clean up mask with morphological operations
    if tissue_type == "air":
        # For air, look for large connected regions (trachea, outside body)
        # Remove small isolated regions
        mask = ndimage.binary_opening(mask, iterations=2)
        
        # Find largest connected component (likely outside body)
        labeled, num_features = ndimage.label(mask)
        if num_features > 0:
            sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
            max_label = np.argmax(sizes) + 1
            mask = labeled == max_label
    
    elif tissue_type == "fat":
        # For fat, look for subcutaneous regions
        # Apply morphological operations to get connected regions
        mask = ndimage.binary_closing(mask, iterations=1)
        mask = ndimage.binary_opening(mask, iterations=1)
    
    # Calculate statistics
    roi_values = array[mask]
    info = {
        "tissue_type": tissue_type,
        "hu_range": tissue_ranges[tissue_type],
        "num_voxels": int(np.sum(mask)),
        "mean_hu": float(np.mean(roi_values)) if len(roi_values) > 0 else None,
        "std_hu": float(np.std(roi_values)) if len(roi_values) > 0 else None,
        "median_hu": float(np.median(roi_values)) if len(roi_values) > 0 else None
    }
    
    return mask, info


def phantom_free_calibration(image: sitk.Image) -> Tuple[sitk.Image, Dict]:
    """
    Perform phantom-free HU calibration using internal tissue references.
    
    Args:
        image: Input SimpleITK image
        
    Returns:
        Tuple of (calibrated image, calibration info)
    """
    # Define expected HU values for reference tissues
    expected_hu = {
        "air": -1000,
        "fat": -100,
        "water": 0
    }
    
    # Segment reference tissues
    console.print("[cyan]Segmenting reference tissues...[/cyan]")
    measurements = {}
    
    for tissue in ["air", "fat"]:  # Using air and fat as primary references
        mask, info = segment_tissue_roi(image, tissue)
        
        if info["mean_hu"] is not None and info["num_voxels"] > 1000:  # Require sufficient voxels
            measurements[tissue] = info["median_hu"]  # Use median for robustness
            console.print(f"  {tissue.capitalize()}: {info['median_hu']:.1f} HU (expected: {expected_hu[tissue]} HU)")
        else:
            console.print(f"  [yellow]Warning: Could not reliably segment {tissue}[/yellow]")
    
    # Perform linear calibration if we have at least 2 reference points
    if len(measurements) >= 2:
        # Set up linear regression
        measured_values = []
        expected_values = []
        
        for tissue, measured_hu in measurements.items():
            measured_values.append(measured_hu)
            expected_values.append(expected_hu[tissue])
        
        # Calculate linear transformation parameters
        # HU_corrected = a * HU_measured + b
        measured_values = np.array(measured_values)
        expected_values = np.array(expected_values)
        
        # Solve for a and b using least squares
        A = np.vstack([measured_values, np.ones(len(measured_values))]).T
        a, b = np.linalg.lstsq(A, expected_values, rcond=None)[0]
        
        console.print(f"\n[green]Calibration equation: HU_corrected = {a:.4f} * HU_measured + {b:.2f}[/green]")
        
        # Apply calibration
        array = sitk.GetArrayFromImage(image)
        calibrated_array = a * array + b
        
        # Create calibrated image
        calibrated_image = sitk.GetImageFromArray(calibrated_array)
        calibrated_image.CopyInformation(image)
        
        info = {
            "method": "phantom-free linear calibration",
            "scale_factor": float(a),
            "offset": float(b),
            "reference_tissues": list(measurements.keys()),
            "measured_values": measurements,
            "calibration_error": float(np.mean(np.abs(a * measured_values + b - expected_values)))
        }
        
    else:
        console.print("[yellow]Warning: Insufficient reference tissues for calibration[/yellow]")
        calibrated_image = image
        info = {
            "method": "none",
            "reason": "insufficient reference tissues"
        }
    
    return calibrated_image, info


def hu_to_attenuation_coefficient(
    image: sitk.Image,
    energy_kev: Optional[float] = None,
    use_polyenergetic: bool = True,
    apply_morphology: bool = True
) -> Tuple[sitk.Image, Dict, Optional[sitk.Image], Optional[sitk.Image]]:
    """
    Convert calibrated HU values to linear attenuation coefficients.
    
    Args:
        image: Input SimpleITK image (in HU)
        energy_kev: If specified, return μ at single energy; 
                   otherwise use full polyenergetic model
        use_polyenergetic: Whether to use polyenergetic model (recommended)
        apply_morphology: Apply morphological cleanup to segmentation
        
    Returns:
        Tuple of (mu_image, conversion_info, material_id_image, density_image)
        For backward compatibility, last two are None if not using polyenergetic
    """
    if not use_polyenergetic:
        # Legacy monoenergetic model (NOT RECOMMENDED)
        console.print("[yellow]Warning: Using monoenergetic model - expect errors up to 155 HU for bone![/yellow]")
        
        # Standard water μ at 70 keV
        water_mu = 0.1928
        if energy_kev is None:
            energy_kev = 70.0
        
        # Simple HU to μ conversion
        array = sitk.GetArrayFromImage(image)
        mu_array = water_mu * (array / 1000.0 + 1.0)
        mu_array[mu_array < 0] = 0.0001
        
        mu_image = sitk.GetImageFromArray(mu_array.astype(np.float32))
        mu_image.CopyInformation(image)
        
        info = {
            "method": "monoenergetic",
            "energy_kev": energy_kev,
            "water_mu": water_mu,
            "warning": "Monoenergetic model - expect large errors for bone"
        }
        
        return mu_image, info, None, None
    
    # Polyenergetic model (RECOMMENDED)
    console.print("[green]Using physics-accurate polyenergetic model[/green]")
    
    # Step 1: Material segmentation
    material_id_image, density_image, seg_info = segment_materials_with_cleanup(
        image, apply_morphology=apply_morphology
    )
    
    # Step 2: Polyenergetic conversion
    mu_image, conv_info = polyenergetic_hu_to_mu(
        image, material_id_image, density_image, energy_kev
    )
    
    # Combine info
    info = {
        "method": "polyenergetic",
        "segmentation": seg_info,
        "conversion": conv_info
    }
    
    return mu_image, info, material_id_image, density_image


def print_calibration_report(calibration_info: Dict, conversion_info: Dict) -> None:
    """Print formatted calibration and conversion report."""
    console.print("\n[bold cyan]HU Calibration Report[/bold cyan]")
    
    if calibration_info.get("method") != "none":
        table = Table(title="Calibration Results")
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Method", calibration_info.get("method", "unknown"))
        if "scale_factor" in calibration_info:
            table.add_row("Scale Factor", f"{calibration_info['scale_factor']:.4f}")
            table.add_row("Offset", f"{calibration_info['offset']:.2f} HU")
            table.add_row("Calibration Error", f"{calibration_info['calibration_error']:.2f} HU")
        
        console.print(table)
        
        # Reference tissue measurements
        if "measured_values" in calibration_info:
            ref_table = Table(title="Reference Tissue Measurements")
            ref_table.add_column("Tissue", style="cyan")
            ref_table.add_column("Measured HU", style="yellow")
            ref_table.add_column("Expected HU", style="green")
            
            expected = {"air": -1000, "fat": -100}
            for tissue, measured in calibration_info["measured_values"].items():
                ref_table.add_row(tissue.capitalize(), f"{measured:.1f}", f"{expected[tissue]}")
            
            console.print(ref_table)
    
    # Handle both old monoenergetic and new polyenergetic formats
    if isinstance(conversion_info, dict):
        if "method" in conversion_info and conversion_info["method"] == "polyenergetic":
            # New polyenergetic format
            if "segmentation" in conversion_info:
                print_segmentation_report(conversion_info["segmentation"])
            if "conversion" in conversion_info:
                print_polyenergetic_report(conversion_info["conversion"])
        else:
            # Old monoenergetic format
            console.print("\n[bold cyan]Attenuation Coefficient Conversion[/bold cyan]")
            
            mu_table = Table()
            mu_table.add_column("Tissue", style="cyan")
            mu_table.add_column("μ (cm⁻¹)", style="green")
            
            if conversion_info.get("mu_air") is not None:
                mu_table.add_row("Air", f"{conversion_info['mu_air']:.4f}")
            if "mu_water" in conversion_info:
                mu_table.add_row("Water", f"{conversion_info['mu_water']:.4f}")
            if conversion_info.get("mu_bone") is not None:
                mu_table.add_row("Bone", f"{conversion_info['mu_bone']:.4f}")
            
            console.print(mu_table)
            if "energy_kev" in conversion_info:
                console.print(f"\nEnergy: [yellow]{conversion_info['energy_kev']} keV[/yellow]")
            
            if "warning" in conversion_info:
                console.print(f"\n[red]Warning: {conversion_info['warning']}[/red]")


if __name__ == "__main__":
    from pathlib import Path
    from dicom_loader import load_dicom_series
    
    # Load test data
    dicom_dir = Path("1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192")
    
    if dicom_dir.exists():
        console.print("[green]Loading DICOM series...[/green]")
        image, metadata = load_dicom_series(dicom_dir)
        
        # Perform phantom-free calibration
        console.print("\n[bold]Performing Phantom-Free HU Calibration[/bold]")
        calibrated, cal_info = phantom_free_calibration(image)
        
        # Test both mono and polyenergetic models for comparison
        console.print("\n[bold]Testing Monoenergetic Model (Legacy)[/bold]")
        mu_mono, info_mono, _, _ = hu_to_attenuation_coefficient(
            calibrated, energy_kev=70.0, use_polyenergetic=False
        )
        
        console.print("\n[bold]Testing Polyenergetic Model (Recommended)[/bold]")
        mu_poly, info_poly, mat_ids, densities = hu_to_attenuation_coefficient(
            calibrated, use_polyenergetic=True
        )
        
        # Print reports
        console.print("\n[bold]===== MONOENERGETIC RESULTS =====[/bold]")
        print_calibration_report(cal_info, info_mono)
        
        console.print("\n[bold]===== POLYENERGETIC RESULTS =====[/bold]")
        print_calibration_report(cal_info, info_poly)
        
        # Save outputs
        output_dir = Path("test_output_polyenergetic")
        output_dir.mkdir(exist_ok=True)
        
        # Save polyenergetic data for GPU processing
        if mat_ids is not None and densities is not None:
            export_info = export_polyenergetic_data(
                output_dir, mat_ids, densities
            )
            console.print(f"\n[green]Exported GPU-ready data to {output_dir}/[/green]")
            console.print(f"  - Material IDs: material_ids.nii.gz")
            console.print(f"  - Densities: densities.nii.gz")
            console.print(f"  - Lookup tables: polyenergetic_data.npz")
            console.print(f"  - Metadata: polyenergetic_metadata.json")
        
        console.print("\n[green]Calibration and conversion complete![/green]")
    else:
        console.print("[red]DICOM directory not found![/red]")