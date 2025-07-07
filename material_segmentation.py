"""Material segmentation and density calibration for polyenergetic CT conversion."""

from typing import Tuple, Dict, Optional, List
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from rich.console import Console
from rich.table import Table
from rich.progress import track

console = Console()

# Material IDs for GPU compatibility (uint8)
MATERIAL_AIR = 0
MATERIAL_LUNG = 1
MATERIAL_ADIPOSE = 2
MATERIAL_SOFT_TISSUE = 3
MATERIAL_BONE = 4

# Material names for display
MATERIAL_NAMES = {
    MATERIAL_AIR: "Air",
    MATERIAL_LUNG: "Lung",
    MATERIAL_ADIPOSE: "Adipose",
    MATERIAL_SOFT_TISSUE: "Soft Tissue",
    MATERIAL_BONE: "Bone"
}


def hu_to_material_and_density(hu_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert HU values to material IDs and mass densities using Schneider method.
    
    Based on the stoichiometric calibration method from:
    Schneider et al. (2000) "The calibration of CT Hounsfield units for 
    radiotherapy treatment planning"
    
    Args:
        hu_array: Input array of Hounsfield Units
        
    Returns:
        Tuple of (material_id_array, density_array)
        - material_id_array: uint8 array with material IDs (0-4)
        - density_array: float32 array with mass densities in g/cm³
    """
    # Initialize output arrays
    material_ids = np.zeros_like(hu_array, dtype=np.uint8)
    densities = np.zeros_like(hu_array, dtype=np.float32)  # Use float32 for SimpleITK compatibility
    
    # Define HU thresholds for material segmentation
    # Based on Schneider et al. stoichiometric calibration method
    HU_AIR_MAX = -950
    HU_LUNG_MAX = -120      # Updated from -150
    HU_ADIPOSE_MAX = -70    # Updated from -50
    HU_SOFT_TISSUE_MAX = 120  # Updated from 100
    
    # Air: HU < -950
    air_mask = hu_array < HU_AIR_MAX
    material_ids[air_mask] = MATERIAL_AIR
    # Air has constant density
    densities[air_mask] = 0.001205  # g/cm³ at STP
    
    # Lung: -950 ≤ HU < -120
    lung_mask = (hu_array >= HU_AIR_MAX) & (hu_array < HU_LUNG_MAX)
    material_ids[lung_mask] = MATERIAL_LUNG
    # Piecewise linear calibration for lung tissue with physical bounds
    # Base formula: ρ = 0.001205 + (HU + 1000) × 0.001174
    # But ensure minimum density of 0.05 g/cm³ for physical validity
    lung_hu = hu_array[lung_mask]
    lung_density = 0.001205 + (lung_hu + 1000) * 0.001174
    # Apply physical bounds: lung density typically 0.05-0.5 g/cm³
    lung_density = np.maximum(lung_density, 0.05)  # Minimum lung density
    lung_density = np.minimum(lung_density, 0.5)   # Maximum lung density
    densities[lung_mask] = lung_density
    
    # Adipose: -120 ≤ HU < -70
    adipose_mask = (hu_array >= HU_LUNG_MAX) & (hu_array < HU_ADIPOSE_MAX)
    material_ids[adipose_mask] = MATERIAL_ADIPOSE
    # Linear calibration for adipose tissue with bounds
    # ρ = 0.930 + (HU + 100) × 0.0013
    adipose_hu = hu_array[adipose_mask]
    adipose_density = 0.930 + (adipose_hu + 100) * 0.0013
    # Apply physical bounds: adipose density typically 0.85-0.97 g/cm³
    adipose_density = np.maximum(adipose_density, 0.85)
    adipose_density = np.minimum(adipose_density, 0.97)
    densities[adipose_mask] = adipose_density
    
    # Soft Tissue: -70 ≤ HU < 120
    soft_tissue_mask = (hu_array >= HU_ADIPOSE_MAX) & (hu_array < HU_SOFT_TISSUE_MAX)
    material_ids[soft_tissue_mask] = MATERIAL_SOFT_TISSUE
    # Linear calibration for soft tissue with bounds
    # ρ = 1.000 + HU × 0.0010
    soft_hu = hu_array[soft_tissue_mask]
    soft_density = 1.000 + soft_hu * 0.0010
    # Apply physical bounds: soft tissue density typically 0.95-1.10 g/cm³
    soft_density = np.maximum(soft_density, 0.95)
    soft_density = np.minimum(soft_density, 1.10)
    densities[soft_tissue_mask] = soft_density
    
    # Bone: HU ≥ 120
    bone_mask = hu_array >= HU_SOFT_TISSUE_MAX
    material_ids[bone_mask] = MATERIAL_BONE
    # Linear calibration for bone with bounds
    # ρ = 1.075 + (HU - 120) × 0.0005
    bone_hu = hu_array[bone_mask]
    bone_density = 1.075 + (bone_hu - 120) * 0.0005
    # Apply physical bounds: bone density typically 1.10-2.5 g/cm³
    bone_density = np.maximum(bone_density, 1.10)
    bone_density = np.minimum(bone_density, 2.5)
    densities[bone_mask] = bone_density
    
    # Final sanity check - no global clamping needed as each material has bounds
    
    return material_ids, densities


def segment_materials_with_cleanup(
    image: sitk.Image,
    apply_morphology: bool = True,
    min_region_size: int = 100
) -> Tuple[sitk.Image, sitk.Image, Dict]:
    """
    Segment CT image into materials with optional morphological cleanup.
    
    Args:
        image: Input SimpleITK image in Hounsfield Units
        apply_morphology: Whether to apply morphological operations
        min_region_size: Minimum voxels for a region to be kept
        
    Returns:
        Tuple of (material_id_image, density_image, info_dict)
    """
    console.print("[cyan]Performing material segmentation...[/cyan]")
    
    # Convert to numpy array
    hu_array = sitk.GetArrayFromImage(image)
    
    # Initial segmentation
    material_ids, densities = hu_to_material_and_density(hu_array)
    
    if apply_morphology:
        console.print("  Applying morphological cleanup...")
        
        # Clean up air regions - find external air
        air_mask = material_ids == MATERIAL_AIR
        
        # Use binary opening to remove small isolated air pockets
        air_mask_cleaned = ndimage.binary_opening(air_mask, iterations=2)
        
        # Find largest connected component (external air)
        labeled, num_features = ndimage.label(air_mask_cleaned)
        if num_features > 0:
            sizes = ndimage.sum(air_mask_cleaned, labeled, range(1, num_features + 1))
            if len(sizes) > 0:
                max_label = np.argmax(sizes) + 1
                external_air = labeled == max_label
                
                # Keep only external air and large internal air regions
                for i in range(1, num_features + 1):
                    if i != max_label and sizes[i - 1] < min_region_size:
                        # Convert small air pockets to lung tissue
                        small_air_region = labeled == i
                        material_ids[small_air_region] = MATERIAL_LUNG
                        # Recalculate density for these voxels with bounds
                        lung_hu = hu_array[small_air_region]
                        lung_density = 0.001205 + (lung_hu + 1000) * 0.001174
                        lung_density = np.maximum(lung_density, 0.05)
                        lung_density = np.minimum(lung_density, 0.5)
                        densities[small_air_region] = lung_density
        
        # Clean up other materials with median filter to reduce noise
        for material_id in [MATERIAL_LUNG, MATERIAL_ADIPOSE, MATERIAL_SOFT_TISSUE, MATERIAL_BONE]:
            material_mask = material_ids == material_id
            if np.any(material_mask):
                # Apply median filter to smooth boundaries
                material_mask_filtered = ndimage.median_filter(material_mask.astype(float), size=3) > 0.5
                # Find voxels that need to be reassigned
                reassign_mask = material_mask != material_mask_filtered
                if np.any(reassign_mask):
                    material_ids[reassign_mask] = MATERIAL_SOFT_TISSUE
                    # Recalculate density for reassigned voxels
                    reassign_hu = hu_array[reassign_mask]
                    soft_density = 1.000 + reassign_hu * 0.0010
                    soft_density = np.maximum(soft_density, 0.95)
                    soft_density = np.minimum(soft_density, 1.10)
                    densities[reassign_mask] = soft_density
    
    # Final safety check - ensure no negative or unrealistic densities
    # This catches any edge cases from morphological operations
    densities = np.maximum(densities, 0.0001)  # Absolute minimum density
    
    # Calculate statistics
    info = {
        "total_voxels": material_ids.size,
        "material_counts": {},
        "material_percentages": {},
        "density_stats": {}
    }
    
    for material_id, material_name in MATERIAL_NAMES.items():
        mask = material_ids == material_id
        count = np.sum(mask)
        percentage = 100.0 * count / material_ids.size
        
        info["material_counts"][material_name] = int(count)
        info["material_percentages"][material_name] = float(percentage)
        
        if count > 0:
            material_densities = densities[mask]
            info["density_stats"][material_name] = {
                "min": float(np.min(material_densities)),
                "max": float(np.max(material_densities)),
                "mean": float(np.mean(material_densities)),
                "std": float(np.std(material_densities))
            }
    
    # Create SimpleITK images
    material_id_image = sitk.GetImageFromArray(material_ids)
    material_id_image.CopyInformation(image)
    
    density_image = sitk.GetImageFromArray(densities)
    density_image.CopyInformation(image)
    
    return material_id_image, density_image, info


def validate_segmentation(info: Dict, min_tissue_percentage: float = 0.1) -> Tuple[bool, List[str]]:
    """
    Validate that segmentation produced reasonable results.
    
    Args:
        info: Segmentation info dictionary
        min_tissue_percentage: Minimum percentage for key tissues
        
    Returns:
        Tuple of (is_valid, warnings)
    """
    warnings = []
    
    # Check for minimum tissue presence
    percentages = info["material_percentages"]
    
    if percentages.get("Soft Tissue", 0) < min_tissue_percentage:
        warnings.append(f"Very little soft tissue detected ({percentages.get('Soft Tissue', 0):.1f}%)")
    
    if percentages.get("Air", 0) < 10.0:
        warnings.append(f"Very little air detected ({percentages.get('Air', 0):.1f}%)")
    
    # Check density ranges
    density_stats = info["density_stats"]
    
    for material_name, expected_range in [
        ("Air", (0.0001, 0.01)),
        ("Lung", (0.1, 0.5)),
        ("Adipose", (0.85, 0.97)),
        ("Soft Tissue", (0.95, 1.10)),
        ("Bone", (1.10, 2.5))
    ]:
        if material_name in density_stats:
            stats = density_stats[material_name]
            if stats["min"] < expected_range[0] or stats["max"] > expected_range[1]:
                warnings.append(
                    f"{material_name} density out of expected range: "
                    f"{stats['min']:.3f}-{stats['max']:.3f} g/cm³ "
                    f"(expected {expected_range[0]:.3f}-{expected_range[1]:.3f})"
                )
    
    is_valid = len(warnings) == 0
    return is_valid, warnings


def print_segmentation_report(info: Dict) -> None:
    """Print formatted segmentation report."""
    console.print("\n[bold cyan]Material Segmentation Report[/bold cyan]")
    
    # Material distribution table
    dist_table = Table(title="Material Distribution")
    dist_table.add_column("Material", style="cyan")
    dist_table.add_column("Voxel Count", style="yellow")
    dist_table.add_column("Percentage", style="green")
    
    for material_name in MATERIAL_NAMES.values():
        count = info["material_counts"].get(material_name, 0)
        percentage = info["material_percentages"].get(material_name, 0.0)
        dist_table.add_row(
            material_name,
            f"{count:,}",
            f"{percentage:.1f}%"
        )
    
    console.print(dist_table)
    
    # Density statistics table
    if info["density_stats"]:
        density_table = Table(title="Density Statistics (g/cm³)")
        density_table.add_column("Material", style="cyan")
        density_table.add_column("Min", style="red")
        density_table.add_column("Mean", style="yellow")
        density_table.add_column("Max", style="green")
        density_table.add_column("Std", style="blue")
        
        for material_name, stats in info["density_stats"].items():
            density_table.add_row(
                material_name,
                f"{stats['min']:.3f}",
                f"{stats['mean']:.3f}",
                f"{stats['max']:.3f}",
                f"{stats['std']:.3f}"
            )
        
        console.print(density_table)


if __name__ == "__main__":
    from pathlib import Path
    
    # Test with sample data
    test_file = Path("test_output/preprocessed_hu.nii.gz")
    
    if test_file.exists():
        console.print(f"[green]Loading test file: {test_file}[/green]")
        
        # Load image
        image = sitk.ReadImage(str(test_file))
        
        # Perform segmentation
        material_image, density_image, info = segment_materials_with_cleanup(image)
        
        # Print report
        print_segmentation_report(info)
        
        # Validate
        is_valid, warnings = validate_segmentation(info)
        if is_valid:
            console.print("\n[green]✓ Segmentation validation passed[/green]")
        else:
            console.print("\n[yellow]⚠ Segmentation warnings:[/yellow]")
            for warning in warnings:
                console.print(f"  - {warning}")
        
        # Save outputs
        output_dir = Path("test_output_polyenergetic")
        output_dir.mkdir(exist_ok=True)
        
        sitk.WriteImage(material_image, str(output_dir / "material_ids.nii.gz"))
        sitk.WriteImage(density_image, str(output_dir / "densities.nii.gz"))
        
        console.print(f"\n[green]Saved outputs to {output_dir}/[/green]")
    else:
        console.print(f"[red]Test file not found: {test_file}[/red]")
        console.print("Run preprocessing first to generate test data.")