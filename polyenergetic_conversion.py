"""Polyenergetic X-ray spectrum and attenuation coefficient conversion."""

from typing import Dict, Tuple, Optional, List
import numpy as np
import SimpleITK as sitk
from scipy import interpolate
from rich.console import Console
from rich.table import Table
from rich.progress import track
import json
from pathlib import Path

from material_segmentation import (
    MATERIAL_AIR, MATERIAL_LUNG, MATERIAL_ADIPOSE, 
    MATERIAL_SOFT_TISSUE, MATERIAL_BONE, MATERIAL_NAMES
)

console = Console()


def get_120kvp_spectrum() -> Tuple[np.ndarray, np.ndarray]:
    """
    Get 120 kVp tungsten anode X-ray spectrum.
    
    Based on SpekPy v2.0 calculations with:
    - Tungsten anode at 12 degrees
    - Filtration: Inherent + 1.0 mm Al + 0.2 mm Cu
    - Energy range: 15-120 keV in 1 keV bins
    
    Returns:
        Tuple of (energies, relative_fluences)
    """
    # Energy values in keV
    energies = np.arange(15, 121, 1)
    
    # Relative fluence values from research document
    # Normalized to peak fluence of 1.0
    fluences = np.array([
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000,  # 15-19 keV
        0.0000, 0.0000, 0.0000, 0.0000, 0.0000,  # 20-24 keV
        0.0000, 0.0002, 0.0019, 0.0076, 0.0195,  # 25-29 keV
        0.0388, 0.0660, 0.1006, 0.1416, 0.1878,  # 30-34 keV
        0.2378, 0.2899, 0.3429, 0.3956, 0.4468,  # 35-39 keV
        0.4956, 0.5409, 0.5821, 0.6186, 0.6500,  # 40-44 keV
        0.6761, 0.6968, 0.7123, 0.7226, 0.7279,  # 45-49 keV
        0.7282, 0.7237, 0.7225, 0.7410, 0.7583,  # 50-54 keV
        0.7744, 0.7892, 0.8028, 0.8152, 0.9408,  # 55-59 keV (59: K-edge peak)
        0.8364, 0.8456, 0.8539, 0.8613, 0.8679,  # 60-64 keV
        0.8737, 0.8787, 0.9634, 0.8858, 0.9022,  # 65-69 keV (67: K-edge peak)
        0.8872, 0.8784, 0.8689, 0.8586, 0.8476,  # 70-74 keV
        0.8359, 0.8236, 0.8105, 0.7968, 0.7825,  # 75-79 keV
        0.7675, 0.7519, 0.7358, 0.7191, 0.7019,  # 80-84 keV
        0.6841, 0.6658, 0.6470, 0.5615, 0.5369,  # 85-89 keV (88: reduced)
        0.5230, 0.5091, 0.4952, 0.4813, 0.4674,  # 90-94 keV
        0.4535, 0.4396, 0.4257, 0.4118, 0.3979,  # 95-99 keV
        0.3840, 0.3701, 0.3562, 0.3423, 0.3284,  # 100-104 keV
        0.3145, 0.3006, 0.2867, 0.2728, 0.2589,  # 105-109 keV
        0.2450, 0.2311, 0.2172, 0.2033, 0.1894,  # 110-114 keV
        0.1755, 0.1616, 0.1477, 0.1338, 0.1199,  # 115-119 keV
        0.1060                                     # 120 keV
    ])
    
    return energies, fluences


def get_nist_mass_attenuation_coefficients() -> Dict[int, np.ndarray]:
    """
    Get NIST mass attenuation coefficients for human tissues.
    
    Data from NIST XCOM database for ICRU-44 tissue definitions.
    
    Returns:
        Dictionary mapping material_id to mass attenuation coefficients (cm²/g)
    """
    # Energy points from NIST tables (keV)
    nist_energies = np.array([20, 30, 40, 50, 60, 80, 100, 120, 150])
    
    # Mass attenuation coefficients (μ/ρ) in cm²/g
    nist_data = {
        MATERIAL_AIR: np.array([0.810, 0.376, 0.268, 0.227, 0.206, 0.184, 0.171, 0.162, 0.151]),
        MATERIAL_LUNG: np.array([0.793, 0.366, 0.261, 0.221, 0.200, 0.178, 0.165, 0.157, 0.145]),
        MATERIAL_ADIPOSE: np.array([0.612, 0.297, 0.220, 0.191, 0.177, 0.161, 0.153, 0.147, 0.138]),
        MATERIAL_SOFT_TISSUE: np.array([0.821, 0.378, 0.269, 0.226, 0.205, 0.182, 0.169, 0.161, 0.149]),
        MATERIAL_BONE: np.array([4.001, 1.331, 0.666, 0.424, 0.315, 0.223, 0.186, 0.165, 0.148])
    }
    
    # Interpolate to 1 keV resolution using log-log cubic splines
    energies = np.arange(15, 121, 1)
    mu_over_rho_tables = {}
    
    for material_id, mu_values in nist_data.items():
        # Use log-log interpolation for smooth curves
        log_interp = interpolate.interp1d(
            np.log(nist_energies), 
            np.log(mu_values),
            kind='cubic',
            fill_value='extrapolate'
        )
        
        # Evaluate at all energies
        log_mu_interp = log_interp(np.log(energies))
        mu_over_rho_tables[material_id] = np.exp(log_mu_interp)
    
    return mu_over_rho_tables


def calculate_effective_spectrum(
    base_spectrum: Tuple[np.ndarray, np.ndarray],
    detector_response: Optional[Dict] = None
) -> np.ndarray:
    """
    Calculate effective spectrum including detector response.
    
    Args:
        base_spectrum: Tuple of (energies, fluences)
        detector_response: Optional detector transmission factors
        
    Returns:
        Effective spectrum array
    """
    energies, fluences = base_spectrum
    effective_spectrum = fluences.copy()
    
    if detector_response:
        # Apply detector-specific transmission factors
        # This would include front detector absorption, mask transmission, etc.
        # For now, we'll use the base spectrum
        pass
    
    # Normalize to sum to 1.0 for numerical stability
    effective_spectrum = effective_spectrum / np.sum(effective_spectrum)
    
    return effective_spectrum


def polyenergetic_hu_to_mu(
    hu_image: sitk.Image,
    material_id_image: sitk.Image,
    density_image: sitk.Image,
    energy_kev: Optional[float] = None
) -> Tuple[sitk.Image, Dict]:
    """
    Convert HU to linear attenuation coefficients using polyenergetic model.
    
    Args:
        hu_image: Original HU image (for reference)
        material_id_image: Material ID assignments (uint8)
        density_image: Mass density values (float16)
        energy_kev: If specified, return μ at single energy; 
                   otherwise return effective μ for full spectrum
        
    Returns:
        Tuple of (mu_image, info_dict)
    """
    console.print("[cyan]Performing polyenergetic HU to μ conversion...[/cyan]")
    
    # Get spectrum and attenuation data
    energies, spectrum = get_120kvp_spectrum()
    mu_over_rho_tables = get_nist_mass_attenuation_coefficients()
    effective_spectrum = calculate_effective_spectrum((energies, spectrum))
    
    # Convert to numpy arrays
    material_ids = sitk.GetArrayFromImage(material_id_image)
    densities = sitk.GetArrayFromImage(density_image)
    
    if energy_kev is not None:
        # Single energy conversion
        console.print(f"  Converting at {energy_kev} keV...")
        
        # Find closest energy index
        energy_idx = np.argmin(np.abs(energies - energy_kev))
        actual_energy = energies[energy_idx]
        
        # Initialize output
        mu_array = np.zeros_like(densities, dtype=np.float32)
        
        # Convert each material
        for material_id in range(5):
            mask = material_ids == material_id
            if np.any(mask):
                mu_over_rho = mu_over_rho_tables[material_id][energy_idx]
                mu_array[mask] = densities[mask] * mu_over_rho
        
        info = {
            "mode": "monoenergetic",
            "energy_kev": float(actual_energy),
            "mean_energy_kev": float(actual_energy)
        }
        
    else:
        # Full polyenergetic conversion - calculate effective μ
        console.print("  Calculating effective μ for full spectrum...")
        
        # For display/registration, we need a single effective μ value
        # This represents the spectrum-weighted average attenuation
        mu_array = np.zeros_like(densities, dtype=np.float32)
        
        # Calculate effective μ for each material
        for material_id in range(5):
            mask = material_ids == material_id
            if np.any(mask):
                # Spectrum-weighted average μ/ρ
                mu_over_rho_effective = np.sum(
                    effective_spectrum * mu_over_rho_tables[material_id]
                )
                mu_array[mask] = densities[mask] * mu_over_rho_effective
        
        # Calculate mean energy
        mean_energy = np.sum(energies * effective_spectrum)
        
        info = {
            "mode": "polyenergetic",
            "energy_range_kev": (float(energies[0]), float(energies[-1])),
            "mean_energy_kev": float(mean_energy),
            "num_energy_bins": len(energies)
        }
    
    # Add material-specific statistics
    info["material_mu_stats"] = {}
    for material_id, material_name in MATERIAL_NAMES.items():
        mask = material_ids == material_id
        if np.any(mask):
            material_mu = mu_array[mask]
            info["material_mu_stats"][material_name] = {
                "min": float(np.min(material_mu)),
                "max": float(np.max(material_mu)),
                "mean": float(np.mean(material_mu)),
                "std": float(np.std(material_mu))
            }
    
    # Create output image
    mu_image = sitk.GetImageFromArray(mu_array)
    mu_image.CopyInformation(hu_image)
    
    return mu_image, info


def export_polyenergetic_data(
    output_dir: Path,
    material_id_image: sitk.Image,
    density_image: sitk.Image,
    mu_tables: Optional[Dict] = None,
    spectrum: Optional[Tuple] = None
) -> Dict:
    """
    Export data in GPU-ready format for ray projection.
    
    Args:
        output_dir: Output directory path
        material_id_image: Material IDs (uint8)
        density_image: Mass densities (float32 in SimpleITK, converted to float16 for GPU)
        mu_tables: Optional precomputed μ/ρ tables
        spectrum: Optional spectrum data
        
    Returns:
        Dictionary with export metadata
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save volume data (keep as float32 for SimpleITK compatibility)
    sitk.WriteImage(material_id_image, str(output_dir / "material_ids.nii.gz"))
    sitk.WriteImage(density_image, str(output_dir / "densities.nii.gz"))
    
    # Save lookup tables and spectrum
    if mu_tables is None:
        mu_tables = get_nist_mass_attenuation_coefficients()
    
    if spectrum is None:
        spectrum = get_120kvp_spectrum()
    
    # Convert to arrays for saving
    energies, fluences = spectrum
    
    # Convert density to float16 for GPU memory efficiency
    density_array = sitk.GetArrayFromImage(density_image)
    density_float16 = density_array.astype(np.float16)
    
    # Save as numpy arrays for easy loading
    np.savez_compressed(
        output_dir / "polyenergetic_data.npz",
        energies=energies,
        spectrum=fluences,
        mu_over_rho_air=mu_tables[MATERIAL_AIR],
        mu_over_rho_lung=mu_tables[MATERIAL_LUNG],
        mu_over_rho_adipose=mu_tables[MATERIAL_ADIPOSE],
        mu_over_rho_soft_tissue=mu_tables[MATERIAL_SOFT_TISSUE],
        mu_over_rho_bone=mu_tables[MATERIAL_BONE],
        density_float16=density_float16  # GPU-optimized density array
    )
    
    # Save metadata
    metadata = {
        "energy_range_kev": [int(energies[0]), int(energies[-1])],
        "num_energy_bins": len(energies),
        "materials": {
            "air": MATERIAL_AIR,
            "lung": MATERIAL_LUNG,
            "adipose": MATERIAL_ADIPOSE,
            "soft_tissue": MATERIAL_SOFT_TISSUE,
            "bone": MATERIAL_BONE
        },
        "volume_dtype": {
            "material_ids": "uint8",
            "densities": "float16"
        }
    }
    
    with open(output_dir / "polyenergetic_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    return metadata


def print_polyenergetic_report(info: Dict) -> None:
    """Print formatted polyenergetic conversion report."""
    console.print("\n[bold cyan]Polyenergetic Conversion Report[/bold cyan]")
    
    # Conversion parameters
    param_table = Table(title="Conversion Parameters")
    param_table.add_column("Parameter", style="cyan")
    param_table.add_column("Value", style="green")
    
    param_table.add_row("Mode", info["mode"])
    param_table.add_row("Mean Energy", f"{info['mean_energy_kev']:.1f} keV")
    
    if "energy_range_kev" in info:
        e_min, e_max = info["energy_range_kev"]
        param_table.add_row("Energy Range", f"{e_min}-{e_max} keV")
        param_table.add_row("Energy Bins", str(info["num_energy_bins"]))
    
    console.print(param_table)
    
    # Material μ statistics
    if "material_mu_stats" in info:
        mu_table = Table(title="Linear Attenuation Coefficients (cm⁻¹)")
        mu_table.add_column("Material", style="cyan")
        mu_table.add_column("Min μ", style="red")
        mu_table.add_column("Mean μ", style="yellow")
        mu_table.add_column("Max μ", style="green")
        
        for material_name, stats in info["material_mu_stats"].items():
            mu_table.add_row(
                material_name,
                f"{stats['min']:.4f}",
                f"{stats['mean']:.4f}",
                f"{stats['max']:.4f}"
            )
        
        console.print(mu_table)


if __name__ == "__main__":
    # Test spectrum generation
    console.print("[bold]Testing X-ray Spectrum Generation[/bold]")
    energies, spectrum = get_120kvp_spectrum()
    
    # Find peak energy
    peak_idx = np.argmax(spectrum)
    peak_energy = energies[peak_idx]
    
    # Calculate mean energy
    mean_energy = np.sum(energies * spectrum) / np.sum(spectrum)
    
    console.print(f"Peak energy: {peak_energy} keV")
    console.print(f"Mean energy: {mean_energy:.1f} keV")
    console.print(f"K-edge peaks visible at 59 and 67-69 keV")
    
    # Test attenuation coefficients
    console.print("\n[bold]Testing Mass Attenuation Coefficients[/bold]")
    mu_tables = get_nist_mass_attenuation_coefficients()
    
    # Show values at 70 keV (mean energy region)
    energy_70_idx = np.where(energies == 70)[0][0]
    
    table = Table(title="μ/ρ at 70 keV (cm²/g)")
    table.add_column("Material", style="cyan")
    table.add_column("μ/ρ", style="green")
    
    for material_id, material_name in MATERIAL_NAMES.items():
        mu_over_rho = mu_tables[material_id][energy_70_idx]
        table.add_row(material_name, f"{mu_over_rho:.4f}")
    
    console.print(table)