"""DICOM loading module with metadata extraction for CT preprocessing."""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import SimpleITK as sitk
import pydicom
from rich.console import Console
from rich.table import Table

console = Console()


def load_dicom_series(dicom_path: Path) -> Tuple[sitk.Image, Dict[str, any]]:
    """
    Load a DICOM series from a directory and extract metadata.
    
    Args:
        dicom_path: Path to directory containing DICOM files
        
    Returns:
        Tuple of (SimpleITK Image, metadata dictionary)
    """
    # Read the DICOM series
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(dicom_path))
    
    if not dicom_names:
        raise ValueError(f"No DICOM series found in {dicom_path}")
    
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Extract metadata from first slice
    first_dcm = pydicom.dcmread(dicom_names[0])
    
    # Convert DICOM values to JSON-serializable types
    def convert_to_serializable(value):
        """Convert DICOM values to JSON-serializable types."""
        if value is None:
            return None
        elif hasattr(value, '__iter__') and not isinstance(value, str):
            # Handle MultiValue and lists
            return [float(v) if isinstance(v, (int, float)) else str(v) for v in value]
        elif isinstance(value, (int, float)):
            return float(value)
        else:
            return str(value)
    
    metadata = {
        "patient_id": convert_to_serializable(getattr(first_dcm, "PatientID", "Unknown")),
        "study_date": convert_to_serializable(getattr(first_dcm, "StudyDate", "Unknown")),
        "modality": convert_to_serializable(getattr(first_dcm, "Modality", "Unknown")),
        "manufacturer": convert_to_serializable(getattr(first_dcm, "Manufacturer", "Unknown")),
        "kvp": convert_to_serializable(getattr(first_dcm, "KVP", None)),
        "slice_thickness": convert_to_serializable(getattr(first_dcm, "SliceThickness", None)),
        "pixel_spacing": convert_to_serializable(getattr(first_dcm, "PixelSpacing", None)),
        "rows": convert_to_serializable(getattr(first_dcm, "Rows", None)),
        "columns": convert_to_serializable(getattr(first_dcm, "Columns", None)),
        "num_slices": len(dicom_names),
        "reconstruction_kernel": convert_to_serializable(getattr(first_dcm, "ConvolutionKernel", "Unknown")),
    }
    
    # Get spacing from SimpleITK image
    spacing = image.GetSpacing()
    metadata["spacing_sitk"] = spacing
    metadata["size"] = image.GetSize()
    metadata["origin"] = image.GetOrigin()
    metadata["direction"] = image.GetDirection()
    
    return image, metadata


def analyze_ct_characteristics(image: sitk.Image) -> Dict[str, any]:
    """
    Analyze CT image characteristics including HU statistics.
    
    Args:
        image: SimpleITK image
        
    Returns:
        Dictionary of image characteristics
    """
    # Convert to numpy array
    array = sitk.GetArrayFromImage(image)
    
    # Calculate statistics
    stats = {
        "min_hu": float(np.min(array)),
        "max_hu": float(np.max(array)),
        "mean_hu": float(np.mean(array)),
        "std_hu": float(np.std(array)),
        "shape": array.shape,
        "dtype": str(array.dtype),
        "memory_mb": array.nbytes / (1024 * 1024),
    }
    
    # Histogram of HU values
    hist, bin_edges = np.histogram(array.flatten(), bins=100, range=(-1000, 3000))
    stats["hu_histogram"] = {
        "counts": hist.tolist(),
        "bin_edges": bin_edges.tolist()
    }
    
    # Identify tissue peaks
    air_mask = (array > -1100) & (array < -900)
    if np.any(air_mask):
        stats["air_hu_mean"] = float(np.mean(array[air_mask]))
    
    fat_mask = (array > -150) & (array < -50)
    if np.any(fat_mask):
        stats["fat_hu_mean"] = float(np.mean(array[fat_mask]))
    
    water_mask = (array > -50) & (array < 50)
    if np.any(water_mask):
        stats["water_hu_mean"] = float(np.mean(array[water_mask]))
    
    return stats


def print_scan_info(metadata: Dict[str, any], stats: Dict[str, any]) -> None:
    """Print formatted scan information."""
    console.print("\n[bold cyan]DICOM Scan Information[/bold cyan]")
    
    # Basic metadata table
    table = Table(title="Metadata")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Patient ID", str(metadata["patient_id"]))
    table.add_row("Study Date", str(metadata["study_date"]))
    table.add_row("Modality", str(metadata["modality"]))
    table.add_row("Manufacturer", str(metadata["manufacturer"]))
    table.add_row("Reconstruction Kernel", str(metadata["reconstruction_kernel"]))
    table.add_row("kVp", str(metadata["kvp"]))
    table.add_row("Number of Slices", str(metadata["num_slices"]))
    
    console.print(table)
    
    # Spacing information
    console.print("\n[bold cyan]Spatial Information[/bold cyan]")
    spacing_table = Table()
    spacing_table.add_column("Dimension", style="cyan")
    spacing_table.add_column("Size (voxels)", style="green")
    spacing_table.add_column("Spacing (mm)", style="yellow")
    
    spacing_table.add_row("X", str(metadata["size"][0]), f"{metadata['spacing_sitk'][0]:.3f}")
    spacing_table.add_row("Y", str(metadata["size"][1]), f"{metadata['spacing_sitk'][1]:.3f}")
    spacing_table.add_row("Z", str(metadata["size"][2]), f"{metadata['spacing_sitk'][2]:.3f}")
    
    console.print(spacing_table)
    
    # HU statistics
    console.print("\n[bold cyan]Hounsfield Unit Statistics[/bold cyan]")
    hu_table = Table()
    hu_table.add_column("Statistic", style="cyan")
    hu_table.add_column("Value", style="green")
    
    hu_table.add_row("Min HU", f"{stats['min_hu']:.1f}")
    hu_table.add_row("Max HU", f"{stats['max_hu']:.1f}")
    hu_table.add_row("Mean HU", f"{stats['mean_hu']:.1f}")
    hu_table.add_row("Std HU", f"{stats['std_hu']:.1f}")
    
    if "air_hu_mean" in stats:
        hu_table.add_row("Air Peak", f"{stats['air_hu_mean']:.1f}")
    if "fat_hu_mean" in stats:
        hu_table.add_row("Fat Peak", f"{stats['fat_hu_mean']:.1f}")
    if "water_hu_mean" in stats:
        hu_table.add_row("Water Peak", f"{stats['water_hu_mean']:.1f}")
    
    console.print(hu_table)
    
    console.print(f"\n[yellow]Memory usage: {stats['memory_mb']:.1f} MB[/yellow]")


if __name__ == "__main__":
    # Test with the LIDC-IDRI data
    dicom_dir = Path("1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192")
    
    if dicom_dir.exists():
        console.print("[green]Loading DICOM series...[/green]")
        image, metadata = load_dicom_series(dicom_dir)
        
        console.print("[green]Analyzing CT characteristics...[/green]")
        stats = analyze_ct_characteristics(image)
        
        print_scan_info(metadata, stats)
    else:
        console.print("[red]DICOM directory not found![/red]")