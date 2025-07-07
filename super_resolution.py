"""Slice-thickness harmonization and super-resolution module for CT preprocessing."""

from typing import Tuple, Dict, Optional
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn

console = Console()


class SimpleSR3D(nn.Module):
    """Simple 3D super-resolution network for z-axis enhancement."""
    
    def __init__(self, scale_factor: int = 2):
        super().__init__()
        self.scale_factor = scale_factor
        
        # Encoder
        self.conv1 = nn.Conv3d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        
        # Upsampling layers
        self.up1 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1)
        
        # Output
        self.conv_out = nn.Conv3d(32, 1, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Encoder
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        
        # Decoder with skip connections
        if self.scale_factor >= 2:
            x = F.relu(self.up1(x3))
            x = x + F.interpolate(x2, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        if self.scale_factor >= 4:
            x = F.relu(self.up2(x))
            x = x + F.interpolate(x1, size=x.shape[2:], mode='trilinear', align_corners=False)
        
        x = self.conv_out(x)
        return x


def interpolate_thick_slices(image: sitk.Image, target_z_spacing: float = 0.5) -> Tuple[sitk.Image, Dict]:
    """
    Interpolate thick slices using advanced spline interpolation.
    This is a simpler alternative to deep learning SR.
    
    Args:
        image: Input SimpleITK image
        target_z_spacing: Target spacing in z-direction
        
    Returns:
        Tuple of (interpolated image, info dict)
    """
    spacing = image.GetSpacing()
    size = image.GetSize()
    
    # Calculate new size for z-dimension
    new_z_size = int(size[2] * spacing[2] / target_z_spacing)
    new_size = [size[0], size[1], new_z_size]
    new_spacing = [spacing[0], spacing[1], target_z_spacing]
    
    info = {
        "method": "Spline Interpolation",
        "original_z_spacing": spacing[2],
        "target_z_spacing": target_z_spacing,
        "original_z_size": size[2],
        "new_z_size": new_z_size,
        "upsampling_factor": spacing[2] / target_z_spacing
    }
    
    # Use Lanczos interpolation for better detail preservation in z-direction
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(-1000)
    resampler.SetInterpolator(sitk.sitkLanczosWindowedSinc)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Interpolating thick slices...", total=1)
        interpolated = resampler.Execute(image)
        progress.update(task, advance=1)
    
    return interpolated, info


def apply_ml_super_resolution(image: sitk.Image, model_path: Optional[str] = None) -> Tuple[sitk.Image, Dict]:
    """
    Apply machine learning-based super-resolution.
    Note: This is a placeholder for a real trained model.
    
    Args:
        image: Input SimpleITK image
        model_path: Path to trained model (uses simple model if None)
        
    Returns:
        Tuple of (super-resolved image, info dict)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Get image data
    array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    
    # Calculate scale factor needed
    scale_factor = int(np.ceil(spacing[2] / spacing[0]))
    
    info = {
        "method": "ML Super-Resolution",
        "device": str(device),
        "scale_factor": scale_factor,
        "model": "SimpleSR3D" if model_path is None else model_path
    }
    
    # Initialize model
    model = SimpleSR3D(scale_factor=scale_factor).to(device)
    model.eval()
    
    # Process in chunks to manage memory
    chunk_size = 64  # Process 64 slices at a time
    num_chunks = (array.shape[0] + chunk_size - 1) // chunk_size
    
    # Prepare output array
    output_shape = (array.shape[0] * scale_factor, array.shape[1], array.shape[2])
    output = np.zeros(output_shape, dtype=array.dtype)
    
    with torch.no_grad():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Applying ML super-resolution...", total=num_chunks)
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min((i + 1) * chunk_size, array.shape[0])
                
                # Get chunk and normalize
                chunk = array[start_idx:end_idx].astype(np.float32)
                chunk_min = chunk.min()
                chunk_max = chunk.max()
                
                if chunk_max > chunk_min:
                    chunk = (chunk - chunk_min) / (chunk_max - chunk_min)
                    
                    # Add batch and channel dimensions
                    chunk_tensor = torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(device)
                    
                    # Apply model
                    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                        sr_chunk = model(chunk_tensor)
                    
                    # Convert back
                    sr_chunk = sr_chunk.squeeze().cpu().numpy()
                    sr_chunk = sr_chunk * (chunk_max - chunk_min) + chunk_min
                    
                    # Store in output
                    output[start_idx * scale_factor:end_idx * scale_factor] = sr_chunk
                
                progress.update(task, advance=1)
    
    # Create output image with updated spacing
    sr_image = sitk.GetImageFromArray(output)
    new_spacing = (spacing[0], spacing[1], spacing[2] / scale_factor)
    sr_image.SetSpacing(new_spacing)
    sr_image.SetOrigin(image.GetOrigin())
    sr_image.SetDirection(image.GetDirection())
    
    return sr_image, info


def harmonize_slice_thickness(image: sitk.Image, method: str = "interpolation", original_z_spacing: float = 2.5) -> Tuple[sitk.Image, Dict]:
    """
    Harmonize slice thickness using specified method.
    
    Args:
        image: Input SimpleITK image (already resampled)
        method: "interpolation" or "ml"
        original_z_spacing: Original z-spacing before resampling (for SR decision)
        
    Returns:
        Tuple of (harmonized image, info dict)
    """
    spacing = image.GetSpacing()
    
    # Check if we actually need SR - if already at or below target, skip
    if spacing[2] <= 0.5:
        console.print(f"[green]Z-spacing already optimal ({spacing[2]:.2f}mm), skipping additional SR[/green]")
        return image, {"method": "none", "reason": f"already at target resolution ({spacing[2]:.2f}mm)"}
    
    # Only apply SR if there's room for improvement
    console.print(f"[yellow]Applying z-axis enhancement (current: {spacing[2]:.1f}mm)[/yellow]")
    
    if method == "ml":
        # Use ML-based super-resolution
        return apply_ml_super_resolution(image)
    else:
        # Only enhance if it makes sense - don't oversample
        target_z = max(0.5, spacing[2] * 0.8)  # Target 0.5mm or 20% improvement
        if abs(target_z - spacing[2]) < 0.05:  # Less than 0.05mm difference
            console.print("[green]Spacing already near optimal, skipping SR[/green]")
            return image, {"method": "none", "reason": "minimal improvement possible"}
        
        return interpolate_thick_slices(image, target_z_spacing=target_z)


if __name__ == "__main__":
    from pathlib import Path
    from dicom_loader import load_dicom_series
    
    # Load test data
    dicom_dir = Path("1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192")
    
    if dicom_dir.exists():
        console.print("[green]Loading DICOM series...[/green]")
        image, metadata = load_dicom_series(dicom_dir)
        
        console.print(f"\nOriginal spacing: {image.GetSpacing()}")
        console.print(f"Original size: {image.GetSize()}")
        
        # Test interpolation method
        console.print("\n[bold cyan]Testing Interpolation-based Harmonization[/bold cyan]")
        harmonized, info = harmonize_slice_thickness(image, method="interpolation")
        
        console.print(f"\nHarmonized spacing: {harmonized.GetSpacing()}")
        console.print(f"Harmonized size: {harmonized.GetSize()}")
        console.print(f"Method: [green]{info['method']}[/green]")
        if 'upsampling_factor' in info:
            console.print(f"Upsampling factor: [yellow]{info['upsampling_factor']:.1f}x[/yellow]")
        
        # Calculate memory usage
        original_memory = np.prod(image.GetSize()) * 2 / (1024**2)  # MB
        harmonized_memory = np.prod(harmonized.GetSize()) * 2 / (1024**2)  # MB
        console.print(f"\nMemory usage: {original_memory:.1f} MB → {harmonized_memory:.1f} MB")
    else:
        console.print("[red]DICOM directory not found![/red]")