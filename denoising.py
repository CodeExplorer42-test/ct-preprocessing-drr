"""Noise suppression module for CT preprocessing using scikit-image methods."""

from typing import Tuple, Dict, Optional
import numpy as np
import SimpleITK as sitk
from skimage import restoration, filters
from scipy.ndimage import gaussian_filter
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
import warnings

console = Console()
warnings.filterwarnings('ignore', category=UserWarning)


def estimate_noise_level(image: sitk.Image) -> float:
    """
    Estimate noise level in CT image using robust MAD estimator.
    
    Args:
        image: SimpleITK image
        
    Returns:
        Estimated noise standard deviation
    """
    array = sitk.GetArrayFromImage(image)
    
    # Use high-pass filter to isolate noise
    # Apply Laplacian to get noise component
    from scipy.ndimage import laplace
    noise_map = laplace(array.astype(np.float32))
    
    # Robust noise estimation using Median Absolute Deviation
    # σ = 1.4826 * MAD(noise) / sqrt(6)
    mad = np.median(np.abs(noise_map - np.median(noise_map)))
    sigma = 1.4826 * mad / np.sqrt(6)
    
    return float(sigma)


def denoise_nlm(image: sitk.Image, h: Optional[float] = None, fast_mode: bool = True) -> Tuple[sitk.Image, Dict]:
    """
    Apply Non-Local Means denoising to CT image.
    
    Args:
        image: Input SimpleITK image
        h: Filter strength (auto-estimated if None)
        fast_mode: Use fast mode for NLM
        
    Returns:
        Tuple of (denoised image, info dict)
    """
    array = sitk.GetArrayFromImage(image)
    
    # Estimate noise if h not provided
    if h is None:
        sigma = estimate_noise_level(image)
        h = 0.8 * sigma  # Conservative filtering
    
    info = {
        "method": "Non-Local Means",
        "estimated_noise": sigma if h is None else "Not estimated",
        "h_parameter": h,
        "fast_mode": fast_mode
    }
    
    # Process slice by slice for memory efficiency
    denoised = np.zeros_like(array)
    patch_size = 5
    patch_distance = 11 if fast_mode else 17
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Applying NLM denoising...", total=array.shape[0])
        
        for i in range(array.shape[0]):
            # Convert to float and normalize
            slice_data = array[i].astype(np.float32)
            slice_min = slice_data.min()
            slice_max = slice_data.max()
            
            if slice_max > slice_min:
                slice_norm = (slice_data - slice_min) / (slice_max - slice_min)
                
                # Apply NLM
                denoised_slice = restoration.denoise_nl_means(
                    slice_norm,
                    h=h / (slice_max - slice_min),  # Scale h to normalized range
                    patch_size=patch_size,
                    patch_distance=patch_distance,
                    fast_mode=fast_mode,
                    preserve_range=True
                )
                
                # Denormalize
                denoised[i] = denoised_slice * (slice_max - slice_min) + slice_min
            else:
                denoised[i] = slice_data
            
            progress.update(task, advance=1)
    
    # Create output image
    denoised_image = sitk.GetImageFromArray(denoised)
    denoised_image.CopyInformation(image)
    
    return denoised_image, info


def denoise_bilateral(image: sitk.Image, sigma_spatial: float = 3.0, sigma_range: Optional[float] = None) -> Tuple[sitk.Image, Dict]:
    """
    Apply bilateral filtering for edge-preserving denoising.
    
    Args:
        image: Input SimpleITK image
        sigma_spatial: Standard deviation for spatial kernel
        sigma_range: Standard deviation for range kernel (auto if None)
        
    Returns:
        Tuple of (denoised image, info dict)
    """
    array = sitk.GetArrayFromImage(image)
    
    # Auto-estimate sigma_range if not provided
    if sigma_range is None:
        noise_level = estimate_noise_level(image)
        sigma_range = 2.0 * noise_level
    
    info = {
        "method": "Bilateral Filter",
        "sigma_spatial": sigma_spatial,
        "sigma_range": sigma_range
    }
    
    # Apply bilateral filter slice by slice
    denoised = np.zeros_like(array)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Applying bilateral filter...", total=array.shape[0])
        
        for i in range(array.shape[0]):
            denoised[i] = restoration.denoise_bilateral(
                array[i].astype(np.float32),
                sigma_color=sigma_range,
                sigma_spatial=sigma_spatial,
                preserve_range=True
            )
            progress.update(task, advance=1)
    
    # Create output image
    denoised_image = sitk.GetImageFromArray(denoised)
    denoised_image.CopyInformation(image)
    
    return denoised_image, info


def denoise_gaussian(image: sitk.Image, sigma: Optional[float] = None) -> Tuple[sitk.Image, Dict]:
    """
    Apply Gaussian filtering (baseline method).
    
    Args:
        image: Input SimpleITK image
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Tuple of (denoised image, info dict)
    """
    if sigma is None:
        noise_level = estimate_noise_level(image)
        sigma = 0.5 * noise_level  # Conservative smoothing
    
    info = {
        "method": "Gaussian Filter",
        "sigma": sigma
    }
    
    # Use SimpleITK's smoothing for efficiency
    denoised_image = sitk.SmoothingRecursiveGaussian(image, sigma)
    
    return denoised_image, info


def denoise_adaptive(image: sitk.Image, method: str = "nlm") -> Tuple[sitk.Image, Dict]:
    """
    Apply adaptive denoising based on estimated noise level.
    
    Args:
        image: Input SimpleITK image
        method: Denoising method ("nlm", "bilateral", "gaussian")
        
    Returns:
        Tuple of (denoised image, info dict)
    """
    # Estimate noise level
    noise_level = estimate_noise_level(image)
    
    console.print(f"[cyan]Estimated noise level: {noise_level:.2f} HU[/cyan]")
    
    # Choose parameters based on noise level
    if noise_level < 10:
        console.print("[green]Low noise detected - applying light denoising[/green]")
        if method == "nlm":
            return denoise_nlm(image, h=0.6 * noise_level, fast_mode=True)
        elif method == "bilateral":
            return denoise_bilateral(image, sigma_spatial=2.0, sigma_range=1.5 * noise_level)
        else:
            return denoise_gaussian(image, sigma=0.3 * noise_level)
    
    elif noise_level < 20:
        console.print("[yellow]Moderate noise detected - applying standard denoising[/yellow]")
        if method == "nlm":
            return denoise_nlm(image, h=0.8 * noise_level, fast_mode=True)
        elif method == "bilateral":
            return denoise_bilateral(image, sigma_spatial=3.0, sigma_range=2.0 * noise_level)
        else:
            return denoise_gaussian(image, sigma=0.5 * noise_level)
    
    else:
        console.print("[red]High noise detected - applying strong denoising[/red]")
        if method == "nlm":
            return denoise_nlm(image, h=1.0 * noise_level, fast_mode=False)
        elif method == "bilateral":
            return denoise_bilateral(image, sigma_spatial=5.0, sigma_range=3.0 * noise_level)
        else:
            return denoise_gaussian(image, sigma=0.7 * noise_level)


if __name__ == "__main__":
    from pathlib import Path
    from dicom_loader import load_dicom_series
    from resampling import resample_to_isotropic
    import time
    
    # Load and resample test data
    dicom_dir = Path("1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720603192")
    
    if dicom_dir.exists():
        console.print("[green]Loading DICOM series...[/green]")
        image, metadata = load_dicom_series(dicom_dir)
        
        # First resample to isotropic
        console.print("\n[green]Resampling to isotropic spacing...[/green]")
        resampled, _ = resample_to_isotropic(image, target_spacing=0.5)
        
        # Test adaptive denoising with default method
        console.print("\n[bold cyan]Testing Adaptive Denoising[/bold cyan]")
        
        # Use NLM as default method (best quality)
        method = "nlm"
        console.print(f"\n[bold]Testing {method.upper()} denoising...[/bold]")
        
        start_time = time.time()
        denoised, info = denoise_adaptive(resampled, method=method)
        elapsed = time.time() - start_time
        
        # Calculate statistics
        original_array = sitk.GetArrayFromImage(resampled)
        denoised_array = sitk.GetArrayFromImage(denoised)
        
        # Simple SNR improvement estimate
        signal = np.mean(np.abs(denoised_array[denoised_array > -900]))
        noise_reduction = np.std(original_array) - np.std(denoised_array)
        
        console.print(f"Method: [green]{info['method']}[/green]")
        console.print(f"Processing time: [yellow]{elapsed:.1f}s[/yellow]")
        console.print(f"Noise reduction: [cyan]{noise_reduction:.1f} HU[/cyan]")
        console.print(f"Estimated SNR improvement: [cyan]{(noise_reduction/np.std(original_array)*100):.1f}%[/cyan]")
    else:
        console.print("[red]DICOM directory not found![/red]")