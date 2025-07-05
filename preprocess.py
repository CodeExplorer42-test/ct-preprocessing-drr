"""Main CT preprocessing pipeline for high-fidelity DRR synthesis."""

import argparse
from pathlib import Path
from typing import Dict, Optional
import SimpleITK as sitk
from rich.console import Console
from rich.panel import Panel
from rich.progress import track
import json
import time

# Import all modules
from dicom_loader import load_dicom_series, print_scan_info, analyze_ct_characteristics
from resampling import resample_to_isotropic, print_resampling_info
from denoising import denoise_adaptive
from super_resolution import harmonize_slice_thickness
from artifact_correction import apply_artifact_correction
from hu_calibration import phantom_free_calibration, hu_to_attenuation_coefficient, print_calibration_report
from quality_control import generate_qc_report

console = Console()


class CTPreprocessor:
    """Main CT preprocessing pipeline class."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self.get_default_config()
        self.processing_info = {
            "steps": [],
            "timings": {},
            "errors": []
        }
    
    @staticmethod
    def get_default_config() -> Dict:
        """Get default preprocessing configuration."""
        return {
            "resampling": {
                "enabled": True,
                "target_spacing": 0.5,
                "interpolator": "bspline"
            },
            "denoising": {
                "enabled": True,
                "method": "nlm",  # "gaussian", "bilateral", "nlm"
                "adaptive": True
            },
            "super_resolution": {
                "enabled": True,
                "method": "interpolation"  # "interpolation" or "ml"
            },
            "artifact_correction": {
                "enabled": True,
                "correct_metal": True,
                "correct_beam_hardening": True,
                "correct_rings": False  # Off by default as less common in thoracic CT
            },
            "calibration": {
                "enabled": True,
                "energy_kev": 70.0
            },
            "quality_control": {
                "enabled": True,
                "save_report": True
            },
            "output": {
                "save_intermediate": False,
                "format": "nifti"  # "nifti" or "mha"
            }
        }
    
    def preprocess(self, dicom_path: Path, output_dir: Path) -> Dict:
        """
        Run complete preprocessing pipeline.
        
        Args:
            dicom_path: Path to DICOM directory
            output_dir: Output directory for processed files
            
        Returns:
            Dictionary with processing results
        """
        console.print(Panel("[bold cyan]CT Preprocessing Pipeline[/bold cyan]", expand=False))
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load DICOM
        console.print("\n[bold]Step 1: Loading DICOM Series[/bold]")
        start_time = time.time()
        
        try:
            image, metadata = load_dicom_series(dicom_path)
            stats = analyze_ct_characteristics(image)
            print_scan_info(metadata, stats)
            
            self.processing_info["steps"].append("dicom_loading")
            self.processing_info["timings"]["dicom_loading"] = time.time() - start_time
            self.processing_info["original_metadata"] = metadata
            
            current_image = image
            
        except Exception as e:
            error_msg = f"Failed to load DICOM: {str(e)}"
            console.print(f"[red]{error_msg}[/red]")
            self.processing_info["errors"].append(error_msg)
            raise
        
        # Step 2: Isotropic Resampling
        if self.config["resampling"]["enabled"]:
            console.print("\n[bold]Step 2: Isotropic Resampling[/bold]")
            start_time = time.time()
            
            try:
                interpolator_map = {
                    "linear": sitk.sitkLinear,
                    "bspline": sitk.sitkBSpline,
                    "lanczos": sitk.sitkLanczosWindowedSinc
                }
                interpolator = interpolator_map.get(
                    self.config["resampling"]["interpolator"],
                    sitk.sitkBSpline
                )
                
                resampled, resample_info = resample_to_isotropic(
                    current_image,
                    target_spacing=self.config["resampling"]["target_spacing"],
                    interpolator=interpolator
                )
                print_resampling_info(resample_info)
                
                self.processing_info["steps"].append("resampling")
                self.processing_info["timings"]["resampling"] = time.time() - start_time
                self.processing_info["resampling"] = resample_info
                
                if self.config["output"]["save_intermediate"]:
                    self._save_intermediate(resampled, output_dir / "01_resampled")
                
                current_image = resampled
                
            except Exception as e:
                error_msg = f"Resampling failed: {str(e)}"
                console.print(f"[red]{error_msg}[/red]")
                self.processing_info["errors"].append(error_msg)
        
        # Step 3: Slice Thickness Harmonization
        if self.config["super_resolution"]["enabled"]:
            console.print("\n[bold]Step 3: Slice Thickness Harmonization[/bold]")
            start_time = time.time()
            
            try:
                # Check if SR is needed based on ORIGINAL spacing
                original_spacing = image.GetSpacing()
                if original_spacing[2] / original_spacing[0] > 2.0:
                    console.print(f"[yellow]Original Z-spacing ({original_spacing[2]:.2f}mm) is {original_spacing[2]/original_spacing[0]:.1f}x larger than XY-spacing[/yellow]")
                    harmonized, sr_info = harmonize_slice_thickness(
                        current_image,
                        method=self.config["super_resolution"]["method"]
                    )
                else:
                    console.print("[green]Original slice thickness already reasonably isotropic[/green]")
                    harmonized = current_image
                    sr_info = {"method": "none", "reason": "original already isotropic"}
                
                console.print(f"[green]Method: {sr_info['method']}[/green]")
                if "upsampling_factor" in sr_info:
                    console.print(f"[green]Upsampling factor: {sr_info['upsampling_factor']:.1f}x[/green]")
                
                self.processing_info["steps"].append("super_resolution")
                self.processing_info["timings"]["super_resolution"] = time.time() - start_time
                self.processing_info["super_resolution"] = sr_info
                
                if self.config["output"]["save_intermediate"]:
                    self._save_intermediate(harmonized, output_dir / "02_harmonized")
                
                current_image = harmonized
                
            except Exception as e:
                error_msg = f"Slice harmonization failed: {str(e)}"
                console.print(f"[red]{error_msg}[/red]")
                self.processing_info["errors"].append(error_msg)
        
        # Step 4: Denoising
        if self.config["denoising"]["enabled"]:
            console.print("\n[bold]Step 4: Noise Suppression[/bold]")
            start_time = time.time()
            
            try:
                if self.config["denoising"]["adaptive"]:
                    denoised, denoise_info = denoise_adaptive(
                        current_image,
                        method=self.config["denoising"]["method"]
                    )
                else:
                    # Use fixed parameters based on method
                    from denoising import denoise_nlm, denoise_bilateral, denoise_gaussian
                    method_map = {
                        "nlm": denoise_nlm,
                        "bilateral": denoise_bilateral,
                        "gaussian": denoise_gaussian
                    }
                    denoise_func = method_map.get(
                        self.config["denoising"]["method"],
                        denoise_bilateral
                    )
                    denoised, denoise_info = denoise_func(current_image)
                
                console.print(f"[green]Method: {denoise_info['method']}[/green]")
                
                self.processing_info["steps"].append("denoising")
                self.processing_info["timings"]["denoising"] = time.time() - start_time
                self.processing_info["denoising"] = denoise_info
                
                if self.config["output"]["save_intermediate"]:
                    self._save_intermediate(denoised, output_dir / "03_denoised")
                
                current_image = denoised
                
            except Exception as e:
                error_msg = f"Denoising failed: {str(e)}"
                console.print(f"[red]{error_msg}[/red]")
                self.processing_info["errors"].append(error_msg)
        
        # Step 5: Artifact Correction
        if self.config["artifact_correction"]["enabled"]:
            console.print("\n[bold]Step 5: Artifact Correction[/bold]")
            start_time = time.time()
            
            try:
                corrected, artifact_info = apply_artifact_correction(
                    current_image,
                    correct_metal=self.config["artifact_correction"]["correct_metal"],
                    correct_beam_hardening=self.config["artifact_correction"]["correct_beam_hardening"],
                    correct_rings=self.config["artifact_correction"]["correct_rings"]
                )
                
                self.processing_info["steps"].append("artifact_correction")
                self.processing_info["timings"]["artifact_correction"] = time.time() - start_time
                self.processing_info["artifact_correction"] = artifact_info
                
                if self.config["output"]["save_intermediate"]:
                    self._save_intermediate(corrected, output_dir / "04_corrected")
                
                current_image = corrected
                
            except Exception as e:
                error_msg = f"Artifact correction failed: {str(e)}"
                console.print(f"[red]{error_msg}[/red]")
                self.processing_info["errors"].append(error_msg)
        
        # Step 6: HU Calibration and Conversion
        if self.config["calibration"]["enabled"]:
            console.print("\n[bold]Step 6: HU Calibration & μ Conversion[/bold]")
            start_time = time.time()
            
            try:
                # Calibrate HU values
                calibrated, cal_info = phantom_free_calibration(current_image)
                
                # Convert to attenuation coefficients
                mu_image, conv_info = hu_to_attenuation_coefficient(
                    calibrated,
                    energy_kev=self.config["calibration"]["energy_kev"]
                )
                
                print_calibration_report(cal_info, conv_info)
                
                self.processing_info["steps"].append("calibration")
                self.processing_info["timings"]["calibration"] = time.time() - start_time
                self.processing_info["calibration"] = {
                    "hu_calibration": cal_info,
                    "mu_conversion": conv_info
                }
                
                if self.config["output"]["save_intermediate"]:
                    self._save_intermediate(calibrated, output_dir / "05_calibrated")
                
                # Save both calibrated HU and mu versions
                current_image = calibrated
                final_mu_image = mu_image
                
            except Exception as e:
                error_msg = f"Calibration failed: {str(e)}"
                console.print(f"[red]{error_msg}[/red]")
                self.processing_info["errors"].append(error_msg)
                final_mu_image = None
        
        # Step 7: Quality Control
        if self.config["quality_control"]["enabled"]:
            console.print("\n[bold]Step 7: Quality Control[/bold]")
            start_time = time.time()
            
            try:
                qc_report_path = output_dir / "qc_report.json" if self.config["quality_control"]["save_report"] else None
                
                qc_report = generate_qc_report(
                    image,  # Original
                    current_image,  # Final processed
                    self.processing_info,
                    output_path=qc_report_path
                )
                
                self.processing_info["timings"]["quality_control"] = time.time() - start_time
                self.processing_info["qc_report"] = qc_report
                
            except Exception as e:
                error_msg = f"QC generation failed: {str(e)}"
                console.print(f"[red]{error_msg}[/red]")
                self.processing_info["errors"].append(error_msg)
        
        # Save final outputs
        console.print("\n[bold]Saving Final Outputs[/bold]")
        
        # Save preprocessed HU image
        output_path_hu = output_dir / f"preprocessed_hu.{self.config['output']['format']}"
        self._save_image(current_image, output_path_hu)
        console.print(f"[green]Saved preprocessed HU image: {output_path_hu}[/green]")
        
        # Save mu image if available
        if final_mu_image is not None:
            output_path_mu = output_dir / f"preprocessed_mu.{self.config['output']['format']}"
            self._save_image(final_mu_image, output_path_mu)
            console.print(f"[green]Saved attenuation coefficient image: {output_path_mu}[/green]")
        
        # Save processing info
        info_path = output_dir / "processing_info.json"
        with open(info_path, 'w') as f:
            json.dump(self.processing_info, f, indent=2)
        console.print(f"[green]Saved processing info: {info_path}[/green]")
        
        # Print summary
        self._print_summary()
        
        return {
            "success": len(self.processing_info["errors"]) == 0,
            "output_hu": output_path_hu,
            "output_mu": output_path_mu if final_mu_image is not None else None,
            "processing_info": self.processing_info
        }
    
    def _save_intermediate(self, image: sitk.Image, path: Path) -> None:
        """Save intermediate result."""
        path = path.with_suffix(f".{self.config['output']['format']}")
        self._save_image(image, path)
        console.print(f"  [dim]Saved intermediate: {path}[/dim]")
    
    def _save_image(self, image: sitk.Image, path: Path) -> None:
        """Save image in specified format."""
        if self.config["output"]["format"] == "nifti":
            path = path.with_suffix(".nii.gz")
        elif self.config["output"]["format"] == "mha":
            path = path.with_suffix(".mha")
        
        sitk.WriteImage(image, str(path), useCompression=True)
    
    def _print_summary(self) -> None:
        """Print processing summary."""
        console.print("\n[bold cyan]Processing Summary[/bold cyan]")
        
        # Timing summary
        total_time = sum(self.processing_info["timings"].values())
        console.print(f"\nTotal processing time: [yellow]{total_time:.1f}s[/yellow]")
        
        # Step timings
        for step, timing in self.processing_info["timings"].items():
            percentage = (timing / total_time) * 100
            console.print(f"  {step}: {timing:.1f}s ({percentage:.1f}%)")
        
        # Errors if any
        if self.processing_info["errors"]:
            console.print("\n[red]Errors encountered:[/red]")
            for error in self.processing_info["errors"]:
                console.print(f"  - {error}")
        else:
            console.print("\n[green]All steps completed successfully![/green]")


def main():
    """Command-line interface for CT preprocessing."""
    parser = argparse.ArgumentParser(
        description="Preprocess CT scans for high-fidelity DRR synthesis"
    )
    
    parser.add_argument(
        "dicom_path",
        type=Path,
        help="Path to DICOM directory"
    )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("preprocessed_output"),
        help="Output directory (default: preprocessed_output)"
    )
    
    parser.add_argument(
        "-c", "--config",
        type=Path,
        help="Configuration file (JSON)"
    )
    
    parser.add_argument(
        "--spacing",
        type=float,
        default=0.5,
        help="Target isotropic spacing in mm (default: 0.5)"
    )
    
    parser.add_argument(
        "--no-denoise",
        action="store_true",
        help="Skip denoising step"
    )
    
    parser.add_argument(
        "--no-artifacts",
        action="store_true",
        help="Skip artifact correction"
    )
    
    parser.add_argument(
        "--save-intermediate",
        action="store_true",
        help="Save intermediate results"
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config and args.config.exists():
        with open(args.config) as f:
            config = json.load(f)
    else:
        config = CTPreprocessor.get_default_config()
    
    # Override config with command-line arguments
    config["resampling"]["target_spacing"] = args.spacing
    config["denoising"]["enabled"] = not args.no_denoise
    config["artifact_correction"]["enabled"] = not args.no_artifacts
    config["output"]["save_intermediate"] = args.save_intermediate
    
    # Run preprocessing
    preprocessor = CTPreprocessor(config)
    
    try:
        result = preprocessor.preprocess(args.dicom_path, args.output)
        
        if result["success"]:
            console.print("\n[bold green]Preprocessing completed successfully![/bold green]")
        else:
            console.print("\n[bold yellow]Preprocessing completed with errors[/bold yellow]")
            
    except Exception as e:
        console.print(f"\n[bold red]Preprocessing failed: {str(e)}[/bold red]")
        raise


if __name__ == "__main__":
    main()