"""
Anti-Crosstalk Mask Model.

This module implements the focused tungsten grid mask that prevents X-rays
from one source from contaminating the detector intended for the other source
in the stereo imaging system.
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass

from materials_db import get_materials_db


@dataclass
class CrosstalkMaskSpecification:
    """Specifications for the anti-crosstalk mask."""
    material: str = 'tungsten'
    grid_ratio: float = 10.0  # Height:width ratio
    septa_thickness_mm: float = 0.025  # 25 µm tungsten septa
    slit_width_mm: float = 0.475  # 475 µm air gaps
    septa_height_mm: float = 4.75  # 4.75 mm height
    grid_period_mm: float = 0.5  # Total period (matches pixel pitch)
    primary_transmission: float = 0.95  # Geometric transmission
    focusing_distance_mm: float = 1500.0  # Source-to-detector distance
    stereo_angle_deg: float = 6.0  # ±6° stereo angle
    
    @property
    def geometric_transmission(self) -> float:
        """Calculate geometric transmission from grid parameters."""
        return self.slit_width_mm / self.grid_period_mm
    
    @property
    def aspect_ratio(self) -> float:
        """Calculate aspect ratio of the grid."""
        return self.septa_height_mm / self.slit_width_mm


class CrosstalkMask:
    """
    Model of the anti-crosstalk mask for stereo DRR imaging.
    
    This mask is positioned between the front and rear detectors to block
    off-axis radiation from the opposing stereo source while allowing
    high transmission of the primary beam.
    """
    
    def __init__(self, spec: Optional[CrosstalkMaskSpecification] = None):
        """
        Initialize the crosstalk mask model.
        
        Args:
            spec: Mask specifications (uses defaults if None)
        """
        self.spec = spec or CrosstalkMaskSpecification()
        self.materials_db = get_materials_db()
        
        # Validate specifications
        self._validate_specifications()
        
    def _validate_specifications(self):
        """Validate mask specifications for consistency."""
        # Check grid period
        expected_period = self.spec.septa_thickness_mm + self.spec.slit_width_mm
        if abs(expected_period - self.spec.grid_period_mm) > 1e-6:
            raise ValueError(
                f"Grid period ({self.spec.grid_period_mm}mm) must equal "
                f"septa + slit width ({expected_period}mm)"
            )
            
        # Check grid ratio
        expected_ratio = self.spec.septa_height_mm / self.spec.slit_width_mm
        if abs(expected_ratio - self.spec.grid_ratio) > 0.1:
            raise ValueError(
                f"Grid ratio ({self.spec.grid_ratio}) inconsistent with "
                f"height/width ({expected_ratio})"
            )
            
    def generate_mask_pattern(self, width_pixels: int, height_pixels: int,
                            pixel_size_mm: float = 0.5) -> np.ndarray:
        """
        Generate 2D binary mask pattern.
        
        Args:
            width_pixels: Detector width in pixels
            height_pixels: Detector height in pixels  
            pixel_size_mm: Pixel size in mm
            
        Returns:
            Binary mask array (1=open/air, 0=blocked/tungsten)
        """
        # Create coordinate grids
        x_mm = np.arange(width_pixels) * pixel_size_mm
        y_mm = np.arange(height_pixels) * pixel_size_mm
        
        # Create 1D mask pattern (periodic)
        # Phase aligned so septa are centered between pixels
        x_phase = x_mm % self.spec.grid_period_mm
        mask_1d = x_phase < self.spec.slit_width_mm
        
        # Extend to 2D (assuming 1D grid oriented horizontally)
        mask_2d = np.tile(mask_1d[np.newaxis, :], (height_pixels, 1))
        
        return mask_2d.astype(np.float32)
    
    def calculate_crosstalk_attenuation(self, incident_angle_deg: float,
                                      energy_kev: float) -> float:
        """
        Calculate attenuation factor for off-axis radiation.
        
        Args:
            incident_angle_deg: Angle of incident ray relative to normal
            energy_kev: X-ray energy in keV
            
        Returns:
            Attenuation factor (0-1, where 0 is complete blocking)
        """
        # For the stereo geometry, unwanted rays arrive at ~12° (2 * 6°)
        angle_rad = np.deg2rad(incident_angle_deg)
        
        # Grid acceptance angle
        acceptance_angle_rad = np.arctan(self.spec.slit_width_mm / self.spec.septa_height_mm)
        acceptance_angle_deg = np.rad2deg(acceptance_angle_rad)
        
        if abs(incident_angle_deg) > acceptance_angle_deg:
            # Ray will hit septa - for high grid ratio, it traverses significant tungsten
            # For rays beyond acceptance angle, they hit the septa walls
            # The path length depends on the geometry
            
            # For a ray at angle θ hitting a vertical septum of height h:
            # - If coming from the side, it travels through the full height
            # - Minimum path is h/cos(θ) when ray is aligned with septa
            # For crosstalk at 12°, rays traverse substantial tungsten
            
            # Conservative estimate: ray travels through at least one septum width
            # at the given angle, but for high angles it's much more
            if abs(incident_angle_deg) > 10:
                # High angle rays traverse multiple septa or full height
                path_length_mm = self.spec.septa_height_mm * 0.5  # Conservative
            else:
                # Lower angles might just clip a septum
                path_length_mm = self.spec.septa_thickness_mm / np.cos(angle_rad)
            
            # Get attenuation through tungsten
            transmission = self.materials_db.get_transmission(
                self.spec.material, path_length_mm, energy_kev
            )
            
            return float(transmission)
        else:
            # Ray passes through slit without hitting septa
            return 1.0
    
    def calculate_primary_transmission_map(self, detector_width_pixels: int,
                                         detector_height_pixels: int,
                                         source_position_mm: Tuple[float, float, float],
                                         pixel_size_mm: float = 0.5) -> np.ndarray:
        """
        Calculate spatially-varying primary transmission due to grid focusing.
        
        Args:
            detector_width_pixels: Detector width in pixels
            detector_height_pixels: Detector height in pixels
            source_position_mm: Source position (x, y, z) in mm
            pixel_size_mm: Pixel size in mm
            
        Returns:
            Transmission map accounting for grid focusing effects
        """
        # For a perfectly focused grid at the correct SDD, transmission is uniform
        # Small deviations from focus cause slight vignetting at edges
        
        # Create detector coordinate grid
        x = (np.arange(detector_width_pixels) - detector_width_pixels/2) * pixel_size_mm
        y = (np.arange(detector_height_pixels) - detector_height_pixels/2) * pixel_size_mm
        xx, yy = np.meshgrid(x, y)
        
        # Calculate ray angles from source to each detector pixel
        detector_z = source_position_mm[2] + self.spec.focusing_distance_mm
        dx = xx - source_position_mm[0]
        dy = yy - source_position_mm[1]
        dz = detector_z - source_position_mm[2]
        
        # Angle relative to central ray
        angles_rad = np.arctan(np.sqrt(dx**2 + dy**2) / dz)
        
        # For focused grid, transmission drops slightly at oblique angles
        # Model as cosine factor for small angles
        transmission_map = self.spec.primary_transmission * np.cos(angles_rad)
        
        # Ensure physically valid range
        transmission_map = np.clip(transmission_map, 0, 1)
        
        return transmission_map
    
    def estimate_scatter_rejection(self, scatter_angle_deg: float) -> float:
        """
        Estimate scatter rejection capability of the grid.
        
        Args:
            scatter_angle_deg: Scattering angle in degrees
            
        Returns:
            Rejection factor (0=complete rejection, 1=no rejection)
        """
        # Grid rejects scatter based on angle relative to grid acceptance
        acceptance_angle_deg = np.rad2deg(np.arctan(1/self.spec.grid_ratio))
        
        if abs(scatter_angle_deg) > acceptance_angle_deg:
            # Scattered photon will hit septa
            return 0.0
        else:
            # Scattered photon passes through
            # Partial rejection based on probability
            rejection = 1.0 - (abs(scatter_angle_deg) / acceptance_angle_deg)
            return float(rejection)
    
    def calculate_effective_grid_ratio(self, energy_kev: float) -> float:
        """
        Calculate effective grid ratio accounting for septal penetration.
        
        Args:
            energy_kev: X-ray energy in keV
            
        Returns:
            Effective grid ratio
        """
        # At higher energies, some penetration through septa occurs
        # This effectively reduces the grid ratio
        
        # Calculate transmission through full septa height
        septa_transmission = self.materials_db.get_transmission(
            self.spec.material, self.spec.septa_height_mm, energy_kev
        )
        
        # Effective height is reduced by penetration
        penetration_depth_mm = -np.log(0.01) / (
            self.materials_db.get_linear_atten_coeff(self.spec.material, energy_kev) * 10
        )
        
        effective_height = max(0, self.spec.septa_height_mm - penetration_depth_mm)
        effective_ratio = effective_height / self.spec.slit_width_mm
        
        return float(effective_ratio)
    
    def get_crosstalk_coefficient(self, source_angle_deg: float = 12.0,
                                mean_energy_kev: float = 80.0) -> float:
        """
        Get overall crosstalk coefficient for the mask.
        
        Args:
            source_angle_deg: Angle between stereo sources (default 12°)
            mean_energy_kev: Mean energy of hardened spectrum
            
        Returns:
            Crosstalk coefficient (fraction of unwanted signal transmitted)
        """
        # Calculate attenuation at the crosstalk angle
        attenuation = self.calculate_crosstalk_attenuation(
            source_angle_deg, mean_energy_kev
        )
        
        # Account for geometric factors
        # Most rays hit septa, but some may pass through slits at edges
        geometric_factor = 0.001  # <0.1% geometric leakage
        
        # Total crosstalk coefficient
        crosstalk = attenuation * geometric_factor
        
        return float(crosstalk)


def test_crosstalk_mask():
    """Test the crosstalk mask functionality."""
    mask = CrosstalkMask()
    
    print("Anti-Crosstalk Mask Test")
    print("=" * 60)
    
    # Print specifications
    print("\nMask Specifications:")
    print(f"  Material: {mask.spec.material}")
    print(f"  Grid ratio: {mask.spec.grid_ratio}:1")
    print(f"  Septa thickness: {mask.spec.septa_thickness_mm*1000:.0f} µm")
    print(f"  Slit width: {mask.spec.slit_width_mm*1000:.0f} µm")
    print(f"  Height: {mask.spec.septa_height_mm} mm")
    print(f"  Primary transmission: {mask.spec.primary_transmission*100:.1f}%")
    
    # Test crosstalk attenuation
    print("\nCrosstalk Attenuation at Various Angles:")
    angles = [0, 3, 6, 12, 20]  # degrees
    energies = [60, 80, 100]  # keV
    
    for angle in angles:
        print(f"\n  {angle}° incident angle:")
        for energy in energies:
            atten = mask.calculate_crosstalk_attenuation(angle, energy)
            print(f"    {energy} keV: {atten:.2e} ({(1-atten)*100:.1f}% blocked)")
            
    # Test effective grid ratio
    print("\nEffective Grid Ratio vs Energy:")
    for energy in [20, 60, 100, 140]:
        eff_ratio = mask.calculate_effective_grid_ratio(energy)
        print(f"  {energy} keV: {eff_ratio:.1f}:1")
        
    # Test overall crosstalk coefficient
    print("\nOverall Crosstalk Coefficient:")
    crosstalk = mask.get_crosstalk_coefficient()
    print(f"  Crosstalk transmission: {crosstalk:.2e} ({crosstalk*100:.3f}%)")
    print(f"  Crosstalk suppression: >{(1-crosstalk)*100:.1f}%")
    
    # Generate sample mask pattern
    print("\nGenerating mask pattern (10x10 pixels)...")
    pattern = mask.generate_mask_pattern(10, 10)
    print("Pattern (1=open, 0=blocked):")
    print(pattern.astype(int))


if __name__ == '__main__':
    test_crosstalk_mask()