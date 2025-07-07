"""
Material Attenuation Look-Up Table Manager.

This module manages mass attenuation coefficients for polyenergetic X-ray
projection, preparing data for efficient GPU computation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from ray_projection_types import MaterialData, RayProjectionConfig
from materials_db import MaterialsDatabase

logger = logging.getLogger(__name__)


class MaterialAttenuationLUT:
    """Manages mass attenuation coefficients for polyenergetic projection.
    
    This class handles:
    1. Loading material properties from the database
    2. Building GPU-optimized lookup tables
    3. Applying detector response to X-ray spectrum
    4. Preparing data for Metal shader consumption
    """
    
    def __init__(self, materials_db: Optional[MaterialsDatabase] = None):
        """Initialize the material attenuation LUT manager.
        
        Args:
            materials_db: MaterialsDatabase instance. If None, creates a new one.
        """
        self.materials_db = materials_db or MaterialsDatabase()
        
        # Standard material mapping for medical CT
        self.material_mapping = {
            0: 'air',
            1: 'lung',
            2: 'adipose',
            3: 'soft_tissue',
            4: 'bone'
        }
        
        # Cache for computed LUTs
        self._lut_cache: Dict[Tuple[float, float, int], np.ndarray] = {}
        
        logger.info("Initialized MaterialAttenuationLUT with %d materials", 
                   len(self.material_mapping))
    
    def _get_medical_mass_atten_coeff(self, material: str, energies: np.ndarray) -> np.ndarray:
        """Get mass attenuation coefficients for medical materials.
        
        These are representative values based on NIST data for medical tissues.
        In a production system, these would come from comprehensive databases.
        
        Args:
            material: Material name ('lung', 'adipose', 'soft_tissue', 'bone')
            energies: Energy array in keV
            
        Returns:
            Mass attenuation coefficients in cm²/g
        """
        # Representative values at 70 keV (approximate)
        # These are simplified - real implementation would interpolate from full NIST tables
        base_values = {
            'lung': 0.2,      # Low density tissue
            'adipose': 0.19,  # Fat tissue
            'soft_tissue': 0.2,  # Muscle/organs
            'bone': 0.35     # Cortical bone
        }
        
        if material not in base_values:
            raise ValueError(f"Unknown medical material: {material}")
        
        # Create energy-dependent curve
        # Simplified model: μ/ρ ∝ E^(-3) for photoelectric region
        # and constant for Compton region
        base_mu = base_values[material]
        
        # Simple energy dependence model
        mu_over_rho = np.zeros_like(energies)
        for i, E in enumerate(energies):
            if E < 50:
                # Photoelectric dominated
                mu_over_rho[i] = base_mu * (70.0 / E) ** 2.5
            else:
                # Compton dominated
                mu_over_rho[i] = base_mu * (70.0 / E) ** 0.3
        
        return mu_over_rho.astype(np.float32)
    
    def build_gpu_lut(self, 
                     energy_range: Tuple[float, float],
                     num_bins: int) -> np.ndarray:
        """Build optimized LUT for GPU constant memory.
        
        The LUT is organized as [material_id][energy_bin] = μ/ρ
        This layout is optimized for SIMD access patterns where
        we process multiple energy bins simultaneously.
        
        Args:
            energy_range: (min_energy, max_energy) in keV
            num_bins: Number of energy bins
            
        Returns:
            2D array of shape (num_materials, num_bins) with μ/ρ values
        """
        cache_key = (energy_range[0], energy_range[1], num_bins)
        if cache_key in self._lut_cache:
            logger.debug("Using cached LUT for energy range %.1f-%.1f keV",
                        energy_range[0], energy_range[1])
            return self._lut_cache[cache_key]
        
        logger.info("Building material LUT for %.1f-%.1f keV with %d bins",
                   energy_range[0], energy_range[1], num_bins)
        
        # Create energy array
        energies = np.linspace(energy_range[0], energy_range[1], num_bins)
        
        # Build LUT
        num_materials = len(self.material_mapping)
        lut = np.zeros((num_materials, num_bins), dtype=np.float32)
        
        for mat_id, mat_name in self.material_mapping.items():
            # Get mass attenuation coefficients
            # For medical materials, use hardcoded values from NIST
            if mat_name in ['lung', 'adipose', 'soft_tissue', 'bone']:
                mu_over_rho = self._get_medical_mass_atten_coeff(mat_name, energies)
            else:
                # For other materials, use the database
                mu_over_rho = self.materials_db.get_mass_atten_coeff(mat_name, energies)
            lut[mat_id, :] = mu_over_rho
            
            logger.debug("Material %d (%s): μ/ρ range [%.4f, %.4f] cm²/g",
                        mat_id, mat_name, mu_over_rho.min(), mu_over_rho.max())
        
        # Cache the result
        self._lut_cache[cache_key] = lut
        
        return lut
    
    def apply_detector_response(self, 
                              source_spectrum: np.ndarray,
                              detector_model: Optional[Dict] = None) -> np.ndarray:
        """Pre-multiply source spectrum with detector transmission.
        
        This folds the detector response into the source spectrum,
        avoiding the need for a separate multiplication in the GPU kernel.
        
        N'_0(E) = N_0(E) × T_front(E) × T_mask(E)
        
        Args:
            source_spectrum: Original X-ray source spectrum
            detector_model: Detector transmission data (optional)
            
        Returns:
            Effective source spectrum with detector response applied
        """
        if detector_model is None:
            logger.warning("No detector model provided, using source spectrum as-is")
            return source_spectrum.copy()
        
        logger.info("Applying detector response to source spectrum")
        
        # Extract transmission factors
        front_transmission = detector_model.get('front_transmission', np.ones_like(source_spectrum))
        mask_transmission = detector_model.get('mask_transmission', np.ones_like(source_spectrum))
        
        # Apply detector response
        effective_spectrum = source_spectrum * front_transmission * mask_transmission
        
        # Log reduction factors
        total_photons_original = np.sum(source_spectrum)
        total_photons_effective = np.sum(effective_spectrum)
        reduction_factor = total_photons_effective / total_photons_original
        
        logger.info("Detector response applied: %.1f%% photon transmission",
                   reduction_factor * 100)
        
        return effective_spectrum.astype(np.float32)
    
    def prepare_for_gpu(self, 
                       config: RayProjectionConfig,
                       detector_model: Optional[Dict] = None) -> Dict[str, np.ndarray]:
        """Prepare all material data for GPU upload.
        
        This method creates GPU-ready arrays with proper memory layout
        and data types for efficient Metal shader access.
        
        Args:
            config: Ray projection configuration
            detector_model: Optional detector transmission data
            
        Returns:
            Dictionary containing:
                - 'material_lut': 2D array of μ/ρ values
                - 'material_lut_flat': Flattened version for linear access
                - 'spectrum': Effective source spectrum
                - 'energies': Energy bin centers
        """
        logger.info("Preparing material data for GPU")
        
        # Build material LUT
        material_lut = self.build_gpu_lut(config.energy_range, config.num_energy_bins)
        
        # Flatten for GPU (row-major order for SIMD access)
        material_lut_flat = material_lut.flatten().astype(np.float32)
        
        # Create energy array
        energies = np.linspace(config.energy_range[0], config.energy_range[1], 
                             config.num_energy_bins, dtype=np.float32)
        
        # Generate typical diagnostic X-ray spectrum if not provided
        spectrum = self._generate_diagnostic_spectrum(energies, kvp=120.0)
        
        # Apply detector response
        if detector_model is not None:
            spectrum = self.apply_detector_response(spectrum, detector_model)
        
        # Ensure proper alignment for SIMD
        if config.use_simd_vectorization:
            # Pad arrays to multiple of SIMD width if needed
            simd_width = config.simd_width
            pad_size = (simd_width - (config.num_energy_bins % simd_width)) % simd_width
            
            if pad_size > 0:
                logger.debug("Padding arrays by %d elements for SIMD alignment", pad_size)
                material_lut_flat = np.pad(material_lut_flat, 
                                         (0, pad_size * len(self.material_mapping)),
                                         mode='constant')
                spectrum = np.pad(spectrum, (0, pad_size), mode='constant')
                energies = np.pad(energies, (0, pad_size), mode='edge')
        
        gpu_data = {
            'material_lut': material_lut,
            'material_lut_flat': material_lut_flat,
            'spectrum': spectrum,
            'energies': energies,
            'num_materials': len(self.material_mapping)
        }
        
        # Log memory usage
        total_bytes = sum(arr.nbytes for arr in gpu_data.values() if isinstance(arr, np.ndarray))
        logger.info("GPU material data prepared: %.2f MB", total_bytes / 1024 / 1024)
        
        return gpu_data
    
    def _generate_diagnostic_spectrum(self, 
                                    energies: np.ndarray,
                                    kvp: float = 120.0) -> np.ndarray:
        """Generate a typical diagnostic X-ray spectrum.
        
        Uses the Kramers' law model with characteristic peaks and
        tungsten anode filtration.
        
        Args:
            energies: Energy bin centers in keV
            kvp: Peak kilovoltage
            
        Returns:
            Normalized photon spectrum
        """
        logger.debug("Generating diagnostic X-ray spectrum at %.0f kVp", kvp)
        
        # Kramers' law (Bremsstrahlung)
        spectrum = np.zeros_like(energies)
        valid = energies < kvp
        spectrum[valid] = (kvp - energies[valid]) / energies[valid]
        
        # Add characteristic peaks for tungsten
        # K-alpha at 59.3 keV, K-beta at 67.2 keV
        if kvp > 69.5:  # K-edge energy
            ka_idx = np.argmin(np.abs(energies - 59.3))
            kb_idx = np.argmin(np.abs(energies - 67.2))
            spectrum[ka_idx] += spectrum[ka_idx] * 0.5  # 50% boost
            spectrum[kb_idx] += spectrum[kb_idx] * 0.3  # 30% boost
        
        # Apply filtration (2.5mm Al equivalent)
        # Simplified exponential attenuation
        al_mu = 0.2  # Approximate for aluminum
        filtration = np.exp(-al_mu * 2.5 * energies / 50.0)  # Energy-dependent
        spectrum *= filtration
        
        # Normalize to unit area
        spectrum = spectrum / np.trapz(spectrum, energies)
        
        # Scale to realistic photon count (arbitrary units)
        spectrum *= 1e6
        
        return spectrum.astype(np.float32)
    
    def get_material_info(self, material_id: int) -> MaterialData:
        """Get detailed information about a specific material.
        
        Args:
            material_id: Material ID from segmentation
            
        Returns:
            MaterialData object with properties
        """
        if material_id not in self.material_mapping:
            raise ValueError(f"Unknown material ID: {material_id}")
        
        material_name = self.material_mapping[material_id]
        
        # Get density range from materials database
        density_info = self.materials_db.materials[material_name]
        density_range = (density_info.density * 0.9, density_info.density * 1.1)
        
        # Get mass attenuation coefficients for standard energy range
        energies = np.linspace(15, 150, 136)
        mass_atten_coeffs = self.materials_db.get_mass_atten_coeff(material_name, energies)
        
        return MaterialData(
            material_id=material_id,
            material_name=material_name,
            density_range=density_range,
            mass_atten_coeffs=mass_atten_coeffs
        )
    
    def validate_lut(self, lut: np.ndarray, config: RayProjectionConfig) -> bool:
        """Validate a material LUT for consistency and physical correctness.
        
        Args:
            lut: Material LUT array
            config: Ray projection configuration
            
        Returns:
            True if valid, raises exception otherwise
        """
        # Check shape
        expected_shape = (len(self.material_mapping), config.num_energy_bins)
        if lut.shape != expected_shape:
            raise ValueError(f"LUT shape {lut.shape} doesn't match expected {expected_shape}")
        
        # Check for negative values
        if np.any(lut < 0):
            raise ValueError("LUT contains negative mass attenuation coefficients")
        
        # Check for NaN or infinity
        if not np.all(np.isfinite(lut)):
            raise ValueError("LUT contains NaN or infinite values")
        
        # Check physical reasonableness
        # Mass attenuation coefficients should generally decrease with energy
        for mat_id in range(len(self.material_mapping)):
            # Check general trend (allowing for K-edges)
            start_val = np.mean(lut[mat_id, :10])
            end_val = np.mean(lut[mat_id, -10:])
            if end_val > start_val and mat_id != 0:  # Air is special case
                logger.warning("Material %d shows increasing attenuation with energy", mat_id)
        
        logger.info("Material LUT validation passed")
        return True