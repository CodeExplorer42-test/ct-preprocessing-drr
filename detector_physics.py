"""
Detector Physics Calculations.

This module implements the physics models for signal generation, gain chains,
and noise propagation in the dual-layer flat-panel detector system.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from scipy import ndimage

from detector_model import DualLayerDetectorModel


@dataclass
class DetectorGainParameters:
    """Parameters for the detector gain chain."""
    optical_photons_per_kev: float = 54.0  # CsI:Tl light yield
    optical_coupling_efficiency: float = 0.75  # Light collection efficiency
    photodiode_quantum_efficiency: float = 0.80  # a-Si:H QE at 550nm
    electronic_gain_adu_per_electron: float = 0.01  # ADC conversion
    
    @property
    def total_gain_adu_per_kev(self) -> float:
        """Calculate total system gain in ADU/keV."""
        return (self.optical_photons_per_kev * 
                self.optical_coupling_efficiency * 
                self.photodiode_quantum_efficiency * 
                self.electronic_gain_adu_per_electron)


@dataclass  
class DetectorNoiseParameters:
    """Parameters for detector noise modeling."""
    read_noise_electrons: float = 1500.0  # Electronic read noise
    swank_factor: float = 0.88  # Swank noise factor for CsI:Tl
    dark_current_electrons_per_s: float = 10.0  # Dark current
    
    @property
    def read_noise_variance_electrons2(self) -> float:
        """Read noise variance in electrons²."""
        return self.read_noise_electrons ** 2


@dataclass
class MTFParameters:
    """Modulation Transfer Function parameters."""
    gaussian_weight: float = 0.7  # Weight for Gaussian component
    gaussian_sigma_mm: float = 0.25  # Gaussian spread
    lorentzian_sigma_mm: float = 0.5  # Lorentzian spread
    
    def mtf_value(self, frequency_lpmm: float) -> float:
        """Calculate MTF value at given spatial frequency."""
        # Combined Gaussian-Lorentzian model
        gaussian_term = np.exp(-(self.gaussian_sigma_mm * frequency_lpmm)**2)
        lorentzian_term = 1.0 / (1.0 + (self.lorentzian_sigma_mm * frequency_lpmm)**2)
        
        return (self.gaussian_weight * gaussian_term + 
                (1 - self.gaussian_weight) * lorentzian_term)


class DetectorPhysics:
    """
    Physics calculations for the dual-layer detector system.
    
    This class implements the signal and noise models for converting
    incident X-ray photons into detector signals with realistic physics.
    """
    
    def __init__(self, detector_model: DualLayerDetectorModel):
        """
        Initialize detector physics calculator.
        
        Args:
            detector_model: The detector geometry model
        """
        self.detector = detector_model
        
        # Initialize gain parameters for each layer
        self.front_gain = DetectorGainParameters()
        self.rear_gain = DetectorGainParameters()
        
        # Initialize noise parameters
        self.front_noise = DetectorNoiseParameters()
        self.rear_noise = DetectorNoiseParameters()
        
        # MTF parameters
        self.front_mtf = MTFParameters()
        self.rear_mtf = MTFParameters(
            gaussian_sigma_mm=0.30,  # Slightly worse MTF for rear
            lorentzian_sigma_mm=0.60
        )
        
    def calculate_mean_signal(self, 
                            incident_fluence: np.ndarray,
                            energy_bins_kev: np.ndarray,
                            detector_layer: str,
                            spectrum_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Calculate mean signal in ADU for given incident fluence.
        
        Args:
            incident_fluence: Photon fluence per energy bin [photons/mm²/bin]
                            Shape: (height, width, n_energy_bins)
            energy_bins_kev: Energy bin centers in keV
            detector_layer: 'front' or 'rear'
            spectrum_weights: Optional pre-calculated weights including detector response
            
        Returns:
            Mean signal image in ADU, shape: (height, width)
        """
        # Get appropriate gain parameters
        gain_params = self.front_gain if detector_layer == 'front' else self.rear_gain
        
        # Calculate absorption probability for each energy
        absorption_prob = self.detector.calculate_absorption_probability(
            detector_layer, energy_bins_kev
        )
        
        # Apply spectrum hardening if this is the rear detector
        if detector_layer == 'rear':
            # Get transmission through front stack
            T_front, _ = self.detector.calculate_stack_transmission(
                energy_bins_kev, include_mask=True
            )
            # Apply to incident fluence
            incident_fluence = incident_fluence * T_front[np.newaxis, np.newaxis, :]
            
        # Calculate deposited energy per pixel
        # Shape operations to match (H, W, E) x (E,) -> (H, W)
        absorbed_photons = incident_fluence * absorption_prob[np.newaxis, np.newaxis, :]
        deposited_energy = np.sum(absorbed_photons * energy_bins_kev[np.newaxis, np.newaxis, :], axis=2)
        
        # Convert to signal in ADU
        mean_signal = deposited_energy * gain_params.total_gain_adu_per_kev
        
        return mean_signal
    
    def calculate_noise_variance(self,
                               incident_fluence: np.ndarray,
                               energy_bins_kev: np.ndarray,
                               detector_layer: str,
                               integration_time_s: float = 0.1) -> np.ndarray:
        """
        Calculate total noise variance for each pixel.
        
        Args:
            incident_fluence: Photon fluence per energy bin [photons/mm²/bin]
            energy_bins_kev: Energy bin centers in keV
            detector_layer: 'front' or 'rear'
            integration_time_s: Integration time in seconds
            
        Returns:
            Total noise variance in ADU², shape: (height, width)
        """
        # Get appropriate parameters
        gain_params = self.front_gain if detector_layer == 'front' else self.rear_gain
        noise_params = self.front_noise if detector_layer == 'front' else self.rear_noise
        
        # Calculate absorption probability
        absorption_prob = self.detector.calculate_absorption_probability(
            detector_layer, energy_bins_kev
        )
        
        # Apply spectrum hardening for rear detector
        if detector_layer == 'rear':
            T_front, _ = self.detector.calculate_stack_transmission(
                energy_bins_kev, include_mask=True
            )
            incident_fluence = incident_fluence * T_front[np.newaxis, np.newaxis, :]
            
        # Calculate quantum noise with Swank factor
        # Variance in deposited energy includes both Poisson and Swank noise
        absorbed_photons = incident_fluence * absorption_prob[np.newaxis, np.newaxis, :]
        
        # Energy-squared term for variance calculation
        energy_squared_sum = np.sum(
            absorbed_photons * energy_bins_kev[np.newaxis, np.newaxis, :]**2, 
            axis=2
        )
        
        # Quantum + Swank noise variance in keV²
        quantum_variance_kev2 = energy_squared_sum / noise_params.swank_factor
        
        # Convert to ADU²
        quantum_variance_adu2 = quantum_variance_kev2 * gain_params.total_gain_adu_per_kev**2
        
        # Electronic noise variance in ADU²
        electronic_variance_adu2 = (
            noise_params.read_noise_variance_electrons2 * 
            gain_params.electronic_gain_adu_per_electron**2
        )
        
        # Dark current noise
        dark_variance_adu2 = (
            noise_params.dark_current_electrons_per_s * integration_time_s *
            gain_params.electronic_gain_adu_per_electron**2
        )
        
        # Total variance
        total_variance = quantum_variance_adu2 + electronic_variance_adu2 + dark_variance_adu2
        
        return total_variance
    
    def apply_mtf_blur(self, image: np.ndarray, detector_layer: str,
                      pixel_size_mm: float = 0.5) -> np.ndarray:
        """
        Apply MTF-based spatial blur to simulate detector resolution.
        
        Args:
            image: Input image
            detector_layer: 'front' or 'rear'  
            pixel_size_mm: Pixel size in mm
            
        Returns:
            Blurred image
        """
        mtf_params = self.front_mtf if detector_layer == 'front' else self.rear_mtf
        
        # Create PSF kernel from MTF
        # Size kernel to capture 99% of energy
        kernel_size_mm = 6 * max(mtf_params.gaussian_sigma_mm, mtf_params.lorentzian_sigma_mm)
        kernel_size_pixels = int(np.ceil(kernel_size_mm / pixel_size_mm))
        
        # Ensure odd kernel size
        if kernel_size_pixels % 2 == 0:
            kernel_size_pixels += 1
            
        # Create coordinate grids
        x = np.arange(kernel_size_pixels) - kernel_size_pixels // 2
        y = np.arange(kernel_size_pixels) - kernel_size_pixels // 2
        xx, yy = np.meshgrid(x * pixel_size_mm, y * pixel_size_mm)
        r = np.sqrt(xx**2 + yy**2)
        
        # Create PSF as inverse Fourier transform of MTF
        # Approximate with combined Gaussian-Lorentzian
        gaussian_psf = np.exp(-r**2 / (2 * mtf_params.gaussian_sigma_mm**2))
        lorentzian_psf = 1 / (1 + (r / mtf_params.lorentzian_sigma_mm)**2)
        
        psf = (mtf_params.gaussian_weight * gaussian_psf + 
               (1 - mtf_params.gaussian_weight) * lorentzian_psf)
        
        # Normalize PSF
        psf /= np.sum(psf)
        
        # Apply convolution
        blurred = ndimage.convolve(image, psf, mode='reflect')
        
        return blurred
    
    def generate_noise_realization(self, mean_signal: np.ndarray,
                                 noise_variance: np.ndarray,
                                 seed: Optional[int] = None) -> np.ndarray:
        """
        Generate a noise realization for the detector signal.
        
        Args:
            mean_signal: Mean signal in ADU
            noise_variance: Noise variance in ADU²
            seed: Random seed for reproducibility
            
        Returns:
            Noisy signal realization in ADU
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Generate Gaussian noise with spatially-varying variance
        noise_std = np.sqrt(noise_variance)
        noise = np.random.normal(0, 1, mean_signal.shape) * noise_std
        
        # Add to mean signal
        noisy_signal = mean_signal + noise
        
        # Ensure non-negative (physical constraint)
        noisy_signal = np.maximum(noisy_signal, 0)
        
        return noisy_signal
    
    def calculate_dqe(self, spatial_frequency_lpmm: float,
                     detector_layer: str,
                     mean_exposure_kev_per_mm2: float = 1000.0) -> float:
        """
        Calculate Detective Quantum Efficiency (DQE) at given frequency.
        
        Args:
            spatial_frequency_lpmm: Spatial frequency in line pairs/mm
            detector_layer: 'front' or 'rear'
            mean_exposure_kev_per_mm2: Mean exposure level
            
        Returns:
            DQE value (0-1)
        """
        # Get MTF value
        mtf_params = self.front_mtf if detector_layer == 'front' else self.rear_mtf
        mtf = mtf_params.mtf_value(spatial_frequency_lpmm)
        
        # Get noise parameters
        noise_params = self.front_noise if detector_layer == 'front' else self.rear_noise
        
        # Approximate quantum efficiency at mean energy (70 keV)
        qe = self.detector.calculate_absorption_probability(detector_layer, 70.0)
        
        # DQE(f) = MTF(f)² * QE * Swank_factor
        dqe = mtf**2 * qe * noise_params.swank_factor
        
        return float(dqe)
    
    def get_performance_metrics(self, detector_layer: str) -> Dict[str, float]:
        """
        Get key performance metrics for a detector layer.
        
        Args:
            detector_layer: 'front' or 'rear'
            
        Returns:
            Dictionary of performance metrics
        """
        # Calculate metrics at standard conditions
        metrics = {
            'total_gain_adu_per_kev': (
                self.front_gain.total_gain_adu_per_kev if detector_layer == 'front' 
                else self.rear_gain.total_gain_adu_per_kev
            ),
            'read_noise_electrons': (
                self.front_noise.read_noise_electrons if detector_layer == 'front'
                else self.rear_noise.read_noise_electrons
            ),
            'mtf_at_nyquist': self.calculate_dqe(
                1.0 / self.detector.pixel_pitch_mm, detector_layer
            ),
            'dqe_at_zero_freq': self.calculate_dqe(0.0, detector_layer),
            'quantum_efficiency_70kev': float(
                self.detector.calculate_absorption_probability(detector_layer, 70.0)
            )
        }
        
        return metrics


def test_detector_physics():
    """Test detector physics calculations."""
    # Create detector model
    detector_model = DualLayerDetectorModel()
    physics = DetectorPhysics(detector_model)
    
    print("Detector Physics Test")
    print("=" * 60)
    
    # Test gain calculations
    print("\nGain Chain Parameters:")
    print(f"Front detector gain: {physics.front_gain.total_gain_adu_per_kev:.3f} ADU/keV")
    print(f"Rear detector gain: {physics.rear_gain.total_gain_adu_per_kev:.3f} ADU/keV")
    
    # Test MTF
    print("\nMTF at Key Frequencies:")
    frequencies = [0.5, 1.0, 1.5, 2.0]  # lp/mm
    for f in frequencies:
        mtf_front = physics.front_mtf.mtf_value(f)
        mtf_rear = physics.rear_mtf.mtf_value(f)
        print(f"  {f:.1f} lp/mm: Front={mtf_front:.3f}, Rear={mtf_rear:.3f}")
        
    # Test signal calculation
    print("\nSignal Calculation Test:")
    # Create test fluence (100x100 pixels, 5 energy bins)
    test_energies = np.array([20, 40, 60, 80, 100])
    test_fluence = np.ones((100, 100, 5)) * 1000  # 1000 photons/mm²/bin
    
    mean_signal_front = physics.calculate_mean_signal(
        test_fluence, test_energies, 'front'
    )
    mean_signal_rear = physics.calculate_mean_signal(
        test_fluence, test_energies, 'rear'
    )
    
    print(f"Mean signal (center pixel):")
    print(f"  Front: {mean_signal_front[50, 50]:.1f} ADU")
    print(f"  Rear: {mean_signal_rear[50, 50]:.1f} ADU")
    
    # Test performance metrics
    print("\nPerformance Metrics:")
    for layer in ['front', 'rear']:
        metrics = physics.get_performance_metrics(layer)
        print(f"\n{layer.capitalize()} Detector:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.3f}")


if __name__ == '__main__':
    test_detector_physics()