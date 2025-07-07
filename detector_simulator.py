"""
Detector Response Simulator.

This module orchestrates the complete detector simulation, taking polyenergetic
ray data and producing realistic detector images with proper physics modeling.
"""

import numpy as np
import nibabel as nib
from typing import Dict, Tuple, Optional, Union
from pathlib import Path
import json
import time

from detector_model import DualLayerDetectorModel
from detector_physics import DetectorPhysics
from crosstalk_mask import CrosstalkMask, CrosstalkMaskSpecification


class DetectorSimulator:
    """
    Main simulator for the dual-layer flat-panel detector system.
    
    This class integrates all detector components to simulate the complete
    signal chain from incident X-rays to final detector images.
    """
    
    def __init__(self, detector_model: Optional[DualLayerDetectorModel] = None,
                 enable_noise: bool = True,
                 enable_blur: bool = True,
                 enable_crosstalk: bool = True):
        """
        Initialize the detector simulator.
        
        Args:
            detector_model: Detector model instance (creates default if None)
            enable_noise: Whether to simulate noise
            enable_blur: Whether to apply MTF blur
            enable_crosstalk: Whether to simulate crosstalk between layers
        """
        self.detector = detector_model or DualLayerDetectorModel()
        self.physics = DetectorPhysics(self.detector)
        self.crosstalk_mask = CrosstalkMask()
        
        # Simulation options
        self.enable_noise = enable_noise
        self.enable_blur = enable_blur
        self.enable_crosstalk = enable_crosstalk
        
        # Cache for energy-dependent calculations
        self._energy_cache = {}
        
    def load_polyenergetic_data(self, data_path: Union[str, Path]) -> Dict:
        """
        Load polyenergetic data from preprocessing pipeline.
        
        Args:
            data_path: Path to polyenergetic data directory
            
        Returns:
            Dictionary with loaded data
        """
        data_path = Path(data_path)
        
        # Load polyenergetic arrays
        poly_data = np.load(data_path / 'polyenergetic_data.npz')
        
        # Load metadata
        with open(data_path / 'polyenergetic_metadata.json', 'r') as f:
            metadata = json.load(f)
            
        # Load mu volume if available
        mu_path = data_path / 'preprocessed_mu.nii.gz'
        if mu_path.exists():
            mu_volume = nib.load(str(mu_path)).get_fdata()
        else:
            mu_volume = None
            
        return {
            'spectrum': poly_data['spectrum'],
            'energies_kev': poly_data['energies'],  # Changed from 'energies_kev'
            'mu_water': poly_data.get('mu_water', None),
            'mu_volume': mu_volume,
            'metadata': metadata
        }
    
    def simulate_detector_response(self, 
                                 incident_fluence: np.ndarray,
                                 energies_kev: np.ndarray,
                                 detector_layer: str,
                                 integration_time_s: float = 0.1,
                                 add_scatter: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Simulate complete detector response for a single layer.
        
        Args:
            incident_fluence: Incident photon fluence [photons/mm²/bin]
                            Shape: (height, width, n_energy_bins)
            energies_kev: Energy bin centers in keV
            detector_layer: 'front' or 'rear'
            integration_time_s: Integration time in seconds
            add_scatter: Optional scatter signal to add
            
        Returns:
            Dictionary containing:
                - 'signal': Final detector signal in ADU
                - 'signal_clean': Noise-free signal
                - 'noise_map': Noise standard deviation map
                - 'snr_map': Signal-to-noise ratio map
        """
        # Calculate mean signal
        mean_signal = self.physics.calculate_mean_signal(
            incident_fluence, energies_kev, detector_layer
        )
        
        # Add scatter if provided
        if add_scatter is not None:
            mean_signal += add_scatter
            
        # Apply MTF blur if enabled
        if self.enable_blur:
            blurred_signal = self.physics.apply_mtf_blur(
                mean_signal, detector_layer, self.detector.pixel_pitch_mm
            )
        else:
            blurred_signal = mean_signal.copy()
            
        # Calculate noise variance
        noise_variance = self.physics.calculate_noise_variance(
            incident_fluence, energies_kev, detector_layer, integration_time_s
        )
        
        # Generate noisy realization if enabled
        if self.enable_noise:
            final_signal = self.physics.generate_noise_realization(
                blurred_signal, noise_variance
            )
        else:
            final_signal = blurred_signal
            
        # Calculate SNR map
        noise_std = np.sqrt(noise_variance)
        snr_map = np.divide(blurred_signal, noise_std, 
                           out=np.zeros_like(blurred_signal),
                           where=noise_std > 0)
        
        return {
            'signal': final_signal,
            'signal_clean': blurred_signal,
            'noise_map': noise_std,
            'snr_map': snr_map
        }
    
    def simulate_crosstalk(self, front_signal: np.ndarray, 
                         rear_signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Simulate crosstalk between detector layers.
        
        Args:
            front_signal: Front detector signal
            rear_signal: Rear detector signal
            
        Returns:
            Tuple of (front_with_crosstalk, rear_with_crosstalk)
        """
        if not self.enable_crosstalk:
            return front_signal.copy(), rear_signal.copy()
            
        # Get crosstalk coefficients
        k_front_to_rear = self.crosstalk_mask.get_crosstalk_coefficient()
        k_rear_to_front = self.crosstalk_mask.get_crosstalk_coefficient()
        
        # Apply crosstalk (ghost images)
        # The ghosts are blurred versions due to different MTF
        ghost_on_front = self.physics.apply_mtf_blur(
            rear_signal * k_rear_to_front, 'front'
        )
        ghost_on_rear = self.physics.apply_mtf_blur(
            front_signal * k_front_to_rear, 'rear'
        )
        
        # Add ghosts to primary signals
        front_with_crosstalk = front_signal + ghost_on_front
        rear_with_crosstalk = rear_signal + ghost_on_rear
        
        return front_with_crosstalk, rear_with_crosstalk
    
    def simulate_stereo_pair(self, 
                           left_fluence: np.ndarray,
                           right_fluence: np.ndarray,
                           energies_kev: np.ndarray,
                           integration_time_s: float = 0.1) -> Dict[str, np.ndarray]:
        """
        Simulate complete stereo detector pair with crosstalk.
        
        Args:
            left_fluence: Incident fluence from left source
            right_fluence: Incident fluence from right source
            energies_kev: Energy bin centers
            integration_time_s: Integration time
            
        Returns:
            Dictionary with 'left' and 'right' detector images
        """
        # Left source primarily illuminates front detector
        left_response = self.simulate_detector_response(
            left_fluence, energies_kev, 'front', integration_time_s
        )
        
        # Right source primarily illuminates rear detector  
        right_response = self.simulate_detector_response(
            right_fluence, energies_kev, 'rear', integration_time_s
        )
        
        # Apply crosstalk correction if enabled
        if self.enable_crosstalk:
            left_final, right_final = self.simulate_crosstalk(
                left_response['signal'], right_response['signal']
            )
        else:
            left_final = left_response['signal']
            right_final = right_response['signal']
            
        return {
            'left': left_final,
            'right': right_final,
            'left_clean': left_response['signal_clean'],
            'right_clean': right_response['signal_clean'],
            'left_snr': left_response['snr_map'],
            'right_snr': right_response['snr_map']
        }
    
    def create_test_fluence(self, shape: Tuple[int, int], 
                          energies_kev: np.ndarray,
                          mean_fluence: float = 1000.0) -> np.ndarray:
        """
        Create test fluence pattern for validation.
        
        Args:
            shape: Detector dimensions (height, width)
            energies_kev: Energy array
            mean_fluence: Mean photon fluence per bin
            
        Returns:
            Test fluence array
        """
        # Create gradient pattern
        h, w = shape
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        
        # Radial gradient from center
        r = np.sqrt((xx - 0.5)**2 + (yy - 0.5)**2)
        intensity = 1.0 - 0.5 * r
        
        # Create energy-dependent fluence
        fluence = np.zeros((h, w, len(energies_kev)))
        for i, E in enumerate(energies_kev):
            # Higher energies have relatively more fluence after hardening
            energy_weight = np.exp(-0.01 * (100 - E))
            fluence[:, :, i] = intensity * mean_fluence * energy_weight
            
        return fluence
    
    def save_results(self, results: Dict[str, np.ndarray], 
                    output_path: Union[str, Path]):
        """
        Save simulation results to disk.
        
        Args:
            results: Dictionary of result arrays
            output_path: Output directory path
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save detector images
        for key, data in results.items():
            if isinstance(data, np.ndarray):
                # Save as NIFTI for consistency with preprocessing
                img = nib.Nifti1Image(data.astype(np.float32), affine=np.eye(4))
                nib.save(img, str(output_path / f'detector_{key}.nii.gz'))
                
        # Save metadata
        metadata = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'detector_config': {
                'pixel_pitch_mm': self.detector.pixel_pitch_mm,
                'detector_size': self.detector.detector_size_pixels,
                'front_scintillator_mm': self.detector.front_detector.scintillator_thickness_mm,
                'rear_scintillator_mm': self.detector.rear_detector.scintillator_thickness_mm
            },
            'simulation_settings': {
                'noise_enabled': self.enable_noise,
                'blur_enabled': self.enable_blur,
                'crosstalk_enabled': self.enable_crosstalk
            }
        }
        
        with open(output_path / 'detector_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)


def test_detector_simulator():
    """Test the detector simulator with synthetic data."""
    print("Detector Simulator Test")
    print("=" * 60)
    
    # Create simulator
    simulator = DetectorSimulator(enable_noise=True, enable_blur=True)
    
    # Create test energy array
    energies = np.linspace(20, 120, 20)  # 20 energy bins
    
    # Create test fluence patterns
    shape = (100, 100)  # Small test size
    left_fluence = simulator.create_test_fluence(shape, energies, 1000)
    right_fluence = simulator.create_test_fluence(shape, energies, 800)
    
    print("\nSimulating stereo detector pair...")
    t0 = time.time()
    
    # Simulate stereo pair
    results = simulator.simulate_stereo_pair(
        left_fluence, right_fluence, energies
    )
    
    dt = time.time() - t0
    print(f"Simulation completed in {dt:.2f} seconds")
    
    # Print statistics
    print("\nDetector Signal Statistics:")
    for side in ['left', 'right']:
        signal = results[side]
        clean = results[f'{side}_clean']
        snr = results[f'{side}_snr']
        
        print(f"\n{side.capitalize()} Detector:")
        print(f"  Signal range: [{signal.min():.1f}, {signal.max():.1f}] ADU")
        print(f"  Clean signal mean: {clean.mean():.1f} ADU")
        print(f"  Noise (std): {(signal - clean).std():.1f} ADU")
        print(f"  Mean SNR: {snr.mean():.1f}")
        print(f"  SNR range: [{snr.min():.1f}, {snr.max():.1f}]")
        
    # Test crosstalk levels
    if simulator.enable_crosstalk:
        # Estimate crosstalk by zeroing one source
        zero_fluence = np.zeros_like(left_fluence)
        crosstalk_test = simulator.simulate_stereo_pair(
            left_fluence, zero_fluence, energies
        )
        
        crosstalk_fraction = crosstalk_test['right'].mean() / results['left'].mean()
        print(f"\nCrosstalk level: {crosstalk_fraction*100:.3f}%")


if __name__ == '__main__':
    test_detector_simulator()