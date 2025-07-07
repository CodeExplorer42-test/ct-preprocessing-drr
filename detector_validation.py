"""
Detector Model Validation Framework.

This module provides validation protocols to verify the accuracy and performance
of the dual-layer detector simulation against theoretical predictions and
empirical measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json
from scipy import ndimage
from scipy.optimize import curve_fit

from detector_model import DualLayerDetectorModel
from detector_physics import DetectorPhysics
from detector_simulator import DetectorSimulator
from materials_db import get_materials_db


class DetectorValidator:
    """
    Validation framework for the detector model.
    
    Provides methods to test various aspects of the detector simulation
    including energy response, MTF, quantum efficiency, and crosstalk.
    """
    
    def __init__(self, simulator: Optional[DetectorSimulator] = None):
        """
        Initialize the validator.
        
        Args:
            simulator: Detector simulator instance (creates default if None)
        """
        self.simulator = simulator or DetectorSimulator()
        self.detector = self.simulator.detector
        self.physics = self.simulator.physics
        self.materials_db = get_materials_db()
        
        # Storage for validation results
        self.results = {}
        
    def validate_energy_response(self, save_plots: bool = False) -> Dict[str, float]:
        """
        Validate detector energy response using copper step wedge.
        
        Returns:
            Dictionary of validation metrics
        """
        print("Validating Energy Response...")
        
        # Define copper thicknesses (mm)
        cu_thicknesses = np.array([0, 1, 2, 5, 10, 20])
        
        # Energy array for polyenergetic calculation
        energies = np.linspace(20, 120, 101)
        
        # Assume 120 kVp tungsten spectrum (simplified)
        spectrum = self._generate_tungsten_spectrum(energies, 120)
        
        # Calculate signal for each thickness
        front_signals = []
        rear_signals = []
        
        for thickness_mm in cu_thicknesses:
            # Calculate beam attenuation through copper
            if thickness_mm > 0:
                cu_transmission = self.materials_db.get_transmission(
                    'aluminum', thickness_mm, energies  # Using Al as proxy for Cu
                ) 
            else:
                cu_transmission = np.ones_like(energies)
                
            # Create attenuated fluence
            fluence = spectrum * cu_transmission
            fluence_3d = fluence[np.newaxis, np.newaxis, :] * 1e6  # photons/mm²
            
            # Simulate detector response
            front_response = self.physics.calculate_mean_signal(
                fluence_3d, energies, 'front'
            )
            rear_response = self.physics.calculate_mean_signal(
                fluence_3d, energies, 'rear'
            )
            
            front_signals.append(float(front_response[0, 0]))
            rear_signals.append(float(rear_response[0, 0]))
            
        # Convert to arrays
        front_signals = np.array(front_signals)
        rear_signals = np.array(rear_signals)
        
        # Fit exponential decay
        def exp_decay(x, a, b, c):
            return a * np.exp(-b * x) + c
            
        # Fit front detector
        popt_front, _ = curve_fit(exp_decay, cu_thicknesses[1:], front_signals[1:],
                                p0=[front_signals[0], 0.1, 0])
        
        # Fit rear detector  
        popt_rear, _ = curve_fit(exp_decay, cu_thicknesses[1:], rear_signals[1:],
                               p0=[rear_signals[0], 0.1, 0])
        
        # Calculate R-squared
        front_fit = exp_decay(cu_thicknesses, *popt_front)
        rear_fit = exp_decay(cu_thicknesses, *popt_rear)
        
        r2_front = 1 - np.sum((front_signals - front_fit)**2) / np.sum((front_signals - front_signals.mean())**2)
        r2_rear = 1 - np.sum((rear_signals - rear_fit)**2) / np.sum((rear_signals - rear_signals.mean())**2)
        
        # Store results
        results = {
            'front_r2': r2_front,
            'rear_r2': r2_rear,
            'front_decay_constant': popt_front[1],
            'rear_decay_constant': popt_rear[1],
            'front_signals': front_signals.tolist(),
            'rear_signals': rear_signals.tolist()
        }
        
        if save_plots:
            self._plot_energy_response(cu_thicknesses, front_signals, rear_signals,
                                     front_fit, rear_fit)
            
        self.results['energy_response'] = results
        return results
    
    def validate_mtf(self, save_plots: bool = False) -> Dict[str, float]:
        """
        Validate MTF using slanted edge method.
        
        Returns:
            Dictionary of MTF values at key frequencies
        """
        print("Validating MTF...")
        
        # Create slanted edge phantom
        size = 200
        edge_angle_deg = 5  # 5 degree slant
        
        phantom = self._create_slanted_edge_phantom(size, edge_angle_deg)
        
        # Create monoenergetic fluence at 70 keV
        energy = np.array([70.0])
        fluence = phantom[:, :, np.newaxis] * 1e6  # photons/mm²
        
        # Simulate detector response without noise
        simulator_no_noise = DetectorSimulator(
            self.detector, enable_noise=False, enable_blur=True
        )
        
        # Get blurred images
        front_response = simulator_no_noise.simulate_detector_response(
            fluence, energy, 'front'
        )
        rear_response = simulator_no_noise.simulate_detector_response(
            fluence, energy, 'rear'
        )
        
        # Calculate MTF from edge spread function
        front_mtf_freq, front_mtf_values = self._calculate_mtf_from_edge(
            front_response['signal']
        )
        rear_mtf_freq, rear_mtf_values = self._calculate_mtf_from_edge(
            rear_response['signal']
        )
        
        # Sample at key frequencies
        key_frequencies = [0.5, 1.0, 1.5, 2.0]  # lp/mm
        results = {}
        
        for freq in key_frequencies:
            # Interpolate MTF values
            front_mtf = np.interp(freq, front_mtf_freq, front_mtf_values)
            rear_mtf = np.interp(freq, rear_mtf_freq, rear_mtf_values)
            
            # Compare with model
            front_model = self.physics.front_mtf.mtf_value(freq)
            rear_model = self.physics.rear_mtf.mtf_value(freq)
            
            results[f'front_mtf_{freq}lpmm'] = float(front_mtf)
            results[f'rear_mtf_{freq}lpmm'] = float(rear_mtf)
            results[f'front_model_{freq}lpmm'] = float(front_model)
            results[f'rear_model_{freq}lpmm'] = float(rear_model)
            
        if save_plots:
            self._plot_mtf_curves(front_mtf_freq, front_mtf_values,
                                rear_mtf_freq, rear_mtf_values)
            
        self.results['mtf'] = results
        return results
    
    def validate_quantum_efficiency(self) -> Dict[str, float]:
        """
        Validate quantum efficiency calculations.
        
        Returns:
            Dictionary of QE values at different energies
        """
        print("Validating Quantum Efficiency...")
        
        # Test energies
        test_energies = np.array([20, 40, 60, 80, 100, 120])
        
        results = {}
        
        for energy in test_energies:
            # Calculate QE for both detectors
            qe_front = self.detector.calculate_absorption_probability('front', energy)
            qe_rear = self.detector.calculate_absorption_probability('rear', energy)
            
            # Store results
            results[f'qe_front_{energy}keV'] = float(qe_front)
            results[f'qe_rear_{energy}keV'] = float(qe_rear)
            
        # Calculate mean QE
        results['mean_qe_front'] = np.mean([v for k, v in results.items() if 'qe_front' in k])
        results['mean_qe_rear'] = np.mean([v for k, v in results.items() if 'qe_rear' in k])
        
        self.results['quantum_efficiency'] = results
        return results
    
    def validate_crosstalk_suppression(self) -> Dict[str, float]:
        """
        Validate crosstalk suppression effectiveness.
        
        Returns:
            Dictionary of crosstalk metrics
        """
        print("Validating Crosstalk Suppression...")
        
        # Create test pattern - uniform field
        size = (100, 100)
        energies = np.linspace(20, 120, 20)
        
        # Create fluence from one source only
        left_fluence = np.ones((*size, len(energies))) * 1000  # photons/mm²
        right_fluence = np.zeros_like(left_fluence)
        
        # Simulate with crosstalk
        with_crosstalk = self.simulator.simulate_stereo_pair(
            left_fluence, right_fluence, energies
        )
        
        # Simulate without crosstalk
        simulator_no_ct = DetectorSimulator(
            self.detector, enable_crosstalk=False
        )
        without_crosstalk = simulator_no_ct.simulate_stereo_pair(
            left_fluence, right_fluence, energies
        )
        
        # Calculate crosstalk level
        primary_signal = with_crosstalk['left'].mean()
        crosstalk_signal = with_crosstalk['right'].mean()
        background_signal = without_crosstalk['right'].mean()
        
        crosstalk_fraction = (crosstalk_signal - background_signal) / primary_signal
        suppression_factor = 1.0 - crosstalk_fraction
        
        results = {
            'crosstalk_fraction': float(crosstalk_fraction),
            'suppression_factor': float(suppression_factor),
            'suppression_percent': float(suppression_factor * 100),
            'primary_signal_adu': float(primary_signal),
            'crosstalk_signal_adu': float(crosstalk_signal)
        }
        
        self.results['crosstalk'] = results
        return results
    
    def validate_noise_model(self) -> Dict[str, float]:
        """
        Validate noise model predictions.
        
        Returns:
            Dictionary of noise metrics
        """
        print("Validating Noise Model...")
        
        # Create uniform fluence at different levels
        size = (100, 100)
        energies = np.array([70.0])  # Single energy for simplicity
        
        fluence_levels = np.logspace(2, 5, 10)  # 100 to 100k photons/mm²
        
        measured_noise_front = []
        predicted_noise_front = []
        measured_noise_rear = []
        predicted_noise_rear = []
        
        for fluence_level in fluence_levels:
            # Create uniform fluence
            fluence = np.ones((*size, 1)) * fluence_level
            
            # Multiple realizations for noise measurement
            n_realizations = 10
            front_signals = []
            rear_signals = []
            
            for i in range(n_realizations):
                response = self.simulator.simulate_stereo_pair(
                    fluence, fluence, energies, integration_time_s=0.1
                )
                front_signals.append(response['left'])
                rear_signals.append(response['right'])
                
            # Calculate measured noise (std across realizations)
            front_signals = np.array(front_signals)
            rear_signals = np.array(rear_signals)
            
            measured_noise_front.append(np.std(front_signals))
            measured_noise_rear.append(np.std(rear_signals))
            
            # Calculate predicted noise
            noise_var_front = self.physics.calculate_noise_variance(
                fluence, energies, 'front', 0.1
            )
            noise_var_rear = self.physics.calculate_noise_variance(
                fluence, energies, 'rear', 0.1
            )
            
            predicted_noise_front.append(np.sqrt(noise_var_front.mean()))
            predicted_noise_rear.append(np.sqrt(noise_var_rear.mean()))
            
        # Convert to arrays
        measured_noise_front = np.array(measured_noise_front)
        predicted_noise_front = np.array(predicted_noise_front)
        measured_noise_rear = np.array(measured_noise_rear)
        predicted_noise_rear = np.array(predicted_noise_rear)
        
        # Calculate correlation
        corr_front = np.corrcoef(measured_noise_front, predicted_noise_front)[0, 1]
        corr_rear = np.corrcoef(measured_noise_rear, predicted_noise_rear)[0, 1]
        
        results = {
            'noise_correlation_front': float(corr_front),
            'noise_correlation_rear': float(corr_rear),
            'mean_noise_ratio_front': float(measured_noise_front.mean() / predicted_noise_front.mean()),
            'mean_noise_ratio_rear': float(measured_noise_rear.mean() / predicted_noise_rear.mean())
        }
        
        self.results['noise_model'] = results
        return results
    
    def run_full_validation(self, save_report: bool = True,
                          output_dir: Optional[Path] = None) -> Dict:
        """
        Run complete validation suite.
        
        Args:
            save_report: Whether to save validation report
            output_dir: Directory for saving results
            
        Returns:
            Complete validation results
        """
        print("\nRunning Full Detector Validation Suite")
        print("=" * 60)
        
        # Run all validations
        self.validate_energy_response()
        self.validate_mtf()
        self.validate_quantum_efficiency()
        self.validate_crosstalk_suppression()
        self.validate_noise_model()
        
        # Generate summary
        summary = self._generate_validation_summary()
        
        if save_report:
            if output_dir is None:
                output_dir = Path('detector_validation_results')
            output_dir.mkdir(exist_ok=True)
            
            # Save detailed results
            with open(output_dir / 'validation_results.json', 'w') as f:
                json.dump(self.results, f, indent=2)
                
            # Save summary
            with open(output_dir / 'validation_summary.txt', 'w') as f:
                f.write(summary)
                
            print(f"\nValidation results saved to {output_dir}")
            
        return self.results
    
    def _generate_tungsten_spectrum(self, energies: np.ndarray, 
                                  kvp: float) -> np.ndarray:
        """Generate simplified tungsten X-ray spectrum."""
        # Simplified spectrum model
        E_max = kvp
        spectrum = np.zeros_like(energies)
        
        mask = energies <= E_max
        spectrum[mask] = energies[mask] * (E_max - energies[mask])
        
        # Add characteristic peaks
        char_energies = [59.3, 67.2, 69.1]  # W K-alpha, K-beta
        for E_char in char_energies:
            if E_char < E_max:
                idx = np.argmin(np.abs(energies - E_char))
                spectrum[idx] *= 2.0
                
        # Normalize
        spectrum /= spectrum.sum()
        
        return spectrum
    
    def _create_slanted_edge_phantom(self, size: int, 
                                   angle_deg: float) -> np.ndarray:
        """Create slanted edge phantom for MTF measurement."""
        # Create coordinate grid
        x = np.arange(size) - size/2
        y = np.arange(size) - size/2
        xx, yy = np.meshgrid(x, y)
        
        # Rotate coordinates
        angle_rad = np.deg2rad(angle_deg)
        x_rot = xx * np.cos(angle_rad) + yy * np.sin(angle_rad)
        
        # Create edge
        phantom = np.zeros((size, size))
        phantom[x_rot > 0] = 1.0
        
        # Smooth slightly to avoid aliasing
        phantom = ndimage.gaussian_filter(phantom, 0.5)
        
        return phantom
    
    def _calculate_mtf_from_edge(self, edge_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate MTF from edge spread function."""
        # Extract edge profile (simplified - assumes horizontal edge)
        mid_row = edge_image.shape[0] // 2
        profile = edge_image[mid_row, :]
        
        # Differentiate to get line spread function
        lsf = np.gradient(profile)
        
        # FFT to get MTF
        mtf = np.abs(np.fft.fft(lsf))
        mtf = mtf[:len(mtf)//2]  # Take positive frequencies
        mtf /= mtf[0]  # Normalize
        
        # Frequency axis (assuming 0.5mm pixels)
        freq = np.fft.fftfreq(len(lsf), 0.5)[:len(mtf)]
        
        return freq, mtf
    
    def _generate_validation_summary(self) -> str:
        """Generate human-readable validation summary."""
        summary = ["Detector Validation Summary", "=" * 40, ""]
        
        # Energy response
        if 'energy_response' in self.results:
            er = self.results['energy_response']
            summary.append("Energy Response:")
            summary.append(f"  Front R²: {er['front_r2']:.3f}")
            summary.append(f"  Rear R²: {er['rear_r2']:.3f}")
            summary.append("")
            
        # MTF
        if 'mtf' in self.results:
            mtf = self.results['mtf']
            summary.append("MTF at Nyquist (1.0 lp/mm):")
            summary.append(f"  Front: {mtf.get('front_mtf_1.0lpmm', 0):.3f}")
            summary.append(f"  Rear: {mtf.get('rear_mtf_1.0lpmm', 0):.3f}")
            summary.append("")
            
        # Quantum efficiency
        if 'quantum_efficiency' in self.results:
            qe = self.results['quantum_efficiency']
            summary.append("Mean Quantum Efficiency:")
            summary.append(f"  Front: {qe['mean_qe_front']*100:.1f}%")
            summary.append(f"  Rear: {qe['mean_qe_rear']*100:.1f}%")
            summary.append("")
            
        # Crosstalk
        if 'crosstalk' in self.results:
            ct = self.results['crosstalk']
            summary.append("Crosstalk Suppression:")
            summary.append(f"  Suppression: {ct['suppression_percent']:.1f}%")
            summary.append(f"  Residual crosstalk: {ct['crosstalk_fraction']*100:.3f}%")
            summary.append("")
            
        # Noise model
        if 'noise_model' in self.results:
            nm = self.results['noise_model']
            summary.append("Noise Model:")
            summary.append(f"  Front correlation: {nm['noise_correlation_front']:.3f}")
            summary.append(f"  Rear correlation: {nm['noise_correlation_rear']:.3f}")
            
        return "\n".join(summary)


def test_validation():
    """Run validation tests."""
    validator = DetectorValidator()
    
    # Run individual tests
    print("\n1. Energy Response Validation")
    energy_results = validator.validate_energy_response()
    print(f"   Front R²: {energy_results['front_r2']:.3f}")
    print(f"   Rear R²: {energy_results['rear_r2']:.3f}")
    
    print("\n2. Quantum Efficiency Validation")
    qe_results = validator.validate_quantum_efficiency()
    print(f"   Mean QE Front: {qe_results['mean_qe_front']*100:.1f}%")
    print(f"   Mean QE Rear: {qe_results['mean_qe_rear']*100:.1f}%")
    
    print("\n3. Crosstalk Validation")
    ct_results = validator.validate_crosstalk_suppression()
    print(f"   Crosstalk suppression: {ct_results['suppression_percent']:.1f}%")


if __name__ == '__main__':
    test_validation()