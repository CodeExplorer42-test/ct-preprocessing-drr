"""
Example script demonstrating detector model usage.

This script shows how to:
1. Load polyenergetic data from preprocessing
2. Simulate detector response
3. Visualize and save results
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

from detector_model import DualLayerDetectorModel
from detector_simulator import DetectorSimulator
from detector_validation import DetectorValidator


def main():
    """Run example detector simulation."""
    print("Detector Model Example")
    print("=" * 60)
    
    # 1. Initialize detector system
    print("\n1. Initializing detector system...")
    detector_model = DualLayerDetectorModel()
    detector_model.print_configuration()
    
    # 2. Create detector simulator
    print("\n2. Creating detector simulator...")
    simulator = DetectorSimulator(
        detector_model=detector_model,
        enable_noise=True,
        enable_blur=True,
        enable_crosstalk=True
    )
    
    # 3. Load polyenergetic data from preprocessing
    poly_data_path = Path("test_output_polyenergetic")
    if poly_data_path.exists():
        print(f"\n3. Loading polyenergetic data from {poly_data_path}...")
        poly_data = simulator.load_polyenergetic_data(poly_data_path)
        
        print(f"   Loaded spectrum with {len(poly_data['energies_kev'])} energy bins")
        print(f"   Energy range: {poly_data['energies_kev'][0]:.1f} - {poly_data['energies_kev'][-1]:.1f} keV")
        
        # Use actual spectrum and energies
        energies = poly_data['energies_kev']
        spectrum = poly_data['spectrum']
    else:
        print("\n3. Polyenergetic data not found, using synthetic data...")
        # Create synthetic energy array
        energies = np.linspace(15, 150, 136)
        
        # Create synthetic tungsten spectrum
        spectrum = create_tungsten_spectrum(energies, kvp=120)
    
    # 4. Create test fluence patterns (simulating after patient attenuation)
    print("\n4. Creating test fluence patterns...")
    detector_size = (100, 100)  # Small size for quick demo
    
    # Simulate fluence for left and right sources
    # In real use, this would come from ray tracing through patient
    left_fluence = create_test_fluence_pattern(
        detector_size, energies, spectrum, 
        mean_attenuation=0.3, pattern='gradient'
    )
    
    right_fluence = create_test_fluence_pattern(
        detector_size, energies, spectrum,
        mean_attenuation=0.35, pattern='circular'
    )
    
    # 5. Simulate detector response
    print("\n5. Simulating stereo detector pair...")
    t_start = time.time()
    
    results = simulator.simulate_stereo_pair(
        left_fluence, right_fluence, energies,
        integration_time_s=0.1
    )
    
    t_elapsed = time.time() - t_start
    print(f"   Simulation completed in {t_elapsed:.2f} seconds")
    
    # 6. Display results
    print("\n6. Detector Response Statistics:")
    print_detector_statistics(results)
    
    # 7. Visualize results
    print("\n7. Creating visualization...")
    visualize_detector_response(results)
    
    # 8. Save results
    output_dir = Path("detector_example_output")
    print(f"\n8. Saving results to {output_dir}...")
    simulator.save_results(results, output_dir)
    
    # 9. Run quick validation
    print("\n9. Running validation checks...")
    validator = DetectorValidator(simulator)
    
    # Run specific validations
    qe_results = validator.validate_quantum_efficiency()
    print(f"   Mean QE Front: {qe_results['mean_qe_front']*100:.1f}%")
    print(f"   Mean QE Rear: {qe_results['mean_qe_rear']*100:.1f}%")
    
    ct_results = validator.validate_crosstalk_suppression()
    print(f"   Crosstalk suppression: {ct_results['suppression_percent']:.1f}%")
    
    print("\nExample completed successfully!")


def create_tungsten_spectrum(energies: np.ndarray, kvp: float) -> np.ndarray:
    """Create simplified tungsten X-ray spectrum."""
    # Bremsstrahlung component
    spectrum = np.zeros_like(energies)
    mask = energies <= kvp
    spectrum[mask] = energies[mask] * (kvp - energies[mask])
    
    # Add characteristic peaks
    char_peaks = {
        59.3: 1.0,   # W K-alpha1
        67.2: 0.6,   # W K-alpha2
        69.1: 0.3    # W K-beta
    }
    
    for peak_energy, relative_intensity in char_peaks.items():
        if peak_energy < kvp:
            idx = np.argmin(np.abs(energies - peak_energy))
            spectrum[idx] += spectrum.max() * relative_intensity
            
    # Apply filtration (2.5mm Al equivalent)
    al_transmission = np.exp(-0.25 * energies**(-2.8))  # Empirical
    spectrum *= al_transmission
    
    # Normalize
    spectrum /= spectrum.sum()
    
    return spectrum


def create_test_fluence_pattern(shape: tuple, energies: np.ndarray,
                               spectrum: np.ndarray, mean_attenuation: float,
                               pattern: str = 'uniform') -> np.ndarray:
    """Create test fluence pattern with different geometries."""
    h, w = shape
    n_energies = len(energies)
    
    # Base fluence (photons/mm²/keV)
    base_fluence = 1e6  # Typical after patient
    
    # Create spatial pattern
    if pattern == 'uniform':
        spatial = np.ones((h, w))
    elif pattern == 'gradient':
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        xx, yy = np.meshgrid(x, y)
        spatial = 0.5 + 0.5 * xx  # Left to right gradient
    elif pattern == 'circular':
        x = np.linspace(-1, 1, w)
        y = np.linspace(-1, 1, h)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        spatial = np.exp(-2 * r**2)  # Gaussian blob
    else:
        spatial = np.ones((h, w))
        
    # Apply mean attenuation
    spatial *= (1 - mean_attenuation)
    
    # Create energy-dependent fluence
    fluence = np.zeros((h, w, n_energies))
    for i in range(n_energies):
        fluence[:, :, i] = spatial * spectrum[i] * base_fluence
        
    return fluence


def print_detector_statistics(results: dict):
    """Print detector response statistics."""
    for side in ['left', 'right']:
        signal = results[side]
        clean = results[f'{side}_clean']
        snr = results[f'{side}_snr']
        
        print(f"\n{side.capitalize()} Detector:")
        print(f"  Signal range: [{signal.min():.1f}, {signal.max():.1f}] ADU")
        print(f"  Mean signal: {signal.mean():.1f} ± {signal.std():.1f} ADU")
        print(f"  SNR: {snr.mean():.1f} (range: [{snr.min():.1f}, {snr.max():.1f}])")
        
        # Estimate noise
        if clean.shape == signal.shape:
            noise_estimate = np.std(signal - clean)
            print(f"  Estimated noise: {noise_estimate:.1f} ADU")


def visualize_detector_response(results: dict):
    """Create visualization of detector response."""
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Left detector
    im1 = axes[0, 0].imshow(results['left'], cmap='gray')
    axes[0, 0].set_title('Left Detector Signal')
    plt.colorbar(im1, ax=axes[0, 0], label='ADU')
    
    im2 = axes[0, 1].imshow(results['left_clean'], cmap='gray')
    axes[0, 1].set_title('Left Clean Signal')
    plt.colorbar(im2, ax=axes[0, 1], label='ADU')
    
    im3 = axes[0, 2].imshow(results['left_snr'], cmap='viridis')
    axes[0, 2].set_title('Left SNR Map')
    plt.colorbar(im3, ax=axes[0, 2], label='SNR')
    
    # Right detector
    im4 = axes[1, 0].imshow(results['right'], cmap='gray')
    axes[1, 0].set_title('Right Detector Signal')
    plt.colorbar(im4, ax=axes[1, 0], label='ADU')
    
    im5 = axes[1, 1].imshow(results['right_clean'], cmap='gray')
    axes[1, 1].set_title('Right Clean Signal')
    plt.colorbar(im5, ax=axes[1, 1], label='ADU')
    
    im6 = axes[1, 2].imshow(results['right_snr'], cmap='viridis')
    axes[1, 2].set_title('Right SNR Map')
    plt.colorbar(im6, ax=axes[1, 2], label='SNR')
    
    # Remove axis labels for cleaner look
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        
    plt.tight_layout()
    plt.savefig('detector_response_visualization.png', dpi=150)
    print("   Saved visualization to detector_response_visualization.png")
    plt.close()


if __name__ == '__main__':
    main()