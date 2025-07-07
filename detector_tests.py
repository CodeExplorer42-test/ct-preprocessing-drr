"""
Unit tests for detector model components.

Run with: pytest test_detector_components.py -v
"""

import pytest
import numpy as np
from pathlib import Path

from materials_db import MaterialsDatabase, get_materials_db
from detector_model import DualLayerDetectorModel, DetectorLayer, LayerType
from detector_physics import DetectorPhysics, DetectorGainParameters
from crosstalk_mask import CrosstalkMask, CrosstalkMaskSpecification
from detector_simulator import DetectorSimulator


class TestMaterialsDatabase:
    """Test materials database functionality."""
    
    def test_materials_initialization(self):
        """Test that materials database initializes correctly."""
        db = MaterialsDatabase()
        
        # Check that all required materials are present
        required_materials = ['aluminum', 'csi_tl', 'tungsten', 'air']
        for material in required_materials:
            assert material in db.materials
            
    def test_mass_atten_coefficients(self):
        """Test mass attenuation coefficient retrieval."""
        db = get_materials_db()
        
        # Test specific energy
        mu_rho = db.get_mass_atten_coeff('aluminum', 60.0)
        assert isinstance(mu_rho, (float, np.floating))
        assert mu_rho > 0
        
        # Test full array
        mu_rho_array = db.get_mass_atten_coeff('tungsten')
        assert len(mu_rho_array) == 136  # 15-150 keV in 1 keV steps
        
    def test_transmission_calculation(self):
        """Test transmission calculations."""
        db = get_materials_db()
        
        # Test tungsten transmission at 80 keV through 1mm
        T = db.get_transmission('tungsten', 1.0, 80.0)
        assert 0 < T < 1
        assert T < 0.1  # Tungsten should be very absorbing
        
    def test_absorption_probability(self):
        """Test absorption probability calculations."""
        db = get_materials_db()
        
        # Test CsI absorption at 60 keV through 0.5mm
        A = db.get_absorption_probability('csi_tl', 0.5, 60.0)
        assert 0 < A < 1
        assert A > 0.5  # CsI should absorb >50% at this energy/thickness


class TestDetectorModel:
    """Test detector model structure."""
    
    def test_detector_initialization(self):
        """Test detector model initialization."""
        detector = DualLayerDetectorModel()
        
        # Check basic properties
        assert detector.pixel_pitch_mm == 0.5
        assert detector.detector_size_pixels == (720, 720)
        
        # Check front detector
        assert detector.front_detector.scintillator_thickness_mm == 0.5
        assert len(detector.front_detector.layers) == 4
        
        # Check rear detector  
        assert detector.rear_detector.scintillator_thickness_mm == 0.6
        assert len(detector.rear_detector.layers) == 3
        
    def test_layer_specifications(self):
        """Test individual layer specifications."""
        detector = DualLayerDetectorModel()
        
        # Check front entrance window
        entrance_window = detector.front_detector.layers[0]
        assert entrance_window.layer_type == LayerType.ENTRANCE_WINDOW
        assert entrance_window.material == 'aluminum'
        assert entrance_window.thickness_mm == 0.5
        
    def test_stack_transmission(self):
        """Test transmission calculations through detector stack."""
        detector = DualLayerDetectorModel()
        
        energies = np.array([20, 60, 100])
        T_front, T_total = detector.calculate_stack_transmission(energies)
        
        # Check shapes
        assert T_front.shape == energies.shape
        assert T_total.shape == energies.shape
        
        # Check physical constraints
        assert np.all(T_front > 0) and np.all(T_front < 1)
        assert np.all(T_total < T_front)  # Total should be less than front only
        
    def test_absorption_probability(self):
        """Test scintillator absorption probability."""
        detector = DualLayerDetectorModel()
        
        # Test at 70 keV
        A_front = detector.calculate_absorption_probability('front', 70.0)
        A_rear = detector.calculate_absorption_probability('rear', 70.0)
        
        assert 0 < A_front < 1
        assert 0 < A_rear < 1
        assert A_rear > A_front  # Rear is thicker, should absorb more


class TestDetectorPhysics:
    """Test detector physics calculations."""
    
    def test_gain_calculations(self):
        """Test gain parameter calculations."""
        gain = DetectorGainParameters()
        
        # Check individual components
        assert gain.optical_photons_per_kev == 54.0
        assert gain.optical_coupling_efficiency == 0.75
        
        # Check total gain
        expected_gain = 54.0 * 0.75 * 0.80 * 0.01
        assert abs(gain.total_gain_adu_per_kev - expected_gain) < 1e-6
        
    def test_signal_calculation(self):
        """Test mean signal calculation."""
        detector = DualLayerDetectorModel()
        physics = DetectorPhysics(detector)
        
        # Create test fluence
        fluence = np.ones((10, 10, 5)) * 1000  # 1000 photons/mm²/bin
        energies = np.array([20, 40, 60, 80, 100])
        
        # Calculate signal
        signal = physics.calculate_mean_signal(fluence, energies, 'front')
        
        assert signal.shape == (10, 10)
        assert np.all(signal > 0)
        
    def test_noise_calculation(self):
        """Test noise variance calculation."""
        detector = DualLayerDetectorModel()
        physics = DetectorPhysics(detector)
        
        # Create test fluence
        fluence = np.ones((10, 10, 1)) * 10000  # High fluence
        energies = np.array([70.0])
        
        # Calculate noise
        noise_var = physics.calculate_noise_variance(
            fluence, energies, 'front', integration_time_s=0.1
        )
        
        assert noise_var.shape == (10, 10)
        assert np.all(noise_var > 0)
        
    def test_mtf_blur(self):
        """Test MTF blur application."""
        detector = DualLayerDetectorModel()
        physics = DetectorPhysics(detector)
        
        # Create sharp edge
        image = np.zeros((50, 50))
        image[:, 25:] = 100.0
        
        # Apply blur
        blurred = physics.apply_mtf_blur(image, 'front')
        
        # Check that blur was applied
        assert blurred.shape == image.shape
        edge_sharpness_original = np.max(np.gradient(image[25, :]))
        edge_sharpness_blurred = np.max(np.gradient(blurred[25, :]))
        assert edge_sharpness_blurred < edge_sharpness_original


class TestCrosstalkMask:
    """Test anti-crosstalk mask functionality."""
    
    def test_mask_initialization(self):
        """Test mask initialization with default parameters."""
        mask = CrosstalkMask()
        
        assert mask.spec.material == 'tungsten'
        assert mask.spec.grid_ratio == 10.0
        assert mask.spec.primary_transmission == 0.95
        
    def test_mask_pattern_generation(self):
        """Test binary mask pattern generation."""
        mask = CrosstalkMask()
        pattern = mask.generate_mask_pattern(20, 20)
        
        assert pattern.shape == (20, 20)
        assert np.all((pattern == 0) | (pattern == 1))
        
        # Check periodicity
        period_pixels = int(mask.spec.grid_period_mm / 0.5)
        assert period_pixels == 1
        
    def test_crosstalk_attenuation(self):
        """Test off-axis attenuation calculation."""
        mask = CrosstalkMask()
        
        # Test at stereo angle (12°)
        atten = mask.calculate_crosstalk_attenuation(12.0, 80.0)
        assert atten < 0.001  # Should block >99.9%
        
        # Test at small angle (within acceptance)
        atten_small = mask.calculate_crosstalk_attenuation(3.0, 80.0)
        assert atten_small > atten  # Less blocking at smaller angles
        
    def test_effective_grid_ratio(self):
        """Test energy-dependent effective grid ratio."""
        mask = CrosstalkMask()
        
        # Test at different energies
        ratio_low = mask.calculate_effective_grid_ratio(20.0)
        ratio_high = mask.calculate_effective_grid_ratio(120.0)
        
        assert ratio_low > ratio_high  # Higher energy penetrates more


class TestDetectorSimulator:
    """Test integrated detector simulator."""
    
    def test_simulator_initialization(self):
        """Test simulator initialization."""
        simulator = DetectorSimulator()
        
        assert simulator.enable_noise == True
        assert simulator.enable_blur == True
        assert simulator.enable_crosstalk == True
        
    def test_single_detector_simulation(self):
        """Test single detector response simulation."""
        simulator = DetectorSimulator(enable_noise=False)
        
        # Create test fluence
        fluence = np.ones((50, 50, 5)) * 1000
        energies = np.array([30, 50, 70, 90, 110])
        
        # Simulate response
        response = simulator.simulate_detector_response(
            fluence, energies, 'front'
        )
        
        assert 'signal' in response
        assert 'signal_clean' in response
        assert 'snr_map' in response
        
        # Check signal properties
        assert response['signal'].shape == (50, 50)
        assert np.all(response['signal'] >= 0)
        
    def test_stereo_pair_simulation(self):
        """Test stereo pair simulation."""
        simulator = DetectorSimulator()
        
        # Create test fluences
        shape = (30, 30)
        energies = np.linspace(20, 120, 10)
        left_fluence = simulator.create_test_fluence(shape, energies, 1000)
        right_fluence = simulator.create_test_fluence(shape, energies, 800)
        
        # Simulate stereo pair
        results = simulator.simulate_stereo_pair(
            left_fluence, right_fluence, energies
        )
        
        assert 'left' in results and 'right' in results
        assert results['left'].shape == shape
        assert results['right'].shape == shape
        
        # Check that signals are different
        assert not np.allclose(results['left'], results['right'])
    
    @pytest.mark.parametrize("enable_option", [
        ('enable_noise', True),
        ('enable_noise', False),
        ('enable_blur', True),
        ('enable_blur', False)
    ])
    def test_simulation_options(self, enable_option):
        """Test different simulation options."""
        option_name, option_value = enable_option
        
        # Create simulator with specific option
        kwargs = {option_name: option_value}
        simulator = DetectorSimulator(**kwargs)
        
        # Run simulation
        fluence = np.ones((20, 20, 1)) * 5000
        energies = np.array([70.0])
        
        response = simulator.simulate_detector_response(
            fluence, energies, 'front'
        )
        
        # Verify option had effect
        if option_name == 'enable_noise' and not option_value:
            # Without noise, signal should equal clean signal
            np.testing.assert_array_almost_equal(
                response['signal'], response['signal_clean']
            )


class TestIntegration:
    """Integration tests for complete workflow."""
    
    def test_complete_workflow(self):
        """Test complete detector simulation workflow."""
        # Initialize components
        detector_model = DualLayerDetectorModel()
        simulator = DetectorSimulator(detector_model)
        
        # Create realistic test data
        energies = np.linspace(15, 150, 136)  # Full energy range
        shape = (100, 100)
        
        # Simulate typical fluence after patient attenuation
        mean_fluence = 5000  # photons/mm²/bin
        left_fluence = np.random.poisson(mean_fluence, (*shape, len(energies)))
        right_fluence = np.random.poisson(mean_fluence * 0.8, (*shape, len(energies)))
        
        # Run simulation
        results = simulator.simulate_stereo_pair(
            left_fluence.astype(float),
            right_fluence.astype(float),
            energies,
            integration_time_s=0.1
        )
        
        # Verify results
        assert results['left'].shape == shape
        assert results['right'].shape == shape
        assert np.mean(results['left']) > np.mean(results['right'])
        
        # Check SNR is reasonable
        assert np.mean(results['left_snr']) > 10  # SNR should be >10 for good imaging
        assert np.mean(results['right_snr']) > 10


if __name__ == '__main__':
    pytest.main([__file__, '-v'])