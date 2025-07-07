"""
Dual-Layer Flat-Panel Detector Model.

This module defines the architecture and specifications of the dual-layer
flat-panel detector system with anti-crosstalk mask for stereo DRR imaging.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

from materials_db import get_materials_db, MaterialsDatabase


class LayerType(Enum):
    """Types of layers in the detector stack."""
    ENTRANCE_WINDOW = "entrance_window"
    SCINTILLATOR = "scintillator"
    PHOTODIODE = "photodiode"
    SUBSTRATE = "substrate"
    AIR_GAP = "air_gap"
    ANTI_CROSSTALK_MASK = "anti_crosstalk_mask"


@dataclass
class DetectorLayer:
    """Specification for a single layer in the detector stack."""
    name: str
    layer_type: LayerType
    material: str
    thickness_mm: float
    position_mm: float  # Distance from entrance surface
    
    def __repr__(self):
        return f"{self.name}: {self.thickness_mm:.3f}mm {self.material}"


@dataclass
class DetectorSpecification:
    """Complete specification for a detector (front or rear)."""
    name: str
    layers: List[DetectorLayer]
    pixel_pitch_mm: float
    active_area_mm: Tuple[float, float]  # (width, height)
    
    @property
    def total_thickness_mm(self) -> float:
        """Calculate total detector thickness."""
        return sum(layer.thickness_mm for layer in self.layers)
    
    @property
    def scintillator_thickness_mm(self) -> float:
        """Get scintillator thickness."""
        scint_layers = [l for l in self.layers if l.layer_type == LayerType.SCINTILLATOR]
        return scint_layers[0].thickness_mm if scint_layers else 0.0


class DualLayerDetectorModel:
    """
    Model of the complete dual-layer flat-panel detector system.
    
    This class encapsulates the physical architecture of the detector,
    including both front and rear detectors and the anti-crosstalk mask.
    """
    
    def __init__(self):
        """Initialize the dual-layer detector model."""
        self.materials_db = get_materials_db()
        
        # Detector specifications
        self.pixel_pitch_mm = 0.5  # 0.5mm pixel pitch
        self.detector_size_pixels = (720, 720)  # 720x720 pixels
        self.active_area_mm = (360.0, 360.0)  # 360x360 mm active area
        
        # Calculate inter-detector spacing
        self.inter_detector_gap_mm = 5.0  # Air gap for crosstalk mask
        
        # Build detector layers
        self._build_front_detector()
        self._build_rear_detector()
        self._build_crosstalk_mask()
        
    def _build_front_detector(self):
        """Build the front detector layer stack."""
        layers = []
        position = 0.0
        
        # Entrance window / Reflector
        layers.append(DetectorLayer(
            name="Front Entrance Window",
            layer_type=LayerType.ENTRANCE_WINDOW,
            material="aluminum",
            thickness_mm=0.5,
            position_mm=position
        ))
        position += 0.5
        
        # Scintillator
        layers.append(DetectorLayer(
            name="Front Scintillator",
            layer_type=LayerType.SCINTILLATOR,
            material="csi_tl",
            thickness_mm=0.5,
            position_mm=position
        ))
        position += 0.5
        
        # Photodiode array
        layers.append(DetectorLayer(
            name="Front Photodiode",
            layer_type=LayerType.PHOTODIODE,
            material="a_si_h",
            thickness_mm=0.001,  # 1.0 µm
            position_mm=position
        ))
        position += 0.001
        
        # Substrate
        layers.append(DetectorLayer(
            name="Front Substrate",
            layer_type=LayerType.SUBSTRATE,
            material="glass",
            thickness_mm=1.0,
            position_mm=position
        ))
        
        self.front_detector = DetectorSpecification(
            name="Front Detector (Layer 1)",
            layers=layers,
            pixel_pitch_mm=self.pixel_pitch_mm,
            active_area_mm=self.active_area_mm
        )
        
    def _build_rear_detector(self):
        """Build the rear detector layer stack."""
        layers = []
        # Position starts after front detector + gap
        position = self.front_detector.total_thickness_mm + self.inter_detector_gap_mm
        
        # Scintillator (thicker for rear detector)
        layers.append(DetectorLayer(
            name="Rear Scintillator",
            layer_type=LayerType.SCINTILLATOR,
            material="csi_tl",
            thickness_mm=0.6,  # 0.6mm for enhanced absorption
            position_mm=position
        ))
        position += 0.6
        
        # Photodiode array
        layers.append(DetectorLayer(
            name="Rear Photodiode",
            layer_type=LayerType.PHOTODIODE,
            material="a_si_h",
            thickness_mm=0.001,  # 1.0 µm
            position_mm=position
        ))
        position += 0.001
        
        # Substrate
        layers.append(DetectorLayer(
            name="Rear Substrate",
            layer_type=LayerType.SUBSTRATE,
            material="glass",
            thickness_mm=1.0,
            position_mm=position
        ))
        
        self.rear_detector = DetectorSpecification(
            name="Rear Detector (Layer 2)",
            layers=layers,
            pixel_pitch_mm=self.pixel_pitch_mm,
            active_area_mm=self.active_area_mm
        )
        
    def _build_crosstalk_mask(self):
        """Define the anti-crosstalk mask specifications."""
        # Grid specifications from research
        self.crosstalk_mask = {
            'material': 'tungsten',
            'grid_ratio': 10.0,  # 10:1
            'septa_thickness_mm': 0.025,  # 25 µm
            'slit_width_mm': 0.475,  # 475 µm
            'septa_height_mm': 4.75,  # 4.75 mm
            'grid_period_mm': 0.5,  # Matches pixel pitch
            'primary_transmission': 0.95,  # 95% transmission
            'focusing_distance_mm': 1500.0  # SDD
        }
        
    def calculate_stack_transmission(self, energy_kev: np.ndarray, 
                                   include_mask: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate transmission through the detector stack.
        
        Args:
            energy_kev: Energy array in keV
            include_mask: Whether to include crosstalk mask transmission
            
        Returns:
            Tuple of (front_stack_transmission, total_stack_transmission)
        """
        if np.isscalar(energy_kev):
            energy_kev = np.array([energy_kev])
            
        # Front detector stack transmission
        T_front = np.ones_like(energy_kev, dtype=float)
        
        for layer in self.front_detector.layers:
            T_layer = self.materials_db.get_transmission(
                layer.material, layer.thickness_mm, energy_kev
            )
            T_front *= T_layer
            
        # Total transmission (through both detectors)
        T_total = T_front.copy()
        
        # Add crosstalk mask if requested
        if include_mask:
            T_total *= self.crosstalk_mask['primary_transmission']
            
        # Add rear detector transmission
        for layer in self.rear_detector.layers:
            T_layer = self.materials_db.get_transmission(
                layer.material, layer.thickness_mm, energy_kev
            )
            T_total *= T_layer
            
        return T_front, T_total
    
    def calculate_absorption_probability(self, detector: str, 
                                       energy_kev: np.ndarray) -> np.ndarray:
        """
        Calculate absorption probability in a specific detector's scintillator.
        
        Args:
            detector: 'front' or 'rear'
            energy_kev: Energy array in keV
            
        Returns:
            Absorption probability array
        """
        if detector == 'front':
            scint_thickness = self.front_detector.scintillator_thickness_mm
        elif detector == 'rear':
            scint_thickness = self.rear_detector.scintillator_thickness_mm
        else:
            raise ValueError("Detector must be 'front' or 'rear'")
            
        return self.materials_db.get_absorption_probability(
            'csi_tl', scint_thickness, energy_kev
        )
    
    def get_detector_parameters(self, detector: str) -> dict:
        """
        Get key parameters for a specific detector.
        
        Args:
            detector: 'front' or 'rear'
            
        Returns:
            Dictionary of detector parameters
        """
        det_spec = self.front_detector if detector == 'front' else self.rear_detector
        
        return {
            'name': det_spec.name,
            'scintillator_thickness_mm': det_spec.scintillator_thickness_mm,
            'total_thickness_mm': det_spec.total_thickness_mm,
            'pixel_pitch_mm': det_spec.pixel_pitch_mm,
            'active_area_mm': det_spec.active_area_mm,
            'num_pixels': self.detector_size_pixels
        }
    
    def print_configuration(self):
        """Print detailed detector configuration."""
        print("Dual-Layer Flat-Panel Detector Configuration")
        print("=" * 60)
        
        # Front detector
        print(f"\n{self.front_detector.name}:")
        print(f"  Total thickness: {self.front_detector.total_thickness_mm:.3f} mm")
        for layer in self.front_detector.layers:
            print(f"  - {layer}")
            
        # Rear detector
        print(f"\n{self.rear_detector.name}:")
        print(f"  Total thickness: {self.rear_detector.total_thickness_mm:.3f} mm")
        for layer in self.rear_detector.layers:
            print(f"  - {layer}")
            
        # Crosstalk mask
        print(f"\nAnti-Crosstalk Mask:")
        print(f"  Material: {self.crosstalk_mask['material']}")
        print(f"  Grid ratio: {self.crosstalk_mask['grid_ratio']}:1")
        print(f"  Septa: {self.crosstalk_mask['septa_thickness_mm']*1000:.0f} µm")
        print(f"  Slits: {self.crosstalk_mask['slit_width_mm']*1000:.0f} µm")
        print(f"  Primary transmission: {self.crosstalk_mask['primary_transmission']*100:.1f}%")
        
        # System parameters
        print(f"\nSystem Parameters:")
        print(f"  Pixel pitch: {self.pixel_pitch_mm} mm")
        print(f"  Detector size: {self.detector_size_pixels[0]}x{self.detector_size_pixels[1]} pixels")
        print(f"  Active area: {self.active_area_mm[0]}x{self.active_area_mm[1]} mm²")
        print(f"  Inter-detector gap: {self.inter_detector_gap_mm} mm")


def test_detector_model():
    """Test the detector model functionality."""
    detector = DualLayerDetectorModel()
    detector.print_configuration()
    
    # Test transmission calculations
    print("\n\nTransmission Analysis:")
    print("-" * 40)
    test_energies = np.array([20, 40, 60, 80, 100, 120])
    
    T_front, T_total = detector.calculate_stack_transmission(test_energies)
    
    print("Energy (keV) | Front Trans | Total Trans")
    print("-" * 40)
    for i, E in enumerate(test_energies):
        print(f"{E:11.0f} | {T_front[i]:11.4f} | {T_total[i]:11.4f}")
        
    # Test absorption probabilities
    print("\n\nScintillator Absorption Probabilities:")
    print("-" * 40)
    A_front = detector.calculate_absorption_probability('front', test_energies)
    A_rear = detector.calculate_absorption_probability('rear', test_energies)
    
    print("Energy (keV) | Front Abs % | Rear Abs %")
    print("-" * 40)
    for i, E in enumerate(test_energies):
        print(f"{E:11.0f} | {A_front[i]*100:11.1f} | {A_rear[i]*100:10.1f}")


if __name__ == '__main__':
    test_detector_model()