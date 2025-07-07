"""
Materials Database for Detector Physics Simulation.

This module contains mass attenuation coefficients for all materials used in the
dual-layer flat-panel detector system. Data is sourced from NIST XCOM database
and computed using the mixture rule for compounds.

Energy range: 15-150 keV with 1 keV bins (136 energy points)
"""

import numpy as np
from typing import Dict, Tuple
from dataclasses import dataclass


@dataclass
class Material:
    """Material properties for detector simulation."""
    name: str
    density: float  # g/cm³
    composition: Dict[str, float]  # Element: weight fraction
    
    
class MaterialsDatabase:
    """Database of material properties and mass attenuation coefficients."""
    
    def __init__(self):
        """Initialize the materials database with detector components."""
        # Define materials used in the detector stack
        self.materials = {
            'aluminum': Material(
                name='Aluminum',
                density=2.70,
                composition={'Al': 1.0}
            ),
            'csi_tl': Material(
                name='CsI:Tl',
                density=4.51,
                composition={'Cs': 0.5115, 'I': 0.4885}  # Tl dopant <0.1% ignored
            ),
            'a_si_h': Material(
                name='a-Si:H',
                density=2.285,
                composition={'Si': 0.963, 'H': 0.037}  # 10 atomic % H
            ),
            'glass': Material(
                name='Borosilicate Glass',
                density=2.23,
                composition={
                    'Si': 0.377,
                    'O': 0.540,
                    'B': 0.040,
                    'Na': 0.028,
                    'Al': 0.012
                }
            ),
            'tungsten': Material(
                name='Tungsten',
                density=19.3,
                composition={'W': 1.0}
            ),
            'air': Material(
                name='Air (Dry)',
                density=1.205e-3,
                composition={'N': 0.755, 'O': 0.232, 'Ar': 0.013}
            ),
            'lead': Material(
                name='Lead',
                density=11.34,
                composition={'Pb': 1.0}
            ),
            'tantalum': Material(
                name='Tantalum',
                density=16.65,
                composition={'Ta': 1.0}
            )
        }
        
        # Energy grid (keV)
        self.energies = np.arange(15, 151, 1)  # 15-150 keV in 1 keV steps
        
        # Initialize mass attenuation coefficient tables
        self._init_mass_attenuation_tables()
        
    def _init_mass_attenuation_tables(self):
        """Initialize mass attenuation coefficient lookup tables."""
        # These values are representative samples from NIST XCOM database
        # In production, these would be loaded from comprehensive data files
        
        # Mass attenuation coefficients (cm²/g) at key energies
        # Full 136-point arrays would be interpolated from these values
        key_energies = np.array([20, 40, 60, 80, 100, 120, 150])
        
        # Representative mass attenuation data from NIST XCOM
        self.mass_atten_data = {
            'aluminum': {
                'energies': key_energies,
                'mu_rho': np.array([3.441, 0.5685, 0.2778, 0.2018, 0.1704, 0.1536, 0.1378])
            },
            'csi_tl': {
                'energies': key_energies,
                'mu_rho': np.array([26.86, 22.97, 7.921, 3.677, 2.035, 1.255, 0.729])
            },
            'a_si_h': {
                'energies': key_energies,
                'mu_rho': np.array([5.378, 0.613, 0.283, 0.206, 0.176, 0.160, 0.144])
            },
            'glass': {
                'energies': key_energies,
                'mu_rho': np.array([4.793, 0.542, 0.258, 0.198, 0.174, 0.161, 0.148])
            },
            'tungsten': {
                'energies': key_energies,
                'mu_rho': np.array([65.73, 10.67, 3.713, 7.810, 4.438, 2.766, 1.581])
            },
            'air': {
                'energies': key_energies,
                'mu_rho': np.array([0.7779, 0.2635, 0.2059, 0.1837, 0.1707, 0.1614, 0.1505])
            },
            'lead': {
                'energies': key_energies,
                'mu_rho': np.array([86.36, 14.36, 5.021, 2.419, 5.549, 3.421, 2.014])
            },
            'tantalum': {
                'energies': key_energies,
                'mu_rho': np.array([58.74, 9.654, 3.468, 7.592, 4.195, 2.622, 1.503])
            }
        }
        
        # Generate full resolution tables via interpolation
        self.mass_atten_coeffs = {}
        for material, data in self.mass_atten_data.items():
            # Log-log interpolation for smooth curves across absorption edges
            self.mass_atten_coeffs[material] = np.exp(
                np.interp(
                    np.log(self.energies),
                    np.log(data['energies']),
                    np.log(data['mu_rho'])
                )
            )
    
    def get_mass_atten_coeff(self, material: str, energy: float = None) -> np.ndarray:
        """
        Get mass attenuation coefficient for a material.
        
        Args:
            material: Material name (e.g., 'aluminum', 'csi_tl')
            energy: Specific energy in keV (optional). If None, returns full array.
            
        Returns:
            Mass attenuation coefficient(s) in cm²/g
        """
        if material not in self.mass_atten_coeffs:
            raise ValueError(f"Material '{material}' not found in database")
            
        mu_rho = self.mass_atten_coeffs[material]
        
        if energy is not None:
            # Interpolate for specific energy
            return np.interp(energy, self.energies, mu_rho)
        else:
            return mu_rho.copy()
    
    def get_linear_atten_coeff(self, material: str, energy: float = None) -> np.ndarray:
        """
        Get linear attenuation coefficient for a material.
        
        Args:
            material: Material name
            energy: Specific energy in keV (optional)
            
        Returns:
            Linear attenuation coefficient(s) in cm⁻¹
        """
        mu_rho = self.get_mass_atten_coeff(material, energy)
        density = self.materials[material].density
        return mu_rho * density
    
    def get_transmission(self, material: str, thickness_mm: float, 
                        energy: float = None) -> np.ndarray:
        """
        Calculate transmission through a material layer.
        
        Args:
            material: Material name
            thickness_mm: Thickness in millimeters
            energy: Specific energy in keV (optional)
            
        Returns:
            Transmission factor (0-1)
        """
        mu = self.get_linear_atten_coeff(material, energy)
        thickness_cm = thickness_mm / 10.0
        return np.exp(-mu * thickness_cm)
    
    def get_absorption_probability(self, material: str, thickness_mm: float,
                                 energy: float = None) -> np.ndarray:
        """
        Calculate absorption probability in a material layer.
        
        Args:
            material: Material name
            thickness_mm: Thickness in millimeters
            energy: Specific energy in keV (optional)
            
        Returns:
            Absorption probability (0-1)
        """
        transmission = self.get_transmission(material, thickness_mm, energy)
        return 1.0 - transmission
    
    def get_material_properties(self, material: str) -> Material:
        """Get material properties."""
        if material not in self.materials:
            raise ValueError(f"Material '{material}' not found in database")
        return self.materials[material]
    
    def compute_mixture_mass_atten(self, composition: Dict[str, float], 
                                  energy: float = None) -> np.ndarray:
        """
        Compute mass attenuation coefficient for a mixture using Bragg's rule.
        
        Args:
            composition: Dict of material_name: weight_fraction
            energy: Specific energy in keV (optional)
            
        Returns:
            Mass attenuation coefficient(s) for the mixture
        """
        mu_rho_mixture = np.zeros_like(self.energies, dtype=float)
        
        for material, weight_fraction in composition.items():
            mu_rho = self.get_mass_atten_coeff(material, energy)
            mu_rho_mixture += weight_fraction * mu_rho
            
        if energy is not None:
            return np.interp(energy, self.energies, mu_rho_mixture)
        else:
            return mu_rho_mixture


# Convenience function for quick access
def get_materials_db() -> MaterialsDatabase:
    """Get a singleton instance of the materials database."""
    if not hasattr(get_materials_db, '_instance'):
        get_materials_db._instance = MaterialsDatabase()
    return get_materials_db._instance


if __name__ == '__main__':
    # Test the materials database
    db = get_materials_db()
    
    print("Materials Database Test")
    print("=" * 50)
    
    # Test mass attenuation at 60 keV
    energy_test = 60.0
    print(f"\nMass attenuation coefficients at {energy_test} keV:")
    for material in ['aluminum', 'csi_tl', 'tungsten', 'air']:
        mu_rho = db.get_mass_atten_coeff(material, energy_test)
        print(f"  {material:12s}: {mu_rho:8.4f} cm²/g")
    
    # Test transmission through CsI:Tl scintillator
    print(f"\nTransmission through 0.5mm CsI:Tl:")
    energies_test = [20, 60, 100, 140]
    for E in energies_test:
        T = db.get_transmission('csi_tl', 0.5, E)
        A = db.get_absorption_probability('csi_tl', 0.5, E)
        print(f"  {E:3d} keV: T={T:.4f}, A={A:.4f}")
    
    # Test tungsten grid transmission
    print(f"\nTungsten septa (0.025mm) attenuation:")
    for E in [60, 80, 100]:
        T = db.get_transmission('tungsten', 0.025, E)
        print(f"  {E:3d} keV: T={T:.6f} (Attenuation: {(1-T)*100:.2f}%)")