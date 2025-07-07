"""
Ray Projection Data Types and Configuration.

This module defines the core data structures used throughout the ray projection
pipeline, including configuration parameters and data containers.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional, Union
import numpy as np
from pathlib import Path


@dataclass
class RayProjectionConfig:
    """Configuration parameters for ray projection.
    
    This class holds all configuration parameters needed for the
    branch-less Joseph's algorithm implementation on Metal GPU.
    """
    
    # Volume parameters
    volume_shape: Tuple[int, int, int]  # (width, height, depth) in voxels
    voxel_spacing: Tuple[float, float, float] = (0.5, 0.5, 0.5)  # mm
    
    # Detector parameters
    detector_size: Tuple[int, int] = (720, 720)  # pixels
    pixel_pitch: float = 0.5  # mm
    
    # Energy spectrum parameters
    energy_range: Tuple[float, float] = (15.0, 150.0)  # keV
    num_energy_bins: int = 136
    energy_bin_width: float = field(init=False)
    
    # GPU performance parameters
    threads_per_group: Tuple[int, int] = (32, 32)
    enable_early_exit: bool = True
    early_exit_threshold: float = 15.0  # Attenuation threshold for early termination
    
    # Ray marching parameters
    step_size_factor: float = 1.0  # Multiplier for voxel size to get step size
    use_adaptive_sampling: bool = False
    
    # Metal-specific parameters
    use_simd_vectorization: bool = True
    simd_width: int = 8  # Process 8 energy bins simultaneously
    
    # Tiling parameters for GPU workload management
    enable_tiling: bool = True
    tile_size: Tuple[int, int] = (180, 180)  # pixels per tile
    max_concurrent_tiles: int = 2  # Maximum tiles to process concurrently
    tile_overlap: int = 0  # Overlap between tiles in pixels
    
    def __post_init__(self):
        """Calculate derived parameters."""
        self.energy_bin_width = (self.energy_range[1] - self.energy_range[0]) / (self.num_energy_bins - 1)
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration parameters."""
        assert self.volume_shape[0] > 0 and self.volume_shape[1] > 0 and self.volume_shape[2] > 0
        assert all(s > 0 for s in self.voxel_spacing)
        assert self.detector_size[0] > 0 and self.detector_size[1] > 0
        assert self.pixel_pitch > 0
        assert self.energy_range[1] > self.energy_range[0]
        assert self.num_energy_bins > 0
        assert self.num_energy_bins % self.simd_width == 0, \
            f"Number of energy bins ({self.num_energy_bins}) must be divisible by SIMD width ({self.simd_width})"
    
    @property
    def num_simd_groups(self) -> int:
        """Number of SIMD groups needed for energy bins."""
        return self.num_energy_bins // self.simd_width


@dataclass
class MaterialData:
    """Container for material-specific data."""
    
    material_id: int
    material_name: str
    density_range: Tuple[float, float]  # g/cm³
    mass_atten_coeffs: np.ndarray  # μ/ρ values for each energy bin
    
    def validate(self, num_energy_bins: int):
        """Validate material data consistency."""
        assert len(self.mass_atten_coeffs) == num_energy_bins
        assert self.density_range[0] <= self.density_range[1]
        assert np.all(self.mass_atten_coeffs >= 0)


@dataclass
class ProjectionData:
    """Container for all data needed for ray projection.
    
    This class holds the preprocessed CT data, material information,
    energy spectrum, and geometry data needed for DRR generation.
    """
    
    # CT volume data
    ct_volume: np.ndarray  # HU values (int16)
    material_ids: np.ndarray  # Material segmentation (uint8)
    densities: np.ndarray  # Physical density in g/cm³ (float32)
    
    # Energy spectrum data
    source_spectrum: np.ndarray  # Photons per energy bin (pre-multiplied with detector)
    material_luts: Dict[int, np.ndarray]  # μ/ρ for each material [material_id][energy]
    
    # Geometry data (from stereo geometry pipeline)
    projection_matrices: Dict[str, np.ndarray]  # 'left' and 'right' 4x4 matrices
    source_positions: Dict[str, np.ndarray]  # World coordinates (x, y, z)
    detector_frames: Dict[str, np.ndarray]  # Detector coordinate frames
    
    # Optional precomputed data
    volume_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None  # Min/max corners
    
    # Output containers
    drr_images: Optional[Dict[str, np.ndarray]] = None  # Generated DRRs
    timing_info: Optional[Dict[str, float]] = None  # Performance metrics
    
    def __post_init__(self):
        """Validate data consistency."""
        self._validate_shapes()
        self._validate_types()
        self._compute_bounds()
    
    def _validate_shapes(self):
        """Ensure all arrays have consistent shapes."""
        assert self.ct_volume.shape == self.material_ids.shape == self.densities.shape, \
            "CT volume, material IDs, and densities must have the same shape"
        
        assert len(self.source_spectrum.shape) == 1, "Source spectrum must be 1D"
        
        for view in ['left', 'right']:
            assert view in self.projection_matrices, f"Missing {view} projection matrix"
            assert view in self.source_positions, f"Missing {view} source position"
            assert self.projection_matrices[view].shape == (4, 4), \
                f"{view} projection matrix must be 4x4"
            assert self.source_positions[view].shape == (3,) or self.source_positions[view].shape == (4,), \
                f"{view} source position must be 3D or homogeneous 4D"
    
    def _validate_types(self):
        """Ensure correct data types for GPU compatibility."""
        assert self.ct_volume.dtype == np.int16, "CT volume must be int16"
        assert self.material_ids.dtype == np.uint8, "Material IDs must be uint8"
        assert self.densities.dtype == np.float32, "Densities must be float32"
        assert self.source_spectrum.dtype == np.float32, "Source spectrum must be float32"
        
        for mat_id, lut in self.material_luts.items():
            assert lut.dtype == np.float32, f"Material {mat_id} LUT must be float32"
    
    def _compute_bounds(self):
        """Compute volume bounding box in world coordinates."""
        if self.volume_bounds is None:
            # Volume bounds in voxel coordinates
            shape = np.array(self.ct_volume.shape)
            self.volume_bounds = (
                np.array([0, 0, 0], dtype=np.float32),
                shape.astype(np.float32) - 1
            )
    
    def get_material_info(self) -> Dict[int, str]:
        """Get mapping of material IDs to names."""
        # Standard material mapping
        return {
            0: 'air',
            1: 'lung',
            2: 'adipose',
            3: 'soft_tissue',
            4: 'bone'
        }


@dataclass
class ProjectionUniforms:
    """GPU uniform buffer structure for ray projection.
    
    This structure is designed to match the Metal shader uniform buffer
    layout for efficient data transfer to GPU.
    """
    
    # Transform matrices
    model_view_projection_matrix: np.ndarray  # 4x4 float32
    volume_to_world_matrix: np.ndarray  # 4x4 float32
    
    # Ray parameters
    source_position: np.ndarray  # float3 in world coordinates
    detector_origin: np.ndarray  # float3 in world coordinates
    detector_u_vec: np.ndarray  # float3 detector horizontal axis
    detector_v_vec: np.ndarray  # float3 detector vertical axis
    
    # Volume parameters
    volume_dimensions: np.ndarray  # int3
    voxel_spacing: np.ndarray  # float3
    
    # Performance flags
    enable_early_exit: bool
    early_exit_threshold: float
    
    # Energy parameters
    num_energy_bins: int
    num_simd_groups: int
    
    def to_bytes(self) -> bytes:
        """Convert to bytes for GPU upload."""
        # This would be implemented to match Metal's memory layout
        # For now, return a placeholder
        return b''
    
    @classmethod
    def from_projection_data(cls, 
                           data: ProjectionData, 
                           config: RayProjectionConfig,
                           view: str) -> 'ProjectionUniforms':
        """Create uniforms from projection data for a specific view."""
        return cls(
            model_view_projection_matrix=data.projection_matrices[view],
            volume_to_world_matrix=np.eye(4, dtype=np.float32),  # Will be set properly
            source_position=data.source_positions[view][:3],
            detector_origin=np.zeros(3, dtype=np.float32),  # Will be computed
            detector_u_vec=np.array([1, 0, 0], dtype=np.float32),  # Will be computed
            detector_v_vec=np.array([0, 1, 0], dtype=np.float32),  # Will be computed
            volume_dimensions=np.array(config.volume_shape, dtype=np.int32),
            voxel_spacing=np.array(config.voxel_spacing, dtype=np.float32),
            enable_early_exit=config.enable_early_exit,
            early_exit_threshold=config.early_exit_threshold,
            num_energy_bins=config.num_energy_bins,
            num_simd_groups=config.num_simd_groups
        )


@dataclass
class RayMarchingStats:
    """Statistics collected during ray marching for debugging/optimization."""
    
    total_rays: int = 0
    early_terminated_rays: int = 0
    average_steps_per_ray: float = 0.0
    min_steps: int = 0
    max_steps: int = 0
    
    # Memory bandwidth statistics
    texture_reads_gb: float = 0.0
    lut_reads_gb: float = 0.0
    total_memory_gb: float = 0.0
    
    # Timing breakdown
    kernel_time_ms: float = 0.0
    upload_time_ms: float = 0.0
    download_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    def report(self) -> str:
        """Generate a formatted report of statistics."""
        early_term_pct = (self.early_terminated_rays / self.total_rays * 100) if self.total_rays > 0 else 0
        bandwidth_gbps = (self.total_memory_gb / (self.kernel_time_ms / 1000)) if self.kernel_time_ms > 0 else 0
        
        return f"""
Ray Marching Statistics:
  Total rays: {self.total_rays:,}
  Early terminated: {self.early_terminated_rays:,} ({early_term_pct:.1f}%)
  Steps per ray: {self.average_steps_per_ray:.1f} (min: {self.min_steps}, max: {self.max_steps})
  
Memory Usage:
  Texture reads: {self.texture_reads_gb:.2f} GB
  LUT reads: {self.lut_reads_gb:.2f} GB
  Total: {self.total_memory_gb:.2f} GB
  Effective bandwidth: {bandwidth_gbps:.1f} GB/s
  
Timing:
  Kernel: {self.kernel_time_ms:.1f} ms
  Upload: {self.upload_time_ms:.1f} ms
  Download: {self.download_time_ms:.1f} ms
  Total: {self.total_time_ms:.1f} ms
"""