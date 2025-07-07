"""
Ray Projector - Main Interface for DRR Generation.

This module provides the high-level Python interface for generating
Digitally Reconstructed Radiographs using the branch-less Joseph's algorithm
on Metal GPU.
"""

import numpy as np
from typing import Dict, Optional, Tuple, Union
import logging
import time
from pathlib import Path
import json
import Metal

from ray_projection_types import (
    RayProjectionConfig, ProjectionData, ProjectionUniforms, RayMarchingStats
)
from material_attenuation_lut import MaterialAttenuationLUT
from metal_compute_pipeline import MetalComputePipeline
from materials_db import MaterialsDatabase
from ray_projection_cpu_reference import CPURayProjector, compare_cpu_gpu_results

logger = logging.getLogger(__name__)


class RayProjector:
    """Main interface for DRR generation using Joseph's algorithm.
    
    This class orchestrates the entire ray projection pipeline:
    1. Data preparation and GPU upload
    2. Stereo DRR generation
    3. Performance monitoring
    4. Result retrieval
    """
    
    def __init__(self, config: RayProjectionConfig):
        """Initialize the ray projector.
        
        Args:
            config: Ray projection configuration
        """
        self.config = config
        self.metal_pipeline = MetalComputePipeline()
        self.material_lut_manager = MaterialAttenuationLUT()
        
        # Performance tracking
        self.last_timing = {}
        self.stats = RayMarchingStats()
        
        # Data state
        self._gpu_data_uploaded = False
        self._current_projection_data = None
        
        logger.info("Initialized RayProjector with %dx%d detector",
                   config.detector_size[0], config.detector_size[1])
    
    def project_stereo_pair(self, 
                          projection_data: ProjectionData) -> Dict[str, np.ndarray]:
        """Generate left and right DRRs.
        
        This is the main entry point for stereo DRR generation.
        
        Args:
            projection_data: Input data including CT volume, materials, and geometry
            
        Returns:
            Dictionary with 'left' and 'right' DRR images
        """
        logger.info("Starting stereo DRR generation")
        start_time = time.time()
        
        # Validate input data
        self._validate_projection_data(projection_data)
        
        # Upload data to GPU if needed
        if not self._gpu_data_uploaded or self._current_projection_data != projection_data:
            self._upload_to_gpu(projection_data)
        
        # Generate DRRs
        drrs = {}
        
        # Process views separately for better GPU memory management
        for view in ['left', 'right']:
            # Create command buffer for this view
            command_buffer = self.metal_pipeline.command_queue.commandBuffer()
            command_buffer.setLabel_(f"DRRGeneration_{view}")
            
            # Add autorelease pool to manage temporary Metal objects
            import objc
            with objc.autorelease_pool():
                # Clear debug statistics before projection
                if view == 'left':
                    self.metal_pipeline.clear_debug_stats()
                
                # Create and update uniforms
                view_start = time.time()
                uniforms = self._create_uniforms(projection_data, view)
                self._update_uniforms_buffer(view, uniforms)
                
                # Encode projection with tiling
                self.metal_pipeline.encode_projection(command_buffer, view, 
                                                    self.config, uniforms)
                encode_time = time.time() - view_start
                
                # Add GPU timeout handler
                if hasattr(command_buffer, 'setTimeout_'):
                    command_buffer.setTimeout_(30.0)  # 30 second timeout
                
                # Commit and wait for completion
                gpu_start = time.time()
                command_buffer.commit()
                
                # Wait with periodic status checks
                max_wait_time = 60.0  # Maximum 60 seconds
                check_interval = 0.5
                elapsed = 0.0
                
                while command_buffer.status() < 3 and elapsed < max_wait_time:  # MTLCommandBufferStatusCompleted = 3
                    time.sleep(check_interval)
                    elapsed += check_interval
                    
                    if elapsed > 10.0 and int(elapsed) % 5 == 0:
                        logger.warning(f"GPU processing {view} view taking longer than expected: {elapsed:.1f}s")
                
                if elapsed >= max_wait_time:
                    raise RuntimeError(f"GPU timeout: {view} view took longer than {max_wait_time}s")
                
                gpu_time = time.time() - gpu_start
                
                # Check for GPU errors
                final_status = command_buffer.status()
                logger.debug(f"Command buffer {view} final status: {final_status}")
                
                if final_status == 5:  # MTLCommandBufferStatusError = 5
                    error = command_buffer.error()
                    raise RuntimeError(f"GPU error in {view} view: {error}")
                
                # Store timing
                if view == 'left':
                    left_encode_time = encode_time
                    left_gpu_time = gpu_time
                else:
                    right_encode_time = encode_time
                    right_gpu_time = gpu_time
        
        # Total GPU time
        gpu_time = left_gpu_time + right_gpu_time
        
        # Read debug statistics
        debug_stats = self.metal_pipeline.read_debug_stats()
        logger.info("Ray projection debug statistics:")
        logger.info("  Total rays: %d", debug_stats['total_rays'])
        logger.info("  Rays hit: %d (%.1f%%)", debug_stats['rays_hit'], 
                   debug_stats['hit_rate'] * 100)
        logger.info("  Rays missed: %d", debug_stats['rays_missed'])
        if debug_stats['total_rays'] > 0:
            logger.info("  Center pixel debug:")
            logger.info("    Ray origin (world): %s", debug_stats['center_pixel']['ray_origin_world'])
            logger.info("    Ray origin (voxel): %s", debug_stats['center_pixel']['ray_origin_voxel'])
            logger.info("    Ray dir (voxel): %s", debug_stats['center_pixel']['ray_dir_voxel'])
            logger.info("    Intersection tMin: %.3f, tMax: %.3f", 
                       debug_stats['center_pixel']['intersection_tMin'],
                       debug_stats['center_pixel']['intersection_tMax'])
            logger.info("    Volume min (shader): %s", debug_stats['center_pixel']['volume_min'])
            logger.info("    Volume max (shader): %s", debug_stats['center_pixel']['volume_max'])
            logger.info("    Volume dimensions (shader): %s", debug_stats['center_pixel']['volume_dimensions'])
        
        # Read results
        read_start = time.time()
        drrs['left'] = self.metal_pipeline.read_output_texture('left', self.config)
        drrs['right'] = self.metal_pipeline.read_output_texture('right', self.config)
        read_time = time.time() - read_start
        
        # Update timing info
        total_time = time.time() - start_time
        self.last_timing = {
            'total_ms': total_time * 1000,
            'gpu_ms': gpu_time * 1000,
            'encode_left_ms': left_encode_time * 1000,
            'encode_right_ms': right_encode_time * 1000,
            'readback_ms': read_time * 1000
        }
        
        # Update stats
        self.stats.kernel_time_ms = gpu_time * 1000
        self.stats.total_time_ms = total_time * 1000
        self.stats.download_time_ms = read_time * 1000
        
        logger.info("Stereo DRR generation complete: %.1f ms (GPU: %.1f ms)",
                   self.last_timing['total_ms'], self.last_timing['gpu_ms'])
        
        # Debug output
        logger.debug("Left DRR stats: min=%.2f, max=%.2f, mean=%.2f", 
                    drrs['left'].min(), drrs['left'].max(), drrs['left'].mean())
        logger.debug("Right DRR stats: min=%.2f, max=%.2f, mean=%.2f",
                    drrs['right'].min(), drrs['right'].max(), drrs['right'].mean())
        
        # Store results in projection data
        projection_data.drr_images = drrs
        projection_data.timing_info = self.last_timing
        
        return drrs
    
    def _validate_projection_data(self, projection_data: ProjectionData) -> None:
        """Validate input projection data.
        
        Args:
            projection_data: Data to validate
            
        Raises:
            ValueError: If validation fails
        """
        # Check volume shape matches config
        if projection_data.ct_volume.shape != tuple(self.config.volume_shape):
            raise ValueError(
                f"CT volume shape {projection_data.ct_volume.shape} doesn't match "
                f"config {self.config.volume_shape}"
            )
        
        # Check required geometry data
        for view in ['left', 'right']:
            if view not in projection_data.projection_matrices:
                raise ValueError(f"Missing {view} projection matrix")
            if view not in projection_data.source_positions:
                raise ValueError(f"Missing {view} source position")
        
        # Check energy bins
        if len(projection_data.source_spectrum) != self.config.num_energy_bins:
            raise ValueError(
                f"Source spectrum has {len(projection_data.source_spectrum)} bins, "
                f"expected {self.config.num_energy_bins}"
            )
        
        logger.debug("Projection data validation passed")
    
    def _upload_to_gpu(self, projection_data: ProjectionData) -> None:
        """Upload volume data and LUTs to GPU.
        
        Args:
            projection_data: Data to upload
        """
        logger.info("Uploading data to GPU")
        upload_start = time.time()
        
        # Prepare material data
        gpu_material_data = self.material_lut_manager.prepare_for_gpu(
            self.config,
            detector_model=None  # Already applied in preprocessing
        )
        
        # Update projection data with GPU-ready LUTs
        projection_data.material_luts = gpu_material_data['material_lut']
        projection_data.source_spectrum = gpu_material_data['spectrum']
        
        # Allocate GPU resources
        self.metal_pipeline.allocate_buffers(projection_data, self.config)
        
        # Mark as uploaded
        self._gpu_data_uploaded = True
        self._current_projection_data = projection_data
        
        upload_time = time.time() - upload_start
        self.stats.upload_time_ms = upload_time * 1000
        logger.info("GPU upload complete: %.1f ms", upload_time * 1000)
    
    def _create_uniforms(self, 
                        projection_data: ProjectionData,
                        view: str) -> ProjectionUniforms:
        """Create projection uniforms for a specific view.
        
        Args:
            projection_data: Projection data
            view: 'left' or 'right'
            
        Returns:
            Filled ProjectionUniforms structure
        """
        # Get projection matrix
        P = projection_data.projection_matrices[view]
        
        # Get source position
        source_pos = projection_data.source_positions[view]
        if source_pos.shape[0] == 4:
            source_pos = source_pos[:3]
        
        # Get detector frame from stereo geometry
        detector_frame = projection_data.detector_frames[view]
        
        # Extract detector parameters from frame
        # The frame columns are: [u_vec, v_vec, n_vec, origin]
        detector_origin = detector_frame[:3, 3].astype(np.float32)
        detector_u = detector_frame[:3, 0].astype(np.float32)  # Horizontal axis
        detector_v = detector_frame[:3, 1].astype(np.float32)  # Vertical axis
        
        # Apply pixel pitch to detector vectors if they are unit vectors
        if abs(np.linalg.norm(detector_u) - 1.0) < 0.01:
            detector_u = detector_u * self.config.pixel_pitch
            detector_v = detector_v * self.config.pixel_pitch
            logger.debug(f"Applied pixel pitch {self.config.pixel_pitch}mm to detector vectors")
        
        # Match CPU implementation: volume starts at (0,0,0) and extends to (shape * spacing)
        # No coordinate adjustments needed - use original geometry positions
        
        # Volume dimensions in mm
        volume_size_mm = np.array(self.config.volume_shape) * np.array(self.config.voxel_spacing)
        
        # Use identity matrix since we're working in world space
        identity_matrix = np.eye(4, dtype=np.float32)
        
        uniforms = ProjectionUniforms(
            model_view_projection_matrix=identity_matrix,
            volume_to_world_matrix=identity_matrix,  # Not used in world-space shader
            source_position=source_pos.astype(np.float32),
            detector_origin=detector_origin,
            detector_u_vec=detector_u,
            detector_v_vec=detector_v,
            volume_dimensions=np.array(self.config.volume_shape, dtype=np.int32),
            voxel_spacing=np.array(self.config.voxel_spacing, dtype=np.float32),
            enable_early_exit=self.config.enable_early_exit,
            early_exit_threshold=self.config.early_exit_threshold,
            num_energy_bins=self.config.num_energy_bins,
            num_simd_groups=self.config.num_simd_groups
        )
        
        # Debug logging
        logger.debug(f"Creating uniforms for {view} view:")
        logger.debug(f"  Source position: {source_pos}")
        logger.debug(f"  Detector origin: {detector_origin}")
        logger.debug(f"  Detector u vec: {detector_u}")
        logger.debug(f"  Detector v vec: {detector_v}")
        logger.debug(f"  Volume shape: {self.config.volume_shape}")
        logger.debug(f"  Voxel spacing: {self.config.voxel_spacing}")
        logger.debug(f"  Volume size (mm): {volume_size_mm}")
        
        # Volume bounds in world space (matching CPU)
        volume_min_world = np.array([0.0, 0.0, 0.0])
        volume_max_world = volume_size_mm
        logger.debug(f"  Volume bounds in world space: {volume_min_world} to {volume_max_world}")
        
        # Note: Source will be outside volume bounds, which is correct for C-arm geometry
        logger.debug(f"  Source-to-isocenter distance: {np.linalg.norm(source_pos[:3]):.1f}mm")
        
        return uniforms
    
    def _update_uniforms_buffer(self, view: str, uniforms: ProjectionUniforms) -> None:
        """Update GPU uniforms buffer for a view.
        
        Args:
            view: 'left' or 'right'
            uniforms: Uniforms to upload
        """
        # Convert uniforms to bytes using proper struct packing
        import struct
        
        # Debug: log input values
        logger.debug(f"=== Packing uniforms buffer for {view} ===")
        logger.debug(f"Volume dimensions: {uniforms.volume_dimensions} (type: {uniforms.volume_dimensions.dtype})")
        logger.debug(f"Voxel spacing: {uniforms.voxel_spacing}")
        logger.debug(f"Source position: {uniforms.source_position}")
        
        # Pack the uniforms buffer according to Metal struct layout
        buffer_data = bytearray()
        offset = 0
        
        # Two 4x4 matrices (16 floats each)
        matrix1_bytes = uniforms.model_view_projection_matrix.flatten(order='F').astype(np.float32).tobytes()
        buffer_data.extend(matrix1_bytes)
        logger.debug(f"Offset {offset}: model_view_projection_matrix ({len(matrix1_bytes)} bytes)")
        offset += len(matrix1_bytes)
        
        matrix2_bytes = uniforms.volume_to_world_matrix.flatten(order='F').astype(np.float32).tobytes()
        buffer_data.extend(matrix2_bytes)
        logger.debug(f"Offset {offset}: volume_to_world_matrix ({len(matrix2_bytes)} bytes)")
        offset += len(matrix2_bytes)
        
        # float3 + padding (4 floats each)
        source_bytes = struct.pack('4f', *uniforms.source_position, 0.0)
        buffer_data.extend(source_bytes)
        logger.debug(f"Offset {offset}: source_position = {uniforms.source_position} ({len(source_bytes)} bytes)")
        offset += len(source_bytes)
        
        detector_origin_bytes = struct.pack('4f', *uniforms.detector_origin, 0.0)
        buffer_data.extend(detector_origin_bytes)
        logger.debug(f"Offset {offset}: detector_origin ({len(detector_origin_bytes)} bytes)")
        offset += len(detector_origin_bytes)
        
        detector_u_bytes = struct.pack('4f', *uniforms.detector_u_vec, 0.0)
        buffer_data.extend(detector_u_bytes)
        logger.debug(f"Offset {offset}: detector_u_vec ({len(detector_u_bytes)} bytes)")
        offset += len(detector_u_bytes)
        
        detector_v_bytes = struct.pack('4f', *uniforms.detector_v_vec, 0.0)
        buffer_data.extend(detector_v_bytes)
        logger.debug(f"Offset {offset}: detector_v_vec ({len(detector_v_bytes)} bytes)")
        offset += len(detector_v_bytes)
        
        # int3 + padding (4 ints)
        vol_dim_bytes = struct.pack('4i', *uniforms.volume_dimensions, 0)
        buffer_data.extend(vol_dim_bytes)
        logger.debug(f"Offset {offset}: volume_dimensions = {uniforms.volume_dimensions} ({len(vol_dim_bytes)} bytes)")
        logger.debug(f"  Raw bytes: {vol_dim_bytes.hex()}")
        offset += len(vol_dim_bytes)
        
        # float3 + padding (4 floats)
        voxel_spacing_bytes = struct.pack('4f', *uniforms.voxel_spacing, 0.0)
        buffer_data.extend(voxel_spacing_bytes)
        logger.debug(f"Offset {offset}: voxel_spacing = {uniforms.voxel_spacing} ({len(voxel_spacing_bytes)} bytes)")
        offset += len(voxel_spacing_bytes)
        
        # bool, float, 2 ints
        bool_bytes = struct.pack('i', int(uniforms.enable_early_exit))
        buffer_data.extend(bool_bytes)
        logger.debug(f"Offset {offset}: enable_early_exit = {uniforms.enable_early_exit} ({len(bool_bytes)} bytes)")
        offset += len(bool_bytes)
        
        threshold_bytes = struct.pack('f', uniforms.early_exit_threshold)
        buffer_data.extend(threshold_bytes)
        logger.debug(f"Offset {offset}: early_exit_threshold = {uniforms.early_exit_threshold} ({len(threshold_bytes)} bytes)")
        offset += len(threshold_bytes)
        
        energy_bytes = struct.pack('2i', uniforms.num_energy_bins, uniforms.num_simd_groups)
        buffer_data.extend(energy_bytes)
        logger.debug(f"Offset {offset}: num_energy_bins={uniforms.num_energy_bins}, num_simd_groups={uniforms.num_simd_groups} ({len(energy_bytes)} bytes)")
        offset += len(energy_bytes)
        
        # Debug: log total buffer size and hex dump of critical section
        logger.debug(f"Total uniforms buffer size: {len(buffer_data)} bytes")
        
        # Hex dump around volume_dimensions (offset should be 192)
        vol_dim_offset = 192  # After 2 matrices (128) + 4 float3s (64)
        if vol_dim_offset + 16 <= len(buffer_data):
            hex_section = buffer_data[vol_dim_offset:vol_dim_offset+16].hex()
            logger.debug(f"Hex at volume_dimensions offset {vol_dim_offset}: {hex_section}")
        
        # Update existing buffer contents instead of creating new buffer
        buffer_key = f'uniforms_{view}'
        uniforms_buffer = self.metal_pipeline.buffers[buffer_key]
        
        # Copy data to existing buffer
        buffer_contents = uniforms_buffer.contents()
        buffer_contents.as_buffer(len(buffer_data))[:] = bytes(buffer_data)
        
        # Mark buffer as modified  
        if hasattr(uniforms_buffer, 'didModifyRange_'):
            uniforms_buffer.didModifyRange_(Metal.NSMakeRange(0, 256))  # Use full buffer size
        
        # Debug: Verify the data was written correctly
        logger.debug(f"Verifying uniforms buffer for {view}:")
        # Read back volume dimensions at offset 192
        vol_dim_offset = 192
        read_back = buffer_contents.as_buffer(vol_dim_offset + 16)[vol_dim_offset:vol_dim_offset+16]
        vol_dims_readback = struct.unpack('4i', read_back)
        logger.debug(f"  Volume dimensions readback: {vol_dims_readback}")
    
    def get_performance_report(self) -> str:
        """Get detailed performance report.
        
        Returns:
            Formatted performance report string
        """
        report = "Ray Projection Performance Report\n"
        report += "=" * 50 + "\n\n"
        
        if self.last_timing:
            report += "Last Run Timing:\n"
            report += f"  Total: {self.last_timing['total_ms']:.1f} ms\n"
            report += f"  GPU kernel: {self.last_timing['gpu_ms']:.1f} ms\n"
            report += f"  Encode left: {self.last_timing['encode_left_ms']:.1f} ms\n"
            report += f"  Encode right: {self.last_timing['encode_right_ms']:.1f} ms\n"
            report += f"  Readback: {self.last_timing['readback_ms']:.1f} ms\n"
            report += "\n"
        
        # Add detailed stats
        report += self.stats.report()
        
        return report
    
    def save_drrs(self, 
                 drrs: Dict[str, np.ndarray],
                 output_dir: Union[str, Path],
                 save_metadata: bool = True) -> None:
        """Save DRR images and metadata.
        
        Args:
            drrs: Dictionary with 'left' and 'right' DRR images
            output_dir: Output directory
            save_metadata: Whether to save metadata JSON
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save DRR images
        np.save(output_dir / "drr_left.npy", drrs['left'])
        np.save(output_dir / "drr_right.npy", drrs['right'])
        
        # Save as compressed NPZ
        np.savez_compressed(
            output_dir / "unrectified_drrs.npz",
            drr_left=drrs['left'],
            drr_right=drrs['right']
        )
        
        logger.info("Saved DRRs to %s", output_dir)
        
        # Save metadata
        if save_metadata:
            metadata = {
                'detector_size': list(self.config.detector_size),
                'pixel_pitch_mm': self.config.pixel_pitch,
                'energy_range_kev': list(self.config.energy_range),
                'num_energy_bins': self.config.num_energy_bins,
                'algorithm': 'branch-less Joseph',
                'timing': self.last_timing,
                'volume_shape': list(self.config.volume_shape),
                'voxel_spacing_mm': list(self.config.voxel_spacing)
            }
            
            with open(output_dir / "drr_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Saved metadata to %s", output_dir / "drr_metadata.json")
    
    def validate_against_cpu(self, 
                           projection_data: ProjectionData,
                           tolerance_hu: float = 2.0) -> Dict[str, Dict[str, float]]:
        """Validate GPU implementation against CPU reference.
        
        This method runs both CPU and GPU implementations and compares
        the results to ensure numerical accuracy.
        
        Args:
            projection_data: Input data for projection
            tolerance_hu: Tolerance in HU units (default: 2.0 HU)
            
        Returns:
            Dictionary with comparison metrics for each view
        """
        logger.info("Starting GPU validation against CPU reference")
        
        # Create CPU projector
        cpu_projector = CPURayProjector(self.config)
        
        # Generate DRRs with GPU
        logger.info("Generating DRRs with GPU...")
        gpu_start = time.time()
        gpu_drrs = self.project_stereo_pair(projection_data)
        gpu_time = time.time() - gpu_start
        
        # Generate DRRs with CPU
        logger.info("Generating DRRs with CPU reference...")
        cpu_start = time.time()
        cpu_drrs = cpu_projector.project_stereo_pair(projection_data)
        cpu_time = time.time() - cpu_start
        
        # Compare results
        logger.info("Comparing CPU and GPU results...")
        results = {}
        
        for view in ['left', 'right']:
            logger.info(f"Validating {view} view...")
            metrics = compare_cpu_gpu_results(
                cpu_drrs[view], 
                gpu_drrs[view],
                tolerance_hu
            )
            results[view] = metrics
        
        # Log performance comparison
        logger.info("Performance comparison:")
        logger.info("  GPU time: %.1f ms", gpu_time * 1000)
        logger.info("  CPU time: %.1f ms", cpu_time * 1000)
        logger.info("  Speedup: %.1fx", cpu_time / gpu_time)
        
        # Overall validation result
        all_passed = all(results[view]['passed'] for view in ['left', 'right'])
        logger.info("Overall validation: %s", "PASSED" if all_passed else "FAILED")
        
        return results