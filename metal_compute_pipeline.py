"""
Metal Compute Pipeline for Ray Projection.

This module manages the Metal GPU infrastructure for high-performance
DRR generation using the branch-less Joseph's algorithm.
"""

import Metal
import MetalKit
import numpy as np
from typing import Dict, Optional, Tuple, Any
import logging
from pathlib import Path
import time
import struct
from Foundation import NSData

from ray_projection_types import (
    RayProjectionConfig, ProjectionData, ProjectionUniforms, RayMarchingStats
)

logger = logging.getLogger(__name__)


class MetalComputePipeline:
    """Manages Metal device, shaders, and command encoding for ray projection.
    
    This class handles:
    1. Metal device initialization
    2. Shader compilation and pipeline state creation
    3. Buffer and texture allocation
    4. Command encoding for projection kernels
    5. Performance monitoring
    """
    
    def __init__(self):
        """Initialize the Metal compute pipeline."""
        self.device = None
        self.command_queue = None
        self.library = None
        self.pipeline_state = None
        self.buffers = {}
        self.textures = {}
        
        # Performance monitoring
        self.enable_gpu_timing = True
        self.stats = RayMarchingStats()
        
        # Buffer pool for tile offsets to avoid constant allocation
        self.tile_offset_pool = []
        self.tile_offset_pool_size = 16  # Pre-allocate some buffers
        
        self._initialize_metal()
        self._compile_shaders()
        self._initialize_buffer_pools()
        
        logger.info("Metal compute pipeline initialized successfully")
    
    def _initialize_metal(self):
        """Initialize Metal device and command queue."""
        # Get default Metal device
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("Metal is not supported on this system")
        
        logger.info("Using Metal device: %s", self.device.name())
        
        # Log device capabilities
        if hasattr(self.device, 'supportsFamily_'):
            gpu_family = "Unknown"
            # Check for Apple GPU families
            try:
                if self.device.supportsFamily_(Metal.MTLGPUFamilyApple9):
                    gpu_family = "Apple GPU Family 9 (M3/M4)"
                elif self.device.supportsFamily_(Metal.MTLGPUFamilyApple8):
                    gpu_family = "Apple GPU Family 8 (M2)"
                elif self.device.supportsFamily_(Metal.MTLGPUFamilyApple7):
                    gpu_family = "Apple GPU Family 7 (M1)"
            except:
                pass
            logger.info("GPU Family: %s", gpu_family)
        
        # Create command queue
        self.command_queue = self.device.newCommandQueue()
        if self.command_queue is None:
            raise RuntimeError("Failed to create Metal command queue")
        
        # Log memory info
        if hasattr(self.device, 'recommendedMaxWorkingSetSize'):
            try:
                max_memory_gb = self.device.recommendedMaxWorkingSetSize() / (1024**3)
                logger.info("Recommended max working set: %.1f GB", max_memory_gb)
            except:
                pass
    
    def _compile_shaders(self):
        """Compile Metal shaders from source."""
        # Find shader file
        shader_path = Path(__file__).parent / "shaders" / "joseph_projector.metal"
        
        if not shader_path.exists():
            raise FileNotFoundError(f"Shader file not found at {shader_path}")
        
        # Read shader source
        with open(shader_path, 'r') as f:
            shader_source = f.read()
        
        # Compile shader library
        options = Metal.MTLCompileOptions.new()
        
        library_result = self.device.newLibraryWithSource_options_error_(
            shader_source, options, None
        )
        
        if library_result is None or library_result[0] is None:
            error_msg = library_result[1] if library_result and library_result[1] else "Unknown error"
            raise RuntimeError(f"Failed to compile shader library: {error_msg}")
        
        self.library = library_result[0]
        
        # Get kernel function
        kernel_name = "projectPolyenergeticDRR"
        kernel_function = self.library.newFunctionWithName_(kernel_name)
        if kernel_function is None:
            raise RuntimeError(f"Failed to find kernel function '{kernel_name}'")
        
        # Create compute pipeline state
        pipeline_result = self.device.newComputePipelineStateWithFunction_error_(
            kernel_function, None
        )
        
        if pipeline_result is None or pipeline_result[0] is None:
            error_msg = pipeline_result[1] if pipeline_result and pipeline_result[1] else "Unknown error"
            raise RuntimeError(f"Failed to create pipeline state: {error_msg}")
        
        self.pipeline_state = pipeline_result[0]
        
        logger.info("Shaders compiled successfully")
        
        # Log thread execution width (warp size)
        thread_width = self.pipeline_state.threadExecutionWidth()
        logger.info("Thread execution width: %d", thread_width)
        
        # Log max threads per threadgroup
        max_threads = self.pipeline_state.maxTotalThreadsPerThreadgroup()
        logger.info("Max threads per threadgroup: %d", max_threads)
    
    def _initialize_buffer_pools(self):
        """Initialize reusable buffer pools."""
        # Pre-allocate tile offset buffers
        for _ in range(self.tile_offset_pool_size):
            tile_offset_buffer = self.device.newBufferWithLength_options_(
                8,  # 2 * uint32
                Metal.MTLResourceStorageModeShared
            )
            self.tile_offset_pool.append(tile_offset_buffer)
        logger.debug("Initialized buffer pool with %d tile offset buffers", self.tile_offset_pool_size)
    
    def _get_tile_offset_buffer(self, tile_x: int, tile_y: int) -> Any:
        """Get a tile offset buffer from the pool or create a new one."""
        if self.tile_offset_pool:
            buffer = self.tile_offset_pool.pop()
        else:
            buffer = self.device.newBufferWithLength_options_(
                8,  # 2 * uint32
                Metal.MTLResourceStorageModeShared
            )
        
        # Update buffer contents
        tile_offset_data = np.array([tile_x, tile_y], dtype=np.uint32)
        buffer.contents().as_buffer(8)[:] = tile_offset_data.tobytes()
        
        return buffer
    
    def _return_tile_offset_buffer(self, buffer: Any):
        """Return a tile offset buffer to the pool."""
        if len(self.tile_offset_pool) < self.tile_offset_pool_size * 2:
            self.tile_offset_pool.append(buffer)
    
    def allocate_buffers(self, 
                        projection_data: ProjectionData,
                        config: RayProjectionConfig) -> None:
        """Allocate GPU buffers and textures.
        
        Args:
            projection_data: Input data for projection
            config: Ray projection configuration
        """
        logger.info("Allocating GPU resources")
        
        # Calculate buffer sizes
        volume_size = projection_data.ct_volume.nbytes
        material_size = projection_data.material_ids.nbytes
        density_size = projection_data.densities.nbytes
        
        logger.info("Volume data: %.1f MB", 
                   (volume_size + material_size + density_size) / (1024**2))
        
        # Create 3D textures for volume data
        self._create_volume_textures(projection_data, config)
        
        # Create buffers for LUTs and spectrum
        self._create_constant_buffers(projection_data, config)
        
        # Create output textures for DRRs
        self._create_output_textures(config)
        
        logger.info("GPU resource allocation complete")
    
    def _create_volume_textures(self, 
                              projection_data: ProjectionData,
                              config: RayProjectionConfig) -> None:
        """Create 3D textures for CT volume data."""
        # CT volume texture (int16)
        ct_descriptor = Metal.MTLTextureDescriptor.alloc().init()
        ct_descriptor.setTextureType_(Metal.MTLTextureType3D)
        ct_descriptor.setPixelFormat_(Metal.MTLPixelFormatR16Sint)
        ct_descriptor.setWidth_(config.volume_shape[0])
        ct_descriptor.setHeight_(config.volume_shape[1])
        ct_descriptor.setDepth_(config.volume_shape[2])
        ct_descriptor.setMipmapLevelCount_(1)
        ct_descriptor.setUsage_(Metal.MTLTextureUsageShaderRead)
        ct_descriptor.setStorageMode_(Metal.MTLStorageModePrivate)
        
        # Create staging buffer for upload
        ct_buffer_size = projection_data.ct_volume.nbytes
        ct_staging_buffer = self.device.newBufferWithBytes_length_options_(
            projection_data.ct_volume.tobytes(),
            ct_buffer_size,
            Metal.MTLResourceStorageModeShared
        )
        
        ct_texture = self.device.newTextureWithDescriptor_(ct_descriptor)
        if ct_texture is None:
            raise RuntimeError("Failed to create CT volume texture")
        
        # Upload CT data via blit encoder
        command_buffer = self.command_queue.commandBuffer()
        blit_encoder = command_buffer.blitCommandEncoder()
        
        bytes_per_row = config.volume_shape[0] * 2  # int16
        bytes_per_image = bytes_per_row * config.volume_shape[1]
        
        blit_encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_(
            ct_staging_buffer,
            0,
            bytes_per_row,
            bytes_per_image,
            Metal.MTLSizeMake(config.volume_shape[0], config.volume_shape[1], config.volume_shape[2]),
            ct_texture,
            0,
            0,
            Metal.MTLOriginMake(0, 0, 0)
        )
        
        blit_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        self.textures['ct_volume'] = ct_texture
        
        # Similarly create material ID and density textures
        # Material ID texture (float32 since uchar not supported for 3D textures)
        material_descriptor = Metal.MTLTextureDescriptor.alloc().init()
        material_descriptor.setTextureType_(Metal.MTLTextureType3D)
        material_descriptor.setPixelFormat_(Metal.MTLPixelFormatR32Float)
        material_descriptor.setWidth_(config.volume_shape[0])
        material_descriptor.setHeight_(config.volume_shape[1])
        material_descriptor.setDepth_(config.volume_shape[2])
        material_descriptor.setMipmapLevelCount_(1)
        material_descriptor.setUsage_(Metal.MTLTextureUsageShaderRead)
        material_descriptor.setStorageMode_(Metal.MTLStorageModePrivate)
        
        material_texture = self.device.newTextureWithDescriptor_(material_descriptor)
        
        # Upload material data - convert uint8 to float32
        material_float = projection_data.material_ids.astype(np.float32)
        material_staging_buffer = self.device.newBufferWithBytes_length_options_(
            material_float.tobytes(),
            material_float.nbytes,
            Metal.MTLResourceStorageModeShared
        )
        
        command_buffer = self.command_queue.commandBuffer()
        blit_encoder = command_buffer.blitCommandEncoder()
        
        blit_encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_(
            material_staging_buffer,
            0,
            config.volume_shape[0] * 4,  # float32
            config.volume_shape[0] * config.volume_shape[1] * 4,
            Metal.MTLSizeMake(config.volume_shape[0], config.volume_shape[1], config.volume_shape[2]),
            material_texture,
            0,
            0,
            Metal.MTLOriginMake(0, 0, 0)
        )
        
        blit_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        self.textures['material_ids'] = material_texture
        
        # Density texture (float32)
        density_descriptor = Metal.MTLTextureDescriptor.alloc().init()
        density_descriptor.setTextureType_(Metal.MTLTextureType3D)
        density_descriptor.setPixelFormat_(Metal.MTLPixelFormatR32Float)
        density_descriptor.setWidth_(config.volume_shape[0])
        density_descriptor.setHeight_(config.volume_shape[1])
        density_descriptor.setDepth_(config.volume_shape[2])
        density_descriptor.setMipmapLevelCount_(1)
        density_descriptor.setUsage_(Metal.MTLTextureUsageShaderRead)
        density_descriptor.setStorageMode_(Metal.MTLStorageModePrivate)
        
        density_texture = self.device.newTextureWithDescriptor_(density_descriptor)
        
        # Upload density data
        density_staging_buffer = self.device.newBufferWithBytes_length_options_(
            projection_data.densities.tobytes(),
            projection_data.densities.nbytes,
            Metal.MTLResourceStorageModeShared
        )
        
        command_buffer = self.command_queue.commandBuffer()
        blit_encoder = command_buffer.blitCommandEncoder()
        
        blit_encoder.copyFromBuffer_sourceOffset_sourceBytesPerRow_sourceBytesPerImage_sourceSize_toTexture_destinationSlice_destinationLevel_destinationOrigin_(
            density_staging_buffer,
            0,
            config.volume_shape[0] * 4,  # float32
            config.volume_shape[0] * config.volume_shape[1] * 4,
            Metal.MTLSizeMake(config.volume_shape[0], config.volume_shape[1], config.volume_shape[2]),
            density_texture,
            0,
            0,
            Metal.MTLOriginMake(0, 0, 0)
        )
        
        blit_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        self.textures['densities'] = density_texture
        
        logger.debug("Created 3D textures for volume data")
    
    def _create_constant_buffers(self,
                               projection_data: ProjectionData,
                               config: RayProjectionConfig) -> None:
        """Create constant buffers for LUTs and spectrum."""
        # Flatten material LUT for GPU
        num_materials = 5
        material_lut_flat = np.zeros((num_materials * config.num_energy_bins,), dtype=np.float32)
        
        for mat_id in range(num_materials):
            if mat_id in projection_data.material_luts:
                start_idx = mat_id * config.num_energy_bins
                end_idx = start_idx + config.num_energy_bins
                material_lut_flat[start_idx:end_idx] = projection_data.material_luts[mat_id]
        
        # Material LUT buffer
        material_lut_buffer = self.device.newBufferWithBytes_length_options_(
            material_lut_flat.tobytes(),
            material_lut_flat.nbytes,
            Metal.MTLResourceStorageModeManaged
        )
        self.buffers['material_lut'] = material_lut_buffer
        
        # Source spectrum buffer
        spectrum_buffer = self.device.newBufferWithBytes_length_options_(
            projection_data.source_spectrum.tobytes(),
            projection_data.source_spectrum.nbytes,
            Metal.MTLResourceStorageModeManaged
        )
        self.buffers['spectrum'] = spectrum_buffer
        
        # Uniforms buffers (per view)
        uniforms_size = 256  # Padded size for alignment
        uniforms_buffer_left = self.device.newBufferWithLength_options_(
            uniforms_size,
            Metal.MTLResourceStorageModeShared
        )
        uniforms_buffer_right = self.device.newBufferWithLength_options_(
            uniforms_size,
            Metal.MTLResourceStorageModeShared
        )
        self.buffers['uniforms_left'] = uniforms_buffer_left
        self.buffers['uniforms_right'] = uniforms_buffer_right
        
        # Debug statistics buffer
        # Format: [0] = total rays, [1] = rays hit, [2] = rays missed
        # [3-18] = debug values for center pixel (extended for volume bounds)
        debug_stats_size = 32 * 4  # 32 uint32 values
        debug_stats_buffer = self.device.newBufferWithLength_options_(
            debug_stats_size,
            Metal.MTLResourceStorageModeShared
        )
        # Initialize to zero
        debug_data = np.zeros(32, dtype=np.uint32)
        debug_stats_buffer.contents().as_buffer(debug_stats_size)[:] = debug_data.tobytes()
        self.buffers['debug_stats'] = debug_stats_buffer
        
        logger.debug("Created constant buffers and debug statistics buffer")
    
    def _create_output_textures(self, config: RayProjectionConfig) -> None:
        """Create output textures for DRRs."""
        # Output texture descriptor
        output_descriptor = Metal.MTLTextureDescriptor.alloc().init()
        output_descriptor.setTextureType_(Metal.MTLTextureType2D)
        output_descriptor.setPixelFormat_(Metal.MTLPixelFormatR32Float)
        output_descriptor.setWidth_(config.detector_size[0])
        output_descriptor.setHeight_(config.detector_size[1])
        output_descriptor.setMipmapLevelCount_(1)
        output_descriptor.setUsage_(Metal.MTLTextureUsageShaderWrite | Metal.MTLTextureUsageShaderRead)
        output_descriptor.setStorageMode_(Metal.MTLStorageModePrivate)
        
        # Create textures for left and right views
        left_texture = self.device.newTextureWithDescriptor_(output_descriptor)
        right_texture = self.device.newTextureWithDescriptor_(output_descriptor)
        
        self.textures['drr_left'] = left_texture
        self.textures['drr_right'] = right_texture
        
        logger.debug("Created output textures for DRRs")
    
    def encode_projection(self,
                        command_buffer: Any,
                        view: str,
                        config: RayProjectionConfig,
                        uniforms: ProjectionUniforms) -> None:
        """Encode projection commands for one view.
        
        Args:
            command_buffer: Metal command buffer
            view: 'left' or 'right'
            config: Ray projection configuration
            uniforms: Projection uniforms for this view
        """
        if config.enable_tiling:
            self._encode_tiled_projection(command_buffer, view, config, uniforms)
        else:
            self._encode_full_projection(command_buffer, view, config, uniforms)
    
    def _encode_full_projection(self,
                              command_buffer: Any,
                              view: str,
                              config: RayProjectionConfig,
                              uniforms: ProjectionUniforms) -> None:
        """Encode projection without tiling (original implementation)."""
        # Create compute command encoder
        encoder = command_buffer.computeCommandEncoder()
        if encoder is None:
            raise RuntimeError("Failed to create compute command encoder")
        
        encoder.setLabel_(f"RayProjection_{view}")
        
        # Set compute pipeline state
        encoder.setComputePipelineState_(self.pipeline_state)
        
        # Set textures
        encoder.setTexture_atIndex_(self.textures['ct_volume'], 0)
        encoder.setTexture_atIndex_(self.textures['material_ids'], 1)
        encoder.setTexture_atIndex_(self.textures['densities'], 2)
        encoder.setTexture_atIndex_(self.textures[f'drr_{view}'], 3)
        
        # Set buffers
        encoder.setBuffer_offset_atIndex_(self.buffers[f'uniforms_{view}'], 0, 0)
        encoder.setBuffer_offset_atIndex_(self.buffers['spectrum'], 0, 1)
        encoder.setBuffer_offset_atIndex_(self.buffers['material_lut'], 0, 2)
        encoder.setBuffer_offset_atIndex_(self.buffers['debug_stats'], 0, 3)
        
        # Calculate thread groups
        threads_per_group = Metal.MTLSizeMake(
            config.threads_per_group[0],
            config.threads_per_group[1],
            1
        )
        
        thread_groups = Metal.MTLSizeMake(
            (config.detector_size[0] + threads_per_group.width - 1) // threads_per_group.width,
            (config.detector_size[1] + threads_per_group.height - 1) // threads_per_group.height,
            1
        )
        
        # Dispatch threads
        encoder.dispatchThreadgroups_threadsPerThreadgroup_(
            thread_groups, threads_per_group
        )
        
        # End encoding
        encoder.endEncoding()
        
        logger.debug("Encoded projection for %s view", view)
    
    def _encode_tiled_projection(self,
                               command_buffer: Any,
                               view: str,
                               config: RayProjectionConfig,
                               uniforms: ProjectionUniforms) -> None:
        """Encode projection with tiling to manage GPU workload."""
        detector_width, detector_height = config.detector_size
        tile_width, tile_height = config.tile_size
        
        # Calculate number of tiles
        num_tiles_x = (detector_width + tile_width - 1) // tile_width
        num_tiles_y = (detector_height + tile_height - 1) // tile_height
        total_tiles = num_tiles_x * num_tiles_y
        
        logger.debug(f"Encoding tiled projection for {view}: {num_tiles_x}x{num_tiles_y} = {total_tiles} tiles")
        
        # Track buffers to return to pool
        used_tile_buffers = []
        
        # Process tiles in batches
        tile_idx = 0
        while tile_idx < total_tiles:
            # Determine how many tiles to process in this batch
            batch_size = min(config.max_concurrent_tiles, total_tiles - tile_idx)
            
            # Create encoder for this batch
            encoder = command_buffer.computeCommandEncoder()
            if encoder is None:
                raise RuntimeError("Failed to create compute command encoder")
            
            encoder.setLabel_(f"RayProjection_{view}_batch{tile_idx//config.max_concurrent_tiles}")
            
            # Process tiles in this batch
            for i in range(batch_size):
                current_tile = tile_idx + i
                tile_y = current_tile // num_tiles_x
                tile_x = current_tile % num_tiles_x
                
                # Calculate tile bounds
                tile_start_x = tile_x * tile_width
                tile_start_y = tile_y * tile_height
                tile_end_x = min(tile_start_x + tile_width, detector_width)
                tile_end_y = min(tile_start_y + tile_height, detector_height)
                
                actual_tile_width = tile_end_x - tile_start_x
                actual_tile_height = tile_end_y - tile_start_y
                
                # Skip empty tiles
                if actual_tile_width <= 0 or actual_tile_height <= 0:
                    continue
                
                # Set compute pipeline state
                encoder.setComputePipelineState_(self.pipeline_state)
                
                # Set textures
                encoder.setTexture_atIndex_(self.textures['ct_volume'], 0)
                encoder.setTexture_atIndex_(self.textures['material_ids'], 1)
                encoder.setTexture_atIndex_(self.textures['densities'], 2)
                encoder.setTexture_atIndex_(self.textures[f'drr_{view}'], 3)
                
                # Set buffers
                encoder.setBuffer_offset_atIndex_(self.buffers[f'uniforms_{view}'], 0, 0)
                encoder.setBuffer_offset_atIndex_(self.buffers['spectrum'], 0, 1)
                encoder.setBuffer_offset_atIndex_(self.buffers['material_lut'], 0, 2)
                encoder.setBuffer_offset_atIndex_(self.buffers['debug_stats'], 0, 3)
                
                # Get tile offset buffer from pool
                tile_offset_buffer = self._get_tile_offset_buffer(tile_start_x, tile_start_y)
                encoder.setBuffer_offset_atIndex_(tile_offset_buffer, 0, 4)
                used_tile_buffers.append(tile_offset_buffer)
                
                # Calculate thread groups for this tile
                threads_per_group = Metal.MTLSizeMake(
                    min(config.threads_per_group[0], actual_tile_width),
                    min(config.threads_per_group[1], actual_tile_height),
                    1
                )
                
                thread_groups = Metal.MTLSizeMake(
                    (actual_tile_width + threads_per_group.width - 1) // threads_per_group.width,
                    (actual_tile_height + threads_per_group.height - 1) // threads_per_group.height,
                    1
                )
                
                # Dispatch threads for this tile
                logger.debug(f"Dispatching tile ({tile_x},{tile_y}): "
                           f"tile_bounds=[{tile_start_x},{tile_start_y}]-[{tile_end_x},{tile_end_y}], "
                           f"threads_per_group={threads_per_group.width}x{threads_per_group.height}, "
                           f"thread_groups={thread_groups.width}x{thread_groups.height}")
                encoder.dispatchThreadgroups_threadsPerThreadgroup_(
                    thread_groups, threads_per_group
                )
            
            # End encoding for this batch
            encoder.endEncoding()
            
            # Add a synchronization point between batches
            if tile_idx + batch_size < total_tiles:
                # Create a blit encoder for synchronization
                blit_encoder = command_buffer.blitCommandEncoder()
                blit_encoder.setLabel_(f"TileSync_{view}_after_batch{tile_idx//config.max_concurrent_tiles}")
                blit_encoder.endEncoding()
            
            tile_idx += batch_size
        
        # Return used buffers to pool after encoding is complete
        for buffer in used_tile_buffers:
            self._return_tile_offset_buffer(buffer)
        
        logger.debug(f"Completed tiled encoding for {view} view, returned {len(used_tile_buffers)} buffers to pool")
    
    def clear_debug_stats(self) -> None:
        """Clear debug statistics buffer before projection."""
        debug_buffer = self.buffers['debug_stats']
        debug_data = np.zeros(32, dtype=np.uint32)
        debug_buffer.contents().as_buffer(32 * 4)[:] = debug_data.tobytes()
    
    def read_debug_stats(self) -> Dict[str, Any]:
        """Read debug statistics from GPU buffer."""
        debug_buffer = self.buffers['debug_stats']
        debug_data = np.frombuffer(
            debug_buffer.contents().as_buffer(32 * 4),
            dtype=np.uint32
        )
        
        stats = {
            'total_rays': int(debug_data[0]),
            'rays_hit': int(debug_data[1]),
            'rays_missed': int(debug_data[2]),
            'hit_rate': float(debug_data[1]) / float(debug_data[0]) if debug_data[0] > 0 else 0.0,
            'center_pixel': {
                'ray_origin_world': (
                    struct.unpack('f', debug_data[3].tobytes())[0] if debug_data[3] > 0 else 0.0,
                    struct.unpack('f', debug_data[4].tobytes())[0] if debug_data[4] > 0 else 0.0,
                    0.0  # Z not stored
                ),
                'ray_origin_voxel': (
                    struct.unpack('f', debug_data[5].tobytes())[0] if debug_data[5] > 0 else 0.0,
                    struct.unpack('f', debug_data[6].tobytes())[0] if debug_data[6] > 0 else 0.0,
                    struct.unpack('f', debug_data[7].tobytes())[0] if debug_data[7] > 0 else 0.0
                ),
                'ray_dir_voxel': (
                    struct.unpack('f', debug_data[8].tobytes())[0] if debug_data[8] > 0 else 0.0,
                    struct.unpack('f', debug_data[9].tobytes())[0] if debug_data[9] > 0 else 0.0,
                    struct.unpack('f', debug_data[10].tobytes())[0] if debug_data[10] > 0 else 0.0
                ),
                'intersection_tMin': struct.unpack('f', debug_data[11].tobytes())[0] if debug_data[11] > 0 else 0.0,
                'intersection_tMax': struct.unpack('f', debug_data[12].tobytes())[0] if debug_data[12] > 0 else 0.0,
                'volume_min': (
                    struct.unpack('f', debug_data[13].tobytes())[0] if debug_data[13] > 0 else 0.0,
                    struct.unpack('f', debug_data[14].tobytes())[0] if debug_data[14] > 0 else 0.0,
                    struct.unpack('f', debug_data[15].tobytes())[0] if debug_data[15] > 0 else 0.0
                ),
                'volume_max': (
                    struct.unpack('f', debug_data[16].tobytes())[0] if debug_data[16] > 0 else 0.0,
                    struct.unpack('f', debug_data[17].tobytes())[0] if debug_data[17] > 0 else 0.0,
                    struct.unpack('f', debug_data[18].tobytes())[0] if debug_data[18] > 0 else 0.0
                ),
                'volume_dimensions': (
                    int(debug_data[19]),
                    int(debug_data[20]),
                    int(debug_data[21])
                )
            }
        }
        
        return stats
    
    def read_output_texture(self, view: str, config: RayProjectionConfig) -> np.ndarray:
        """Read DRR from GPU texture.
        
        Args:
            view: 'left' or 'right'
            config: Ray projection configuration
            
        Returns:
            2D numpy array with DRR image
        """
        texture = self.textures[f'drr_{view}']
        
        # Create temporary buffer for readback
        buffer_size = config.detector_size[0] * config.detector_size[1] * 4  # float32
        readback_buffer = self.device.newBufferWithLength_options_(
            buffer_size,
            Metal.MTLResourceStorageModeShared
        )
        
        # Create blit encoder to copy texture to buffer
        command_buffer = self.command_queue.commandBuffer()
        blit_encoder = command_buffer.blitCommandEncoder()
        
        blit_encoder.copyFromTexture_sourceSlice_sourceLevel_sourceOrigin_sourceSize_toBuffer_destinationOffset_destinationBytesPerRow_destinationBytesPerImage_(
            texture,
            0,
            0,
            Metal.MTLOriginMake(0, 0, 0),
            Metal.MTLSizeMake(config.detector_size[0], config.detector_size[1], 1),
            readback_buffer,
            0,
            config.detector_size[0] * 4,
            buffer_size
        )
        
        blit_encoder.endEncoding()
        command_buffer.commit()
        command_buffer.waitUntilCompleted()
        
        # Read data from buffer
        data_ptr = readback_buffer.contents()
        data = np.frombuffer(
            data_ptr.as_buffer(buffer_size),
            dtype=np.float32
        )
        
        # Reshape to 2D image
        drr_image = data.reshape(config.detector_size[1], config.detector_size[0])
        
        return drr_image
    
    def update_stats(self, kernel_time_ms: float, view: str, config: RayProjectionConfig) -> None:
        """Update performance statistics.
        
        Args:
            kernel_time_ms: Kernel execution time in milliseconds
            view: Which view was processed
            config: Ray projection configuration
        """
        self.stats.kernel_time_ms += kernel_time_ms
        self.stats.total_rays += config.detector_size[0] * config.detector_size[1]
        
        # Estimate memory bandwidth usage
        # Rough estimate: each ray reads ~1000 voxels, 2 bytes each
        voxels_read = self.stats.total_rays * 1000
        self.stats.texture_reads_gb += voxels_read * 2 / (1024**3)
        
        logger.debug("%s view kernel time: %.2f ms", view, kernel_time_ms)