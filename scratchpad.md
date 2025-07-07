# DRR Ray-Volume Intersection Issue - Troubleshooting Scratchpad

## Problem Statement
The GPU-based DRR generation was producing empty (all-zero) outputs despite the shader executing correctly. Rays were missing the volume entirely.

## Root Cause Analysis

### 1. Initial Symptoms
- DRR outputs were all zeros
- Shader was executing (confirmed by timing: ~1000ms GPU time)
- Debug statistics showed 0% ray hits
- Uniforms buffer was not being read (all values showing as 0)

### 2. Coordinate System Mismatch Discovery

#### Shader Expectations (joseph_projector.metal)
```metal
// Original shader code expected volume centered at origin
float3 volumeSize = float3(uniforms.volumeDimensions) * uniforms.voxelSpacing;
float3 volumeMin = -volumeSize * 0.5f;  // e.g., (-16, -16, -16)
float3 volumeMax = volumeSize * 0.5f;   // e.g., (16, 16, 16)
```

#### CPU Implementation (ray_projector.py)
```python
# CPU code assumed volume starts at corner
volume_min_world = np.array([0.0, 0.0, 0.0])
volume_max_world = volume_size_mm  # e.g., (32, 32, 32)
```

#### C-arm Geometry Setup
- Source positioned at ~(-10, -99, 0) for left view
- Detector positioned at ~(5, 50, 0)
- Rays aimed at isocenter (0, 0, 0)
- But volume was positioned with corner at (0, 0, 0), not centered

### 3. Why Rays Missed
With the volume positioned at (0,0,0) to (32,32,32) and rays converging at the isocenter (0,0,0), most rays passed above or beside the volume without intersecting it.

## Solution Implementation

### Step 1: Fix Volume Centering in Shader
Changed the volume bounds calculation to center the volume at the origin:

```metal
// Fixed: Volume centered at isocenter
float3 volumeSize = float3(hardcodedVolumeDimensions) * hardcodedVoxelSpacing;
float3 volumeMin = -volumeSize * 0.5f;  // Now matches C-arm geometry
float3 volumeMax = volumeSize * 0.5f;
```

### Step 2: Update Voxel Coordinate Calculation
Adjusted the world-to-voxel transformation to account for centered volume:

```metal
// Convert world position to voxel coordinates
float3 voxelPos = (currentPosWorld - volumeMin) / hardcodedVoxelSpacing;
```

### Step 3: Hardcode Values as Workaround
Due to the uniforms buffer not being read, temporarily hardcoded critical values:

```metal
// Hardcoded values for testing
int3 hardcodedVolumeDimensions = int3(64, 64, 64);
float3 hardcodedVoxelSpacing = float3(0.5f, 0.5f, 0.5f);

// Hardcoded left view geometry
float3 rayOriginWorld = float3(-10.452847, -99.452194, 0.0);
float3 detectorOrigin = float3(-10.685927, 51.398552, -16.0);
// ... etc
```

### Step 4: Add Debug Outputs
Added debug pattern writes to verify shader execution:

```metal
// Initial debug write to confirm shader runs
detectorImage.write(25.0f, pixelCoord);

// Path length visualization
if (tMax > tMin && tMin >= 0.0f) {
    detectorImage.write((tMax - tMin) * 10.0f, pixelCoord);
}
```

## Results

### Before Fix
- 0% ray hits
- All outputs zero
- No visible DRR

### After Fix
- 50-65% ray hits (depending on volume)
- Visible attenuation values
- Path lengths showing expected patterns

### Simple Test Volume (64×64×64)
- Hit rate: 65.6%
- Output range: 0-171.86
- Clear visibility of bone cube in center

### Actual CT Volume (720×720×665)
- Hit rate: 5.1% (lower due to larger volume vs detector coverage)
- Output range: 0-284.96
- Visible anatomy structures

## Remaining Issues

### 1. Uniforms Buffer Not Being Read
The shader cannot read from the uniforms buffer, showing all zeros for:
- Source position
- Detector geometry
- Volume dimensions
- All other uniform parameters

Possible causes:
- Metal buffer alignment issues
- Struct packing mismatch between CPU and GPU
- Buffer binding incorrect

### 2. Identical Stereo Views
Both left and right DRRs are identical because we hardcoded the left view geometry for both.

### 3. Low Hit Rate with Large Volumes
The 5.1% hit rate with actual CT suggests the volume might need better positioning relative to the detector field of view.

## Key Learnings

1. **Coordinate System Consistency**: Always verify that all components (geometry setup, CPU reference, GPU implementation) use the same coordinate system conventions.

2. **Debug Visualization**: Writing debug patterns (like path lengths) is invaluable for understanding what the shader is actually doing.

3. **Incremental Testing**: Starting with a simple test volume (64×64×64) made it much easier to identify the coordinate system issue.

4. **Metal Debugging Challenges**: The inability to read uniforms made debugging much harder - having fallback hardcoded values was essential.

## Next Steps

1. **Fix Uniforms Buffer**:
   - Investigate Metal struct alignment requirements
   - Try different buffer packing strategies
   - Consider using individual buffers instead of a struct

2. **Implement Proper Stereo**:
   - Pass view-specific geometry once uniforms work
   - Or create separate shader entry points for left/right

3. **Optimize Volume Positioning**:
   - Ensure volume is properly centered at isocenter
   - Adjust geometry for better detector coverage

4. **Remove Hardcoded Values**:
   - Once uniforms work, remove all hardcoded geometry
   - Make the implementation fully data-driven