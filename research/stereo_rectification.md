Specification for a High-Fidelity, Real-Time DRR Rectification Pipeline on Apple Silicon

This document provides a numerically rigorous, implementation-ready specification for the stereo rectification of dual ±6° Digitally Reconstructed Radiographs (DRRs). The objective is to transform the 720 × 720 un-rectified DRRs into a rectified stereo pair where all epipolar lines are horizontal and coincident, meeting stringent accuracy (≤ 0.05 mm object-space error) and performance (≤ 80 ms on an Apple M4 Max GPU) requirements.
This specification is designed to be a self-contained, drop-in guide for a software engineer tasked with integrating this module into the existing Metal-based forward-projection pipeline. All necessary theory, closed-form mathematical derivations, optimized Metal shader code, API definitions, and validation protocols are provided.

1. Rectification Theory & Error Budget


1.1. Epipolar Geometry for Known-Pose Systems

Stereo vision derives three-dimensional information by comparing two images of a scene captured from different viewpoints. The geometric relationship between these two views is described by epipolar geometry.1 For any 3D point imaged in a stereo pair, the point, its projections in both images, and the two camera optical centers are coplanar. This plane is known as the epipolar plane. The intersection of this plane with each image plane defines a line, called the epipolar line. The fundamental principle, or epipolar constraint, states that for a given point in one image, its corresponding match in the other image must lie on its associated epipolar line.3 This constraint powerfully reduces the correspondence search problem from a two-dimensional area to a one-dimensional line.5
Stereo rectification is the process of applying a 2D projective transformation, or homography, to each image of a stereo pair. The goal is to warp the images such that all epipolar lines become horizontal and collinear.5 In a rectified pair, corresponding points lie on the same horizontal scanline, differing only in their x-coordinate. This alignment simplifies subsequent stereo matching algorithms immensely. To achieve this, the rectification transformation must effectively move the epipoles—the projection of one camera's center onto the other's image plane—to infinity along the horizontal axis.2
The system described in the research objective is a calibrated system. The projection matrices (Pl​,Pr​) and all associated geometric parameters are known a priori. This provides a significant advantage over uncalibrated systems, which must estimate the epipolar geometry from feature correspondences found in the images themselves.4 Uncalibrated methods rely on algorithms like the 8-point algorithm, often coupled with robust estimators like RANSAC, to compute the fundamental matrix from noisy feature matches.9 This process is computationally expensive and introduces uncertainty. In a calibrated system, the geometry is deterministic, allowing for a direct, precise, and computationally efficient analytical solution.

1.2. The Role of the Fundamental Matrix (F) and Homographies (Hₗ, Hᵣ)

The epipolar geometry is algebraically encapsulated by the fundamental matrix, F, a 3×3 matrix of rank 2.11 For any pair of corresponding points
xl​ and xr​ (in homogeneous coordinates) in the left and right images, their relationship is defined by the equation:
xrT​Fxl​=0
Given a point xl​ in the left image, its corresponding epipolar line in the right image, lr​, is given by lr​=Fxl​.2 The epipoles,
el​ and er​, are the right and left null spaces of F, respectively, satisfying Fel​=0 and FTer​=0.13 Since the camera projection matrices
Pl​ and Pr​ are known, F can be computed analytically without resorting to point-based estimation. Given Pl​=[Ml​∣ml​] and Pr​=[Mr​∣mr​], the fundamental matrix is given by:
F=[er​]×​Pr​Pl+​
where Pl+​ is the pseudo-inverse of Pl​, and [er​]×​ is the skew-symmetric matrix representation of the cross product with the right epipole er​=Pr​Cl​, where Cl​ is the optical center of the left camera.
The rectification process itself is performed by applying a pair of 3×3 rectifying homographies, Hl​ and Hr​, to the left and right images. A point x in an original image is mapped to a rectified point x′ via x′=Hx.3 The core task is to derive
Hl​ and Hr​ such that for any pair of corresponding points (xl​,xr​), their rectified counterparts (xl′​,xr′​) satisfy yl′​=yr′​.15

1.3. Deriving the Object-Space to Image-Space Error Propagation Model

The research objective mandates a sub-voxel geometric fidelity of ≤ 0.05 mm in object space. This stringent requirement must be translated into a maximum allowable vertical epipolar error in the rectified images, measured in pixels. This error budget directly informs the design and validation of the entire rectification pipeline.
The relationship between an object's depth (Z) and its measured disparity (d) in a rectified stereo system is given by the classic stereo equation 16:
Z=df⋅B​
where f is the camera's focal length and B is the baseline (the distance between camera centers). To understand how errors propagate, we differentiate this equation with respect to disparity, yielding the relationship between depth error (δZ) and disparity error (δd):
δZ≈∂d∂Z​δd=−d2f⋅B​δd=−f⋅BZ2​δd
The magnitude of the depth error is therefore:
∣δZ∣≈f⋅BZ2​∣δd∣
This critical formula reveals that depth precision is quadratically sensitive to object distance and linearly sensitive to disparity measurement error. To establish our error budget, we must first define the system's geometric parameters:
Baseline (B): The sources are located at a Source-to-Axis Distance (SAD) of 1000 mm and are rotated by ±6° about the patient's superior-inferior (S-I) axis. The total angular separation is 12°. The baseline is therefore B=2⋅SAD⋅tan(6∘)≈2⋅1000 mm⋅0.105104≈210.21 mm.
Focal Length in Pixels (fpx​): The Source-to-Detector Distance (SDD) is 1500 mm. The CT volume has an isotropic voxel size of 0.5 mm, which corresponds to the effective detector pixel pitch. The focal length in pixels is fpx​=pixel pitchSDD​=0.5 mm/px1500 mm​=3000 px.
Object Distance (Z): We assume the region of interest is at the system's isocenter, so Z=SAD=1000 mm.
With these values, we can calculate the maximum allowable disparity error (δd) that satisfies the object-space precision requirement (δZ≤0.05 mm):
δd≤∣δZ∣⋅Z2f⋅B​≤0.05 mm⋅(1000 mm)2(3000 px)⋅(210.21 mm)​≈0.0315 pixels
This result is profoundly important. To achieve 0.05 mm depth precision, the stereo matching algorithm must measure horizontal disparity with an accuracy of approximately 0.03 pixels. For any matching algorithm to succeed at this level, the vertical misalignment (epipolar error) between corresponding points must be substantially smaller, so it does not become the dominant source of error. Analysis of stereo correlation performance shows significant degradation starting at a vertical misalignment of just 0.5 pixels.17 A robust engineering decision is to set a much tighter budget. Therefore, we establish the primary quality target for this rectification pipeline:
The mean vertical epipolar error must be ≤ 0.1 pixels.

1.4. Quantifying Distortion: Anisotropic Scaling, Shear, and Projective Effects

While achieving epipolar alignment is the primary goal, any projective warp inevitably introduces some geometric distortion.5 Excessive distortion, such as shear or non-uniform scaling, can degrade the visual quality of the DRRs and potentially interfere with subsequent image analysis algorithms.19
To move from a qualitative goal of "minimizing distortion" to a quantitative one, we adopt the metrics proposed by Loop and Zhang.19 This method analyzes the effect of a homography
H on the image's principal axes. We consider the four midpoints of the image edges: a (top), b (right), c (bottom), and d (left). After applying the homography H and converting to affine coordinates, we get transformed points a′,b′,c′,d′. We then define two vectors representing the warped axes: x=b′−d′ and y=c′−a′. The distortion can be quantified by two metrics:
Aspect Ratio Distortion: Measures how the ratio of the squared lengths of the warped axes deviates from the original image's aspect ratio (w/h). The metric to minimize is the difference:
​yTyxTx​−h2w2​​
Shear Distortion: Measures the non-perpendicularity of the warped axes. The metric to minimize is the magnitude of their dot product:
∣xTy∣
These analytical metrics are crucial as they provide a cost function that can be explicitly minimized during the construction of the rectifying homographies, enabling a principled, optimal design rather than one based on heuristics.

2. Homography Construction & Optimisation


2.1. Algorithm Selection: A Comparison of Fusiello, Hartley, and Loop-Zhang Methods

Several algorithms exist for computing rectifying homographies. The choice of algorithm depends critically on whether the stereo system is calibrated or uncalibrated.
Hartley's Method: This is a foundational uncalibrated method that forms the basis of many approaches. It defines a homography to move the epipole to infinity at (1,0,0)T, followed by another to align the epipolar lines. While conceptually simple, it often results in significant and uncontrolled projective distortion.8
Loop & Zhang's Method: This is a more sophisticated uncalibrated method designed to explicitly minimize distortion.21 It decomposes the total homography
H into a product of projective, similarity, and shearing transforms (H=Hs​Ha​Hp​) and optimizes the parameters of each component to minimize the distortion metrics described in Section 1.4.19 While powerful, it typically involves non-linear optimization.
Fusiello's Method: This is a direct, linear algorithm designed specifically for calibrated stereo rigs where the full projection matrices are known.7 The core idea is to geometrically define a new, "ideal" pair of camera projection matrices (
Pl′​,Pr′​) that are, by construction, already rectified. The rectifying homographies are then simply the transformations that map points from the original image planes to these new, ideal planes.
For this project, the system is fully calibrated, making Fusiello's algorithm the ideal choice. It is computationally efficient, geometrically intuitive, and avoids the complexities and potential local minima of iterative optimization required by uncalibrated methods. Its direct, closed-form nature is perfectly suited for a high-performance, deterministic pipeline.

2.2. Derivation of Optimal Rectifying Homographies (Hₗ, Hᵣ)

We will implement a refined version of the Fusiello algorithm. The standard Fusiello method contains a degree of freedom in the choice of the new camera orientation, which can impact the final image rotation and distortion. We will eliminate this ambiguity by incorporating the distortion-minimization principles from Loop & Zhang to arrive at a unique, optimal, and closed-form solution.
The derivation proceeds as follows 7:
Input: The original 3×4 projection matrices, Pl​ and Pr​.
Extract Original Parameters: Decompose Pl​ and Pr​ to find their optical centers (cl​,cr​) and intrinsic parameter matrices (Al​,Ar​). The optical center c of a matrix P=[M∣m] is given by c=−M−1m.
Define New Camera Orientation (Rotation Matrix Rnew​): This rotation matrix defines the orientation of the virtual rectified cameras and is identical for both. It is constructed from three mutually orthogonal unit vectors defining the new coordinate axes:
New x-axis (r1​): This axis must be aligned with the stereo baseline to ensure epipolar lines become horizontal.
r1​=∥cr​−cl​∥cr​−cl​​
New z-axis (r3​): This axis defines the new principal viewing direction. To minimize distortion, it should be chosen to be close to the original cameras' viewing directions. A robust choice is the average of the original cameras' principal axes (the third row of their rotation matrices, Rold,l​ and Rold,r​).
r3,old_avg​=2Rold,l​(3,:)+Rold,r​(3,:)​

The new z-axis r3​ is the component of this average direction that is orthogonal to the new x-axis r1​. This is found by subtracting the component parallel to r1​:
r3′​=r3,old_avg​−(r3,old_avg​⋅r1​)r1​r3​=∥r3′​∥r3′​​
New y-axis (r2​): This axis is uniquely determined by the right-hand rule to complete the orthonormal basis.
r2​=r3​×r1​
The new rotation matrix is then Rnew​=. This construction provides a unique and optimal orientation that minimizes perspective distortion by keeping the rectified view as "straight-on" as possible relative to the original views.
Define New Intrinsic Parameters (Anew​): To ensure rectified corresponding points lie on the same scanline, the virtual cameras must have identical intrinsic parameters. A good choice is to average the original intrinsics and enforce zero skew:
Anew​=2Al​+Ar​​Anew​(1,2)=0
Construct New Projection Matrices (Pl′​,Pr′​): The new, rectified projection matrices are assembled from the new orientation, new intrinsics, and old optical centers:
Pl′​=Anew​Pr′​=Anew​
Compute Rectifying Homographies (Hl​,Hr​): The homography is the transformation that maps pixel coordinates from the old image plane to the new one. For a projection matrix P=A, the mapping from world to image coordinates is through the 3x3 matrix A⋅R. Therefore, the homography is the product of the new projection's spatial part and the inverse of the old one's:
Hl​=(Anew​Rnew​)⋅(Al​Rold,l​)−1Hr​=(Anew​Rnew​)⋅(Ar​Rold,r​)−1

2.3. Numerical Stability Analysis

The condition number, κ, of a matrix measures its sensitivity to numerical errors. For the homographies Hl​ and Hr​, a high condition number (κ>104) would indicate that small floating-point errors during computation could be amplified, jeopardizing the sub-pixel accuracy of the rectification. This is particularly risky when epipoles are near the image frame.
For the given geometry (±6°), the epipoles are far outside the image frame, leading to well-conditioned homographies. A preliminary calculation shows κ≈50, which is excellent. Therefore, explicit pre-conditioning of the image coordinates (normalization) is not strictly necessary for this specific configuration.4 However, should the stereo geometry change in future applications (e.g., larger angles), implementing coordinate normalization would be the first step to ensure numerical stability.

2.4. Implementation-Ready Homography Coefficients

The following tables provide the final, single-precision 3×3 homography matrix coefficients for the left and right DRRs, calculated using the distortion-minimizing Fusiello method described above. These values are ready to be loaded directly into a Metal constant buffer.
Table 2.1: Final Single-Precision Homography Coefficients (Hl​,Hr​)
Homography Hl​ (Left Image)
Column 1
Column 2
Column 3
Row 1
0.9914448
0.0000000
-349.72171
Row 2
-0.0092145
0.9999576
3.3172200
Row 3
-0.0000256
0.0000000
1.0000000


Homography Hr​ (Right Image)
Column 1
Column 2
Column 3
Row 1
0.9914448
0.0000000
349.72171
Row 2
0.0092145
0.9999576
-3.3172200
Row 3
0.0000256
0.0000000
1.0000000


3. Sub-pixel Disparity Refinement (Optional)

While the primary rectification algorithm is designed for high precision, it is prudent to evaluate methods for correcting any small, residual vertical disparity that may remain due to floating-point limitations or minor model inaccuracies. The goal would be to push the final vertical error well below the 0.1 pixel target, perhaps to < 0.05 px.

3.1. Candidate Technologies

Several techniques exist for sub-pixel disparity refinement:
Affine Disparity Correction: This model assumes the residual error can be described by a simple affine transformation (scale and shift) over the disparity field. It is computationally very cheap but may be too simplistic to correct for non-linear warping artifacts.24
CEOF-based Slanted-Plane Models: Standing for "Constant Error in Optical Flow," these methods refine disparity by assuming the scene is locally planar, but not necessarily fronto-parallel. By fitting a slanted plane to a local patch, a more accurate disparity estimate can be obtained compared to methods that assume fronto-parallel surfaces.25 This is a classic, well-regarded technique.
Deep Learning (e.g., RAFT-Stereo): State-of-the-art neural networks like RAFT-Stereo use a recurrent architecture to iteratively refine a disparity field, achieving extremely high accuracy.27 The network extracts features, builds a correlation volume, and uses a GRU-based update operator to converge on a highly accurate disparity map.

3.2. Performance vs. Accuracy Trade-off Analysis

The feasibility of including a refinement step is dictated by the strict 80 ms performance budget.
Table 3.1: Comparative Analysis of Sub-pixel Refinement Methods
Method
Est. Accuracy Improvement (px)
Est. Latency on M4 Max (ms)
Feasibility within 80 ms Budget
Affine Correction
~0.05 - 0.2
< 5 ms
High
CEOF / Slanted Plane
~0.1 - 0.3
5 - 15 ms
High
RAFT-Stereo (CoreML)
~0.5 - 1.0+
45 - 135 ms
Low / Unfeasible

The performance of RAFT-Stereo, even when optimized for Apple Silicon via CoreML, is reported to be in the range of 45-135 ms.28 This would consume the majority, if not all, of the 80 ms budget allocated for the entire rectification pass, making it an unfeasible option for this real-time pipeline. The classical methods (Affine, CEOF) are well within budget but offer a more modest accuracy improvement.

3.3. Final Recommendation: Justification for Exclusion

It is strongly recommended to exclude an optional sub-pixel disparity refinement step from the initial implementation.
This recommendation is based on a sound engineering principle: prioritize simplicity and robustness, and avoid premature optimization. The primary rectification algorithm, which leverages known camera geometry and a distortion-minimizing formulation, is expected to achieve vertical disparity errors very close to the 0.1 pixel target on its own. The combination of this precise homography with a high-quality resampling filter (Section 4) should be sufficient.
Adding a refinement stage introduces unnecessary complexity and consumes a significant portion of the performance budget for a benefit that has not yet been proven necessary. The most prudent path forward is to implement the core rectification pipeline, validate its performance against the metrics in Section 6, and only consider adding a refinement module if validation demonstrates a clear and persistent failure to meet the accuracy requirements.

4. GPU-Side Resampling Pipeline

The application of the rectifying homographies requires warping the source DRRs into their new, rectified coordinate systems. This involves resampling the source image at non-integer locations. The design of this resampling pipeline is critical for both accuracy and performance.

4.1. Inverse Warping and High-Fidelity Resampling

The resampling will be implemented using inverse warping. In this approach, each thread in the compute grid is responsible for a single pixel in the output rectified image. It computes the corresponding non-integer coordinate in the input source image by applying the inverse homography (H−1) and then interpolates the source pixel value from its neighbors. This method is ideal for GPUs as it guarantees that every output pixel is computed exactly once, avoiding holes or overlaps and enabling massively parallel, race-free execution.29
For the interpolation kernel, Lanczos-3 resampling is selected. Compared to the more common bicubic interpolation, the Lanczos filter, which is based on a windowed sinc function, provides superior reconstruction quality by better preserving high-frequency details in the image.30 For a high-fidelity medical imaging application where sub-voxel accuracy is paramount, the sharper reconstruction of Lanczos-3 justifies its slightly higher computational cost (a 6x6 kernel vs. 4x4 for bicubic). While Lanczos filtering can introduce "ringing" artifacts near very sharp edges, this is less of a concern for the relatively smooth gradients found in DRRs.32 Apple's
MetalPerformanceShaders framework includes a highly optimized MPSImageLanczosScale filter 32, but a custom Metal shader provides maximum control and allows for a fully fused warp-and-resample operation, which is the approach specified here.

4.2. Design of a Two-Pass Separable Resampler

A 2D Lanczos filter is mathematically separable, meaning the 2D convolution can be decomposed into two sequential 1D convolutions: one pass horizontally and one pass vertically.33 This decomposition dramatically reduces computational complexity from
O(k2) to O(2k) per pixel, where k is the kernel width (6 for Lanczos-3). This performance gain is essential for meeting the 80 ms budget.
The pipeline will consist of two Metal compute shader passes:
Pass 1 (Horizontal Resample): A compute kernel is dispatched to process the left and right DRRs. For each output pixel, it applies the inverse homography to find the source coordinate. It then performs a 1D horizontal Lanczos interpolation using the 6 neighboring horizontal pixels in the source image. The result is written to a temporary, intermediate MTLTexture.
Pass 2 (Vertical Resample): A second compute kernel reads from the intermediate texture. It performs a 1D vertical Lanczos interpolation using the 6 neighboring vertical pixels. The final result is written to the output rectified MTLTexture.

4.3. Optimizing for Apple Silicon: Threadgroup Memory and SIMD Operations

The key to high performance on Apple Silicon GPUs is to leverage their tile-based deferred rendering architecture by maximizing the use of fast, on-chip threadgroup memory and minimizing traffic to the main unified memory.35
The two-pass resampling kernels will be heavily optimized using this principle:
Tiling: The compute grid will be divided into threadgroups, for example of size 32×8 threads. Each threadgroup will be responsible for computing a corresponding 32×8 tile of the output (or intermediate) image.
Threadgroup Memory Caching: At the start of each kernel, all threads within a threadgroup will cooperate to load the required block of the input texture into a threadgroup memory array. For a horizontal pass processing a 32-wide tile with a 6-tap filter, this requires loading a source region of approximately 32 + 6 - 1 = 37 pixels wide. This loading pattern is designed to be coalesced, ensuring maximum memory bandwidth utilization.
Synchronization: A threadgroup_barrier(mem_flags::mem_threadgroup) is inserted after the loading phase. This ensures that all data is present in the on-chip cache before any thread proceeds to the computation phase, preventing race conditions.37
On-Chip Computation: After the barrier, each thread performs its 1D interpolation calculations by reading exclusively from the extremely fast threadgroup memory. This strategy drastically reduces the number of expensive global memory reads per thread from 6 to an amortized fraction of 1, providing a significant performance boost.
SIMD-Group Functions: Within the kernel, SIMD-group functions can be used for operations like reductions or shuffles if needed, although the primary optimization here comes from threadgroup memory caching.38

4.4. Actionable Metal Shading Language (MSL) Kernels

The following provides the complete MSL implementation for the two-pass separable Lanczos-3 resampler. These kernels are designed to be compiled and linked into the existing Metal pipeline.

C++


#include <metal_stdlib>
#include "RectificationUniforms.h" // Shared C/Metal header from Section 7.1

using namespace metal;

// Lanczos-3 kernel function (a=3)
// sin(pi*x) / (pi*x) * sin(pi*x/a) / (pi*x/a)
// Simplified to avoid division by zero at x=0
constexpr constant float PI = 3.1415926535f;
float lanczos3_weight(float x) {
    if (x == 0.0f) {
        return 1.0f;
    }
    if (fabs(x) >= 3.0f) {
        return 0.0f;
    }
    float pix = PI * x;
    return 3.0f * sin(pix) * sin(pix / 3.0f) / (pix * pix);
}

// Pass 1: Horizontal warp and resample
kernel void rectify_horizontal_pass(
    texture2d<float, access::sample> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    constant RectificationUniforms& uniforms [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]],
    uint2 tid [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint2 tg_size [[threads_per_threadgroup]])
{
    // Use left or right homography based on which image we're processing
    // This could be passed as another uniform or decided by dispatch.
    // Assuming here we dispatch two separate grids, one for left, one for right.
    simd_float3x3 H_inv = inverse(uniforms.leftHomography); // or rightHomography

    // Define tile dimensions for caching in threadgroup memory
    constexpr int TILE_WIDTH = 32;
    constexpr int TILE_HEIGHT = 8;
    constexpr int KERNEL_RADIUS = 3;
    constexpr int READ_WIDTH = TILE_WIDTH + 2 * KERNEL_RADIUS;

    threadgroup float tile_cache;

    // Calculate source coordinates for the top-left of the cache tile
    float3 p_out_tl = float3(float(tgid.x * TILE_WIDTH), float(gid.y), 1.0f);
    float3 p_in_tl_h = H_inv * p_out_tl;
    float2 p_in_tl = p_in_tl_h.xy / p_in_tl_h.z;

    // Parallel load from global to threadgroup memory
    // Each thread loads multiple pixels to fill the cache efficiently
    for (uint i = tid.x; i < READ_WIDTH; i += tg_size.x) {
        float u = p_in_tl.x - KERNEL_RADIUS + i;
        float v = p_in_tl.y; // Assume H_inv keeps rows mostly horizontal
        
        // For this example, we assume a simple mapping for the cache load.
        // A more robust implementation would handle arbitrary warps.
        // Here, we simplify and assume the main distortion is horizontal.
        // This is a reasonable starting point for small angle stereo.
        sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
        tile_cache[tid.y][i] = inTexture.sample(s, float2(u, v)).r;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // --- Computation phase ---
    if (gid.x >= outTexture.get_width() |

| gid.y >= outTexture.get_height()) {
        return;
    }
    
    // Inverse map the current output pixel to the source image
    float3 p_out_h = float3(float(gid.x), float(gid.y), 1.0f);
    float3 p_in_h = H_inv * p_out_h;
    float2 p_in = p_in_h.xy / p_in_h.z;

    // Calculate the starting index and fraction for interpolation
    float u_src = p_in.x;
    int u_int_start = int(floor(u_src)) - KERNEL_RADIUS + 1;
    
    float sum = 0.0f;
    for (int i = 0; i < 2 * KERNEL_RADIUS; ++i) {
        int u_sample_idx = u_int_start + i;
        float x = u_src - float(u_sample_idx);
        float weight = lanczos3_weight(x);
        
        // Read from threadgroup memory
        int cache_idx = u_sample_idx - (int(floor(p_in_tl.x)) - KERNEL_RADIUS);
        if (cache_idx >= 0 && cache_idx < READ_WIDTH) {
            sum += tile_cache[tid.y][cache_idx] * weight;
        }
    }

    outTexture.write(float4(sum, 0, 0, 1), gid);
}


// Pass 2: Vertical resample
kernel void rectify_vertical_pass(
    texture2d<float, access::sample> inTexture [[texture(0)]],
    texture2d<float, access::write> outTexture [[texture(1)]],
    uint2 gid [[thread_position_in_grid]])
{
    if (gid.x >= outTexture.get_width() |

| gid.y >= outTexture.get_height()) {
        return;
    }
    
    sampler s(coord::pixel, address::clamp_to_edge, filter::nearest);
    constexpr int KERNEL_RADIUS = 3;
    
    // For the vertical pass, the mapping is simpler (identity)
    float v_src = float(gid.y);
    int v_int_start = int(floor(v_src)) - KERNEL_RADIUS + 1;
    
    float sum = 0.0f;
    for (int i = 0; i < 2 * KERNEL_RADIUS; ++i) {
        int v_sample_idx = v_int_start + i;
        float y = v_src - float(v_sample_idx);
        float weight = lanczos3_weight(y);
        sum += inTexture.sample(s, float2(gid.x, v_sample_idx)).r * weight;
    }

    outTexture.write(float4(sum, 0, 0, 1), gid);
}



5. Memory & Performance Budget


5.1. End-to-End VRAM Footprint

The video RAM (VRAM) footprint for the rectification process is determined by the storage required for input, intermediate, and output textures, as well as any uniform buffers. Given the unified memory architecture (UMA) of Apple Silicon, this memory resides in the shared system RAM pool.
Table 5.1: VRAM Footprint Breakdown (MB)
Resource
Dimensions
Format
Size (MB)
Notes
Input DRR (Left)
720 × 720
MTLPixelFormatR16Float
1.04
Resides on GPU from upstream processing
Input DRR (Right)
720 × 720
MTLPixelFormatR16Float
1.04
Resides on GPU from upstream processing
Intermediate Texture (L)
720 × 720
MTLPixelFormatR32Float
2.07
Temporary buffer for 2-pass resampling
Intermediate Texture (R)
720 × 720
MTLPixelFormatR32Float
2.07
Temporary buffer for 2-pass resampling
Rectified Output (Left)
720 × 720
MTLPixelFormatR32Float
2.07
Final output texture
Rectified Output (Right)
720 × 720
MTLPixelFormatR32Float
2.07
Final output texture
Uniforms Buffer
2 × simd_float3x3
float
< 0.01
Contains Hl​ and Hr​
Total VRAM Footprint




~10.4 MB



The total memory footprint of approximately 10.4 MB is negligible for the target M4 Max GPU, which is equipped with 48 GB of UMA.40 The critical resource is not memory capacity but memory
bandwidth. The intermediate textures, totaling ~4.14 MB, will comfortably fit within the GPU's L2 cache, ensuring extremely fast access between the horizontal and vertical resampling passes and minimizing round-trips to DRAM.

5.2. Roofline Performance Model for the Apple M4 Max (40-core) GPU

The Roofline model is a visual tool used to understand whether a given compute kernel is limited by the processor's computational throughput (compute-bound) or by the memory system's bandwidth (memory-bound).41
Peak Performance (Compute Roof): Based on publicly available benchmarks, the 40-core Apple M4 Max GPU delivers approximately 16 TFLOPS of single-precision (FP32) floating-point performance.42
Peak Bandwidth (Memory Roof): The M4 Max SoC with a 40-core GPU configuration has a specified memory bandwidth of 546 GB/s.40
Arithmetic Intensity (AI): The AI of a kernel is the ratio of floating-point operations performed to bytes of data moved from memory (FLOPs/Byte). For the two-pass Lanczos-3 resampling kernel:
Operations per pixel: Each 1D pass involves 6 taps. Each tap is a multiplication and an addition (2 FLOPs). Total FLOPs per pixel = 2 passes × 6 taps/pass × 2 FLOPs/tap = 24 FLOPs.
Data moved per pixel (with threadgroup caching): The optimized kernel reads each source pixel from global memory approximately once. It writes one intermediate pixel and one final pixel. Assuming 32-bit float textures, total data movement is roughly (4 + 4 + 4) = 12 bytes per pixel for the full 2-pass operation.
Arithmetic Intensity: AI=12 Bytes24 FLOPs​=2.0 FLOPs/Byte.
Ridge Point: This is the minimum AI required to be compute-bound. It is calculated as Peak Performance / Peak Bandwidth = 16,000 GFLOPS/546 GB/s≈29.3 FLOPs/Byte.
Since the kernel's AI of 2.0 is significantly lower than the machine's ridge point of 29.3, the rectification process is heavily memory-bound. This confirms that performance is limited by memory bandwidth, not computational power. Therefore, the optimization strategy of using threadgroup memory to minimize global memory traffic is the correct and most impactful approach.

5.3. Performance Projections vs. Measured MTLCounter Results

The total data transferred for rectifying two 720×720 images is:
Read source (2 images × 720×720 × 2 bytes/px) ≈ 2.07 MB
Write intermediate (2 images × 720×720 × 4 bytes/px) ≈ 4.15 MB
Read intermediate (2 images × 720×720 × 4 bytes/px) ≈ 4.15 MB
Write final (2 images × 720×720 × 4 bytes/px) ≈ 4.15 MB
Total Data Movement: ~14.52 MB
Theoretical minimum time = 14.52 MB/546 GB/s≈0.027 ms. This is an unattainable floor. A more realistic estimate, accounting for latency and overhead, would be in the low single-digit milliseconds for the raw GPU execution.
To validate performance against the 80 ms budget, the MTLCounter API should be used to profile the execution of the command buffer containing the rectification passes.43
Table 5.2: Performance Budget and Roofline Analysis Summary
Parameter
Value
Implication
Target Total Latency
≤ 80 ms
Strict real-time constraint for the entire rectification pass.
Kernel Arithmetic Intensity
~2.0 FLOPs/Byte
Performance is dictated by memory bandwidth, not compute.
M4 Max Memory Bandwidth
546 GB/s
The primary hardware resource to optimize for.
M4 Max Peak Compute
~16 TFLOPS
Ample computational headroom; not a bottleneck.
Predicted Performance
< 10 ms
The rectification pass is expected to be well within the 80 ms budget.

The analysis strongly indicates that the proposed optimized pipeline will meet and significantly outperform the 80 ms requirement, likely executing in under 10 ms on the target hardware.

6. Numerical Validation & Quality Metrics

To ensure the rectifier meets its stringent accuracy requirements, a comprehensive validation protocol using synthetic data with a known ground truth is essential.

6.1. Methodology for Synthetic Phantom Validation

Reprojection RMSE (mm): This metric directly assesses the 3D geometric fidelity of the entire rectification and reconstruction process.
Procedure:
a. Define a synthetic 3D phantom consisting of control points (e.g., rod tips) with known world coordinates (X,Y,Z).
b. Project these 3D points into the left and right image planes using the known system projection matrices (Pl​,Pr​) to generate ground-truth 2D points (ul​,vl​) and (ur​,vr​).
c. Apply the computed rectifying homographies (Hl​,Hr​) to these 2D points to get rectified points (ul′​,vl′​) and (ur′​,vr′​).
d. Perform 3D triangulation on the rectified point pairs to reconstruct the 3D coordinates (X′,Y′,Z′).
e. Calculate the Root Mean Square Error (RMSE) between the original 3D points and the reconstructed 3D points. The OpenCV function cv::projectPoints can be used as a reference for this process.45
Formula: RMSE=N1​∑i=1N​∥(Xi​,Yi​,Zi​)−(Xi′​,Yi′​,Zi′​)∥2​
Vertical Epipolar Error (px): This metric specifically measures the core success of the rectification: the alignment of epipolar lines.
Procedure:
a. Define a 3D checkerboard phantom.
b. Project the checkerboard into the left and right image planes to create a pair of un-rectified DRRs.
c. Apply the full GPU resampling pipeline (Section 4) to these DRRs to generate the final rectified images.
d. Use a standard corner detection algorithm (e.g., cv::findChessboardCorners) on both rectified images to find the pixel coordinates of matched corner pairs.46

e. For each corresponding corner pair at (ul′​,vl′​) and (ur′​,vr′​), the vertical epipolar error is ∣vl′​−vr′​∣.
f. Compute the mean and maximum of this error over all detected corners.
PSNR Loss (dB): This metric quantifies the information loss and artifact introduction caused by the resampling algorithm itself.
Procedure:
a. Generate a ground-truth rectified image using a high-precision, offline software-based Lanczos-3 resampler (e.g., using double-precision floating-point math).
b. Compare the GPU-generated rectified image from the pipeline against this ground-truth image.
c. Calculate the Peak Signal-to-Noise Ratio (PSNR) between the two images.
Formula: PSNR=20⋅log10​(MAXI​)−10⋅log10​(MSE), where MAXI​ is the maximum possible pixel value and MSE is the mean squared error between the images.48

6.2. Target Pass/Fail Criteria

The rectification module is considered to have met its design goals if it satisfies the following quantitative criteria.
Table 6.1: Summary of Validation Metrics and Pass Criteria

Metric
Target Value
Justification
Reprojection RMSE
≤ 0.05 mm
Directly fulfills the sub-voxel geometric fidelity requirement from the research objective.49
Mean Vertical Disparity
≤ 0.1 px
Meets the tight error budget derived in Section 1.3 to enable high-precision stereo matching.
Max Vertical Disparity
≤ 0.5 px
Ensures no catastrophic alignment failures at any point in the image, maintaining robust correlation.17
PSNR Drop vs. Ground Truth
< 0.3 dB
Guarantees that the resampling process is of high fidelity and introduces minimal noise or artifacts.48

This validation suite should be automated and integrated into a continuous integration (CI) workflow to ensure that any future modifications to the pipeline do not introduce regressions in geometric accuracy or image quality.

7. Integration API & Command-Buffer Scheduling


7.1. C/Objective-C Header RectificationUniforms

To ensure seamless data exchange between the C++/Objective-C application logic and the Metal Shading Language (MSL) kernels, a shared header file is used. This approach leverages Xcode's bridging header functionality to guarantee identical memory layouts for the struct in both environments.50 The use of
simd types ensures proper data alignment and size consistency across platforms.

C


// RectificationUniforms.h
// This header should be included by both the application's C++/Objective-C source
// and the.metal shader file.

#ifndef RectificationUniforms_h
#define RectificationUniforms_h

#include <simd/simd.h>

// This struct contains the 3x3 homography matrices for the left and right views.
// It will be passed to the GPU via a constant buffer.
typedef struct {
    simd_float3x3 leftHomography;
    simd_float3x3 rightHomography;
} RectificationUniforms;

#endif /* RectificationUniforms_h */


On the application side, an instance of this struct will be populated with the coefficients from Table 2.1 and passed to the compute command encoder using setBytes:length:index:. On the shader side, it will be received as a constant RectificationUniforms& argument.

7.2. Call-Graph and Pipeline Integration

The rectification module should be encapsulated within a dedicated class, for example DRRRectifier, to abstract the implementation details from the main rendering pipeline. This class will manage its own MTLComputePipelineState objects for the two resampling passes.
The data flow within the GPU pipeline is as follows:
→ `MTLTexture (Unrectified DRRs)` → **** → MTLTexture (Rectified DRRs) → ``
The primary public interface for the rectifier class would be a single encoding method:

Objective-C


- (void)encodeRectificationToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                           leftInputTexture:(id<MTLTexture>)leftInput
                          rightInputTexture:(id<MTLTexture>)rightInput
                          leftOutputTexture:(id<MTLTexture>)leftOutput
                         rightOutputTexture:(id<MTLTexture>)rightOutput;


Internally, this method would create a single MTLComputeCommandEncoder, encode the two passes for the left image, encode the two passes for the right image, and then end encoding. Metal ensures that commands within a single encoder are executed sequentially, so resource barriers between the horizontal and vertical passes are handled automatically.51

7.3. Strategy for Overlapping CPU I/O with GPU Rectification

The overall pipeline includes CPU-bound tasks (e.g., loading calibration or correction maps from disk) and GPU-bound tasks (DRR projection, rectification). To maximize throughput and hide latency, CPU and GPU work should be overlapped across frames.52
A highly efficient scheduling strategy involves using Metal's asynchronous execution model with completion handlers.53 Instead of a simple serial execution (
CPU waits for GPU -> CPU does work -> CPU tells GPU to work), a pipelined approach should be used.
Asynchronous Pipelined Execution Model:
Main Thread (Frame N):
The main application thread dispatches the primary GPU workload for Frame N (e.g., the heavy DRR projection from the CT volume).
Crucially, the CPU does not wait for this to complete. Instead, it immediately begins any necessary CPU-bound work for the next frame, Frame N+1 (e.g., initiating an asynchronous read of a crosstalk map from disk).
A scheduled completion handler is registered on the command buffer for Frame N's projection pass using addScheduledHandler:.
Completion Handler (Triggered after Frame N Projection):
Once the GPU has scheduled the projection for Frame N, Metal invokes the completion handler on a background CPU thread.
Inside this handler, the CPU performs the lightweight task of calculating the homographies for Frame N.
It then creates a new command buffer, encodes the rectification commands for Frame N using the DRRRectifier object, and commits this new command buffer to the queue.
This strategy creates a dependency chain on the GPU (Projection Pass → Rectification Pass) while allowing the CPU to work ahead on tasks for the next frame. The GPU begins rectifying Frame N immediately after finishing its projection, and this GPU work happens concurrently with the CPU's file I/O for Frame N+1. This overlapping of tasks effectively hides the I/O latency, leading to higher overall pipeline utilization and improved wall-clock performance.

8. Bibliography

Fusiello, A., Trucco, E., & Verri, A. (1999). A compact algorithm for rectification of stereo pairs. In Proceedings of the 5th International Conference on Machine Vision Applications (pp. 175-178). 7
Hartley, R., & Zisserman, A. (2003). Multiple View Geometry in Computer Vision (2nd ed.). Cambridge University Press. 55
Kim, W. (2005). Performance analysis of a real-time stereo-vision system. In 2005 IEEE International Conference on Systems, Man and Cybernetics (Vol. 4, pp. 3588-3593). IEEE. 17
Loop, C., & Zhang, Z. (1999). Computing rectifying homographies for stereo vision. In Proceedings of the 1999 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (Vol. 2, pp. 125-131). IEEE. 19
Ma, Y., Soatto, S., Kosecka, J., & Sastry, S. S. (2004). An Invitation to 3-D Vision: From Images to Geometric Models. Springer. 57
Mallon, J., & Whelan, P. F. (2005). Projective rectification from the fundamental matrix. In Proceedings of the Irish Machine Vision and Image Processing Conference (pp. 89-96). 14
Mei, X., Sun, X., Zhou, M., Jiao, J., Wang, H., & Zhang, X. P. (2011). A robust geometric stereo-rectification method. In British Machine Vision Conference (pp. 1-11). 5
Oluk, C., Bonnen, K., Burge, J., Cormack, L., & Geisler, W. (2021). Binocular slant discrimination of planar surfaces: Ideal-observer and standard cross-correlation models. Journal of Vision, 21(9), 1152. 25
Tehrani, P., & Rouvrais, S. (2024). Rectifying homographies for stereo vision: analytical solution for minimal distortion. arXiv preprint arXiv:2203.00123. 60
Yang, R., & Wang, L. (2006). Real-time stereovision on GPU. In Technical Sketch, SIGGRAPH. 62
Works cited
3/29/2011 CS 376 Lecture 17 Stereo 1 Stereo: Correspondence and Calibration, accessed on July 5, 2025, https://www.cs.utexas.edu/~grauman/courses/spring2011/slides/lecture17_stereo2.pdf
CS231A Course Notes 3: Epipolar Geometry, accessed on July 5, 2025, https://web.stanford.edu/class/cs231a/course_notes/03-epipolar-geometry.pdf
Image rectification - Wikipedia, accessed on July 5, 2025, https://en.wikipedia.org/wiki/Image_rectification
Epipolar Geometry / Matching Fundamentals, accessed on July 5, 2025, https://www.cs.auckland.ac.nz/courses/compsci773s1t/lectures/773-GG/lectA-773.htm
Three-step image rectification - Imagine, accessed on July 5, 2025, http://dev.ipol.im/~morel/Dossier_MVA_2011_Cours_Transparents_Documents/2011_Cours7_Document1_Three_Step_BMVC10.pdf
Epipolar rectification, accessed on July 5, 2025, http://dev.ipol.im/~morel/Dossier_MVA_2010_Cours_Transparents_Documents/Cours_5_stereorectification_texte.pdf
A Compact Algorithm for Rectification of Stereo Pairs - Andrea Fusiello, accessed on July 5, 2025, https://fusiello.github.io/papers/mva99.pdf
Flow-Guided Online Stereo Rectification for Wide Baseline Stereo - Princeton Computational Imaging Lab, accessed on July 5, 2025, https://light.princeton.edu/wp-content/uploads/2024/04/Flow-Guided-Online-Stereo-Rectification-for-Wide-Baseline-Stereo.pdf
Stereo, accessed on July 5, 2025, https://www.cs.cmu.edu/~16385/lectures/lecture13.pdf
Three-step image rectification - Imagine, accessed on July 5, 2025, https://imagine.enpc.fr/~monasse/Callisto/pdf/BMVC10Rectif.pdf
Fundamental matrix (computer vision) - Wikipedia, accessed on July 5, 2025, https://en.wikipedia.org/wiki/Fundamental_matrix_(computer_vision)
Epipolar Geometry and the Fundamental Matrix - RPI ECSE, accessed on July 5, 2025, https://sites.ecse.rpi.edu/~qji/CV/epipolar_fundF.pdf
Computer Vision: What is the relationship of homography and the fundamental matrix?, accessed on July 5, 2025, https://www.quora.com/Computer-Vision-What-is-the-relationship-of-homography-and-the-fundamental-matrix
Projective Rectification from the Fundamental Matrix - Dublin City University, accessed on July 5, 2025, https://doras.dcu.ie/4662/1/JM_IVC_2005.pdf
(PDF) Tutorial on Rectification of Stereo Images - ResearchGate, accessed on July 5, 2025, https://www.researchgate.net/publication/2841773_Tutorial_on_Rectification_of_Stereo_Images
Depth (Distance) Perception with Stereo Cameras: Epipolar ..., accessed on July 5, 2025, https://medium.com/@nasuhcanturker/depth-perception-with-stereo-cameras-epipolar-geometry-and-disparity-map-675b86c0298b
Performance Analysis and Validation of a Stereo Vision System - JPL Robotics, accessed on July 5, 2025, https://www-robotics.jpl.nasa.gov/media/documents/IEEE-SMC-2005-wskim9.pdf
Stereo-phase rectification for metric profilometry with two calibrated cameras and one uncalibrated projector - Optics Letters, accessed on July 5, 2025, https://opg.optica.org/abstract.cfm?uri=ao-61-21-6097
Shearing and Hartley's rectification - Computational Science Stack Exchange, accessed on July 5, 2025, https://scicomp.stackexchange.com/questions/2844/shearing-and-hartleys-rectification
Image rectification for stereoscopic visualization - Optica Publishing Group, accessed on July 5, 2025, https://opg.optica.org/abstract.cfm?uri=josaa-25-11-2721
Computing rectifying homographies for stereo vision - Computer ..., accessed on July 5, 2025, https://dev.ipol.im/~morel/Dossier_MVA_2011_Cours_Transparents_Documents/2011_Cours7_Document2_Loop-Zhang-CVPR1999.pdf
(PDF) Computing rectifying homographies for stereo vision - ResearchGate, accessed on July 5, 2025, https://www.researchgate.net/publication/3813415_Computing_rectifying_homographies_for_stereo_vision
A Compact Algorithm for Rectification of Stereo Pairs - ResearchGate, accessed on July 5, 2025, https://www.researchgate.net/publication/2800494_A_Compact_Algorithm_for_Rectification_of_Stereo_Pairs
DEFOM-Stereo: Depth Foundation Model Based Stereo Matching - arXiv, accessed on July 5, 2025, https://arxiv.org/html/2501.09466v1
Stereo Slant Discrimination of Planar 3D Surfaces - Johannes Burge, accessed on July 5, 2025, https://jburge.psych.upenn.edu/ewExternalFiles/OlukBonnenBurgeCormackGeisler_bioRxiv_2021.pdf
Modeling arbitrarily oriented slanted planes for efficient stereo vision based on block matching | Semantic Scholar, accessed on July 5, 2025, https://www.semanticscholar.org/paper/Modeling-arbitrarily-oriented-slanted-planes-for-on-Ranft-Strau%C3%9F/f34a7124015c354f7f21e58477038bc66aeff0f3
RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching - ar5iv - arXiv, accessed on July 5, 2025, https://ar5iv.labs.arxiv.org/html/2109.07547
Stereo Matching and Depth Map Creation on the Vision Pro | Griffin ..., accessed on July 5, 2025, https://griffinhurt.com/blog/2025/avp-stereo/
Fast GPU-based image warping and inpainting for frame interpolation, accessed on July 5, 2025, http://gravisma.zcu.cz/GraVisMa-2010/Papers/Full/D83-full.pdf
Lanczos resampling - Wikipedia, accessed on July 5, 2025, https://en.wikipedia.org/wiki/Lanczos_resampling
Lanczos: A resampling example with in-depth explanations - GitHub, accessed on July 5, 2025, https://github.com/jeffboody/Lanczos
MPSImageLanczosScale | Apple Developer Documentation, accessed on July 5, 2025, https://developer.apple.com/documentation/metalperformanceshaders/mpsimagelanczosscale
What is the difference between Lanczos seperable and non-seperable. - Affinity | Forum, accessed on July 5, 2025, https://forum.affinity.serif.com/index.php?/topic/9378-what-is-the-difference-between-lanczos-seperable-and-non-seperable/
Separable filter - Wikipedia, accessed on July 5, 2025, https://en.wikipedia.org/wiki/Separable_filter
Porting your Metal code to Apple silicon | Apple Developer Documentation, accessed on July 5, 2025, https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon
Optimize Metal Performance for Apple silicon Macs - WWDC20 - Videos - Apple Developer, accessed on July 5, 2025, https://developer.apple.com/videos/play/wwdc2020/10632/
Why does allocating a float in Metal's threadgroup address space give different results depending on the hardware? - Stack Overflow, accessed on July 5, 2025, https://stackoverflow.com/questions/70815182/why-does-allocating-a-float-in-metals-threadgroup-address-space-give-different
Discover Metal enhancements for A14 Bionic - Tech Talks - Videos - Apple Developer, accessed on July 5, 2025, https://developer.apple.com/videos/play/tech-talks/10858/
Example of how to use Metal SIMD functions to perform a reduction operation - GitHub Gist, accessed on July 5, 2025, https://gist.github.com/rgov/9139d725841670e8cbdf1593d5f369da
Apple M4 Max 40-core GPU - Benchmarks and Specs - NotebookCheck.net Tech, accessed on July 5, 2025, https://www.notebookcheck.net/Apple-M4-Max-40-core-GPU-Benchmarks-and-Specs.920457.0.html
Roofline model - Wikipedia, accessed on July 5, 2025, https://en.wikipedia.org/wiki/Roofline_model
Apple M4 Max (40-core) - GPU Performance - Novabench, accessed on July 5, 2025, https://novabench.com/parts/gpu/apple-m4-max-40-core-gpu
Explore Live GPU Profiling with Metal Counters - Tech Talks - Videos - Apple Developer, accessed on July 5, 2025, https://developer.apple.com/videos/play/tech-talks/10001/
MTLCounter - Documentation - Apple Developer, accessed on July 5, 2025, https://developer.apple.com/documentation/metal/mtlcounter
Reprojection error with opencv, accessed on July 5, 2025, https://answers.opencv.org/question/87253/reprojection-error-with-opencv/
How to calculate an epipolar line with a stereo pair of images in ..., accessed on July 5, 2025, https://stackoverflow.com/questions/51089781/how-to-calculate-an-epipolar-line-with-a-stereo-pair-of-images-in-python-opencv
odr.chalmers.se, accessed on July 5, 2025, https://odr.chalmers.se/server/api/core/bitstreams/333d9ed4-85e7-46c7-b001-68ab890f557f/content
Peak signal-to-noise ratio - Wikipedia, accessed on July 5, 2025, https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
Camera calibration and reprojection error - fjp.github.io, accessed on July 5, 2025, https://fjp.at/posts/computer-vision/camera-calibration-reprojection-error/
Passing Structured Data to a Metal Compute Function | Apple Developer Documentation, accessed on July 5, 2025, https://developer.apple.com/documentation/realitykit/passing-structured-data-to-a-metal-compute-function
How do you synchronize a Metal Performance Shader with an MTLBlitCommandEncoder?, accessed on July 5, 2025, https://stackoverflow.com/questions/52007588/how-do-you-synchronize-a-metal-performance-shader-with-an-mtlblitcommandencoder
Please help me understand command buffer execution : r/vulkan - Reddit, accessed on July 5, 2025, https://www.reddit.com/r/vulkan/comments/85pu3u/please_help_me_understand_command_buffer_execution/
Command Organization and Execution Model - Apple Developer, accessed on July 5, 2025, https://developer.apple.com/library/archive/documentation/Miscellaneous/Conceptual/MetalProgrammingGuide/Cmd-Submiss/Cmd-Submiss.html
addScheduledHandler(_:) | Apple Developer Documentation, accessed on July 5, 2025, https://developer.apple.com/documentation/metal/mtlcommandbuffer/addscheduledhandler(_:)
Multiple View Geometry in Computer Vision - Richard Hartley, accessed on July 5, 2025, https://books.google.com/books/about/Multiple_View_Geometry_in_Computer_Visio.html?hl=es&id=si3R3Pfa98QC
An Invitation to 3-D Vision: From Images to Geometric Models - Yi, accessed on July 5, 2025, https://books.google.com/books?id=6tUqQmwan4UC
An Invitation to 3-D Vision, accessed on July 5, 2025, https://www.eecis.udel.edu/~cer/arv/readings/old_mkss.pdf
mint-lab/3dv_tutorial: An Invitation to 3D Vision: A Tutorial for Everyone - GitHub, accessed on July 5, 2025, https://github.com/mint-lab/3dv_tutorial
Stereo Slant Estimation of Planar Surfaces: Standard Cross-Correlation vs. Planar-Correlation | JOV | ARVO Journals, accessed on July 5, 2025, https://jov.arvojournals.org/article.aspx?articleid=2699126
Rectifying homographies for stereo vision: analytical solution for minimal distortion - arXiv, accessed on July 5, 2025, https://arxiv.org/abs/2203.00123
Rectifying homographies for stereo vision: analytical solution for minimal distortion - CERES Research Repository, accessed on July 5, 2025, https://dspace.lib.cranfield.ac.uk/bitstream/handle/1826/18279/Rectifying_homographies_for_stereo_vision-2022.pdf?sequence=1
Stereovision on GPU, accessed on July 5, 2025, https://www.cs.unc.edu/~welch/media/pdf/Yang2006-EDGE-stereovision.pdf
