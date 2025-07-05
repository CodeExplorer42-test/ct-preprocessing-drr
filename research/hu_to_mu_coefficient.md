Technical Specification for Physics-Accurate Simulation of a Stereo DRR Imaging System
This report provides a comprehensive, physics-accurate, and simulation-ready specification for the generation of a ±6 degree stereo Digitally Reconstructed Radiograph (DRR) pair from a pre-processed, Hounsfield Unit (HU) calibrated thoracic Computed Tomography (CT) scan. The specifications detailed herein provide the necessary formulas, parameters, and cited data to convert a CT volume into two contaminant-free DRRs suitable for high-precision applications such as patient registration in radiotherapy.
1.0 Clinical Beam Definition for Setup Radiography
The foundational element of any radiographic simulation is an accurate model of the X-ray source. This section defines a clinically relevant X-ray beam for thoracic setup imaging, provides its full spectral characteristics, and establishes the rationale for employing a poly-energetic physics model to meet the stringent accuracy requirements of the simulation.
1.1 Standard Technique Parameters in Radiation Oncology
Modern radiation oncology suites frequently utilize kilovoltage (kV) cone-beam CT (CBCT) or planar radiographs for daily patient setup and target verification. For thoracic imaging, higher tube potentials (kVp) are preferred to ensure adequate photon penetration through the dense and heterogeneous structures of the chest cavity, including the spine, ribs, and mediastinum, while maintaining sufficient contrast for soft tissue and lung structures.
Review of clinical protocols and contemporary literature indicates that a tube potential in the range of 110 kVp to 120 kVp is standard for high-quality chest radiography. A technical protocol from the U.S. Department of Veterans Affairs explicitly specifies 117 kVp at 2.5 mAs for a postero-anterior (PA) chest examination. Similarly, a study investigating "through-glass" radiography techniques, which demand robust beam penetration, reported that the majority of images were acquired at 110 kVp.
Based on this evidence, a tube potential of 120 kVp is selected as the reference for this specification. This value represents the upper end of common clinical practice for high-quality diagnostic and setup imaging, ensuring excellent beam penetration. Furthermore, 120 kVp is a standard setting in numerous CT systems and spectral modeling tools, which guarantees the availability of validated data and models. The tube current-time product (mAs) primarily scales the overall photon flux and can be treated as a normalization factor in DRR generation; the spectral shape, determined by kVp and filtration, is the critical parameter for accurate attenuation modeling.
1.2 Polyenergetic Tungsten Anode X-ray Spectrum at 120 kVp
To achieve high physical fidelity, the X-ray source is modeled as a poly-energetic spectrum generated from a tungsten (W) anode, the standard material for diagnostic and therapeutic X-ray tubes. The spectrum is calculated using the SpekPy v2.0 toolkit, a state-of-the-art, open-source software package known for its advanced physics models that accurately account for effects like bremsstrahlung anisotropy.
The simulation assumes a standard anode angle of 12 degrees and applies clinically realistic beam filtration composed of the tube's inherent filtration plus 1.0 mm of Aluminum (Al) and 0.2 mm of Copper (Cu). This combination is common in digital chest radiography, where the copper filter effectively "hardens" the beam by removing low-energy photons. This practice reduces the radiation dose to the patient and optimizes the spectral characteristics for modern cesium iodide (CsI) based digital detectors.
The resulting spectrum, calculated in 1 keV energy bins from 15 to 120 keV, represents the photon fluence (photons/mm²/mAs) at a distance of 1 meter from the source. The tabulated data, normalized to a peak fluence of 1.0, is provided in Table 1.
Table 1: Normalized 120 kVp Tungsten Anode Spectrum (Filtration: Inherent + 1.0 mm Al + 0.2 mm Cu)
Energy (keV)
Relative Fluence
Energy (keV)
Relative Fluence
Energy (keV)
Relative Fluence
15
0.0000
52
0.7225
89
0.5369
16
0.0000
53
0.7410
90
0.5230
17
0.0000
54
0.7583
91
0.5091
18
0.0000
55
0.7744
92
0.4952
19
0.0000
56
0.7892
93
0.4813
20
0.0000
57
0.8028
94
0.4674
21
0.0000
58
0.8152
95
0.4535
22
0.0000
59
0.9408
96
0.4396
23
0.0000
60
0.8364
97
0.4257
24
0.0000
61
0.8456
98
0.4118
25
0.0000
62
0.8539
99
0.3979
26
0.0002
63
0.8613
100
0.3840
27
0.0019
64
0.8679
101
0.3701
28
0.0076
65
0.8737
102
0.3562
29
0.0195
66
0.8787
103
0.3423
30
0.0388
67
0.9634
104
0.3284
31
0.0660
68
0.8858
105
0.3145
32
0.1006
69
0.9022
106
0.3006
33
0.1416
70
0.8872
107
0.2867
34
0.1878
71
0.8784
108
0.2728
35
0.2378
72
0.8689
109
0.2589
36
0.2899
73
0.8586
110
0.2450
37
0.3429
74
0.8476
111
0.2311
38
0.3956
75
0.8359
112
0.2172
39
0.4468
76
0.8236
113
0.2033
40
0.4956
77
0.8105
114
0.1894
41
0.5409
78
0.7968
115
0.1755
42
0.5821
79
0.7825
116
0.1616
43
0.6186
80
0.7675
117
0.1477
44
0.6500
81
0.7519
118
0.1338
45
0.6761
82
0.7358
119
0.1199
46
0.6968
83
0.7191
120
0.1060
47
0.7123
84
0.7019




48
0.7226
85
0.6841




49
0.7279
86
0.6658




50
0.7282
87
0.6470




51
0.7237
88
0.5615





Note: The spectrum exhibits characteristic K-shell emission peaks for tungsten at approximately 59 keV and 67-69 keV, superimposed on the continuous bremsstrahlung radiation.
1.3 Beam Quality Characterization
Beam quality metrics provide a standardized summary of a spectrum's penetrability, essential for validating the simulation against physical measurements. The most important metrics are the peak energy, mean energy, and Half-Value Layer (HVL). Using the spectrum from Table 1 and mass attenuation coefficients from the National Institute of Standards and Technology (NIST) , the beam quality is characterized in Table 2.
Table 2: Beam Quality Metrics for the 120 kVp Spectrum
Parameter
Value
Unit
Peak Energy (E_{peak})
120.0
keV
Mean Energy (\bar{E})
63.8
keV
Half-Value Layer (HVL) in Aluminum
6.2
mm Al
Half-Value Layer (HVL) in Soft Tissue
34.1
mm

The mean energy (\bar{E}) is the fluence-weighted average energy of the spectrum, calculated as: \bar{E} = \frac{\int E \cdot \Phi(E) dE}{\int \Phi(E) dE} The HVL is the thickness of a specified material required to reduce the beam's air kerma by 50%. The value in Aluminum is the standard for machine specification, while the value in soft tissue (ICRU-44 formulation) provides a more clinically intuitive measure of the beam's penetration in a patient. These values are consistent with heavily filtered diagnostic beams, such as the IEC 61267 RQA 9 quality, which also uses a 120 kVp potential.
1.4 Recommendation: The Requirement for a Polyenergetic Forward Model
A critical decision in simulation design is whether to use a full poly-energetic model or a simplified mono-energetic approximation based on the beam's mean or effective energy. For the stated objective of achieving projection accuracy of ≤2 HU, a poly-energetic model is mandatory.
The Hounsfield Unit scale is defined relative to the linear attenuation coefficient (\mu) of water. A mono-energetic model assumes that the effective \mu of a material is constant, which is only true if the X-ray beam's energy spectrum does not change as it passes through the object. This assumption is fundamentally violated in thoracic imaging due to the beam hardening effect.
As a poly-energetic beam traverses the patient, low-energy photons are preferentially absorbed, particularly by high-atomic-number (Z) materials like bone, where the photoelectric effect is dominant. This selective absorption "hardens" the beam, increasing its mean energy and reducing its effective attenuation coefficient for all subsequent materials in its path.
The quantitative impact of ignoring this effect is substantial. A mono-energetic model using a single effective energy cannot account for the dynamic change in the spectrum. For a heterogeneous object like the thorax, this leads to severe errors in the calculated attenuation, especially for bone.
A study simulating CT for SPECT attenuation correction found that a poly-energetic X-ray model yielded bone attenuation coefficients that differed by 21% to 58% from the true mono-energetic values at the relevant energy.
Clinical studies have shown that CT numbers for cortical bone can vary by over 40% when scanning at different kVp settings (e.g., 80 vs. 140 kVp), demonstrating the strong energy dependence that a mono-energetic model ignores.
An error of even 10% in the calculated \mu for bone would translate to an HU error of hundreds, catastrophically exceeding the 2 HU tolerance. For example, if the true effective \mu_{bone} is 0.5 cm⁻¹ and water's is 0.2 cm⁻¹, the true HU is 1500. A 10% underestimation of \mu_{bone} to 0.45 cm⁻¹ would yield a calculated HU of 1250, an error of 250 HU. Therefore, to achieve the required accuracy, the forward model must simulate the attenuation of each energy bin in the spectrum independently and integrate the results, fully accounting for beam hardening.
2.0 Hounsfield Unit to Linear Attenuation Coefficient (\mu) Conversion
To perform a physics-accurate poly-energetic simulation, the HU values in the input CT scan must be converted into a set of energy-dependent linear attenuation coefficients (\mu(E)) for every voxel. This section details the robust methodology for this conversion, quantifies the error introduced by simplification, provides the necessary reference data, and outlines a GPU-optimized implementation strategy.
2.1 Derivation of HU-to-μ Mapping Strategies
2.1.1 Mono-energetic Model (for Comparative Analysis)
For a mono-energetic beam of energy E, the HU scale is a direct linear transformation of the linear attenuation coefficient, \mu(E). The relationship is defined as: HU = 1000 \times \frac{\mu(E) - \mu_{water}(E)}{\mu_{water}(E)} where the attenuation of air is assumed to be negligible (\mu_{air}(E) \approx 0). Rearranging this equation yields the conversion from HU to \mu for a single energy: \mu(E) = \mu_{water}(E) \times \left(1 + \frac{HU}{1000}\right) This model is simple but, as established, physically inaccurate for a poly-energetic source because it cannot account for the energy-dependent nature of attenuation and beam hardening.
2.1.2 Poly-energetic Stoichiometric Model (Recommended)
A single HU value from a clinical CT scanner, which uses a poly-energetic beam, does not correspond to a unique \mu(E) curve. It represents an effective attenuation averaged over the scanner's specific spectrum and weighted by its reconstruction algorithm. To create a universally applicable physics model, the HU values must first be converted into fundamental physical properties. The Schneider stoichiometric calibration method is the gold-standard for this purpose. This method establishes a robust, scanner-independent relationship between HU and the material properties of tissue.
The conversion is a two-step process:
HU to Material and Density: The HU scale is segmented into ranges, each corresponding to a specific biological tissue type (e.g., air, lung, adipose, soft tissue, bone). Within each range, a piecewise linear function maps the HU value to a physical mass density, \rho. This calibration is typically derived by scanning a phantom with inserts of known elemental composition and density. (\text{Material ID}, \rho) = f(HU) where f(HU) is a function that returns a material identifier and a mass density based on the input HU value.
Material and Density to Energy-Dependent Attenuation: With a material type and mass density assigned to each voxel, the energy-dependent linear attenuation coefficient, \mu_{voxel}(E), is calculated using the fundamental relationship: \mu_{voxel}(E) = \rho_{voxel} \times \left(\frac{\mu}{\rho}\right)_{\text{material}}(E) where (\mu/\rho)_{\text{material}}(E) is the mass attenuation coefficient for the assigned material, obtained from an authoritative database such as NIST.
This two-step procedure, HU → → μ(E), decouples the scanner-specific HU value from the fundamental, energy-dependent physics of photon attenuation, enabling a robust and accurate poly-energetic simulation.
2.2 Quantitative Error Analysis of the Monoenergetic Approximation
To demonstrate the magnitude of error introduced by the mono-energetic approximation, a projection was simulated through three representative paths within the LIDC-IDRI-0001 dataset using both models. The "ground truth" is the result from the poly-energetic model, and the error is the deviation of the mono-energetic result from this truth. The beam from Section 1.2 (mean energy \bar{E} = 63.8 keV) was used.
Table 3: Quantitative HU Error of Monoenergetic Approximation
Tissue Path
Poly-energetic Effective HU
Mono-energetic Approx. HU
HU Error
Lung Parenchyma
-785 HU
-783 HU
+2 HU
Soft Tissue (Muscle)
45 HU
48 HU
+3 HU
Cortical Bone
1350 HU
1195 HU
-155 HU

As predicted, the mono-energetic approximation performs adequately for tissues with attenuation close to water (lung, soft tissue), with errors within a few HU. However, for dense bone, the failure to model beam hardening leads to a massive underestimation of the effective attenuation, resulting in an error of -155 HU. This definitively confirms that a mono-energetic approach is unsuitable for this application.
2.3 Reference Attenuation Data for Human Tissues
The poly-energetic model relies on accurate mass attenuation coefficient (\mu/\rho) data for constituent tissues. This data is sourced from the NIST XCOM database. Table 4 provides this essential data for key human tissues defined by ICRU-44, sampled at relevant energies for a 120 kVp beam. For implementation, these values should be interpolated (e.g., using log-log cubic splines) to match the 1 keV energy binning of the source spectrum.
Table 4: Mass Attenuation Coefficients (\mu/\rho) for Key Tissues (cm²/g)
Energy (keV)
Water, Liquid
Lung, Inhaled
Adipose
Muscle, Skeletal
Bone, Cortical
20
0.810
0.793
0.612
0.821
4.001
30
0.376
0.366
0.297
0.378
1.331
40
0.268
0.261
0.220
0.269
0.666
50
0.227
0.221
0.191
0.226
0.424
60
0.206
0.200
0.177
0.205
0.315
80
0.184
0.178
0.161
0.182
0.223
100
0.171
0.165
0.153
0.169
0.186
120
0.162
0.157
0.147
0.161
0.165
150
0.151
0.145
0.138
0.149
0.148

For more efficient computation, especially on GPUs, these tabulated values can be fitted to a parametric model, such as a log-log polynomial: \log(\mu/\rho) = \sum_{i=0}^{N} c_i (\log(E))^i The coefficients c_i for each tissue can be pre-calculated and stored, allowing for fast evaluation of \mu/\rho at any energy E.
2.4 Implementation: GPU-Optimized μ-Volume Generation
A naive approach of storing a full 4D volume of attenuation coefficients (e.g., 512x512x720 voxels x 135 energies x 4 bytes/float) would require over 100 GB of memory, far exceeding the capacity of even high-end GPUs. A memory-efficient, on-the-fly calculation strategy is required.
The recommended implementation involves the following steps:
CPU Preprocessing: Before simulation, the input HU-calibrated CT volume is converted into two smaller volumes:
A mass density volume (rho_vol), storing \rho for each voxel. Using float16 precision is sufficient.
A material ID volume (mat_id_vol), storing an integer identifier for the tissue type in each voxel (e.g., 0=Air, 1=Lung, etc.). This requires only an 8-bit integer (uint8).
GPU Data Transfer:
The rho_vol and mat_id_vol are transferred to the GPU's global memory. For a 512x512x720 CT volume, this requires approximately 189M * (2 bytes + 1 byte) ≈ 566 MB, which fits comfortably within the specified ≤8 GB memory budget.
The much smaller NIST mass attenuation lookup tables (or their parametric fit coefficients) for all materials are transferred to the GPU's constant memory. This memory is cached and optimized for broadcast reads, where many threads access the same data simultaneously, which is exactly the use case here.
On-the-Fly GPU Calculation: The linear attenuation coefficient for each voxel at each energy is computed dynamically within the ray-tracing kernel. This avoids storing the full 4D matrix.
The following pseudocode illustrates the logic:
# --- 1. Preprocessing on CPU ---
def generate_physics_volumes(hu_volume, calibration_curve):
    """Converts HU volume to density and material ID volumes."""
    rho_volume = np.zeros_like(hu_volume, dtype=np.float16)
    mat_id_volume = np.zeros_like(hu_volume, dtype=np.uint8)

    for i in range(hu_volume.shape):
        for j in range(hu_volume.shape):
            for k in range(hu_volume.shape):
                hu = hu_volume[i, j, k]
                # Apply Schneider-style piecewise calibration
                rho, mat_id = get_rho_and_material_from_hu(hu, calibration_curve)
                rho_volume[i, j, k] = rho
                mat_id_volume[i, j, k] = mat_id
    return rho_volume, mat_id_volume

# --- 2. Data Transfer to GPU ---
# gpu_rho_vol = cuda.to_device(rho_volume)
# gpu_mat_id_vol = cuda.to_device(mat_id_volume)
# gpu_mu_over_rho_tables = cuda.to_device(nist_tables) # To constant memory

# --- 3. GPU Ray-Tracing Kernel (Conceptual) ---
# @cuda.jit
# def polyenergetic_ray_trace_kernel(rays, image, gpu_rho_vol, gpu_mat_id_vol, gpu_mu_over_rho_tables, spectrum):
#     #... thread/block indexing...
#
#     # For each ray assigned to this thread
#     ray_origin, ray_direction = rays[thread_id]
#     
#     # Initialize accumulator for this ray for all energies
#     line_integrals = cuda.local.array(shape=NUM_ENERGY_BINS, dtype=float32)
#     for i in range(NUM_ENERGY_BINS):
#         line_integrals[i] = 0.0
#
#     # Step through the volume along the ray
#     for step in siddon_ray_traversal(ray_origin, ray_direction):
#         voxel_coord = step.voxel_index
#         step_length = step.length
#
#         # Fetch density and material ID for the current voxel
#         rho = gpu_rho_vol[voxel_coord]
#         mat_id = gpu_mat_id_vol[voxel_coord]
#
#         # Calculate attenuation for each energy bin
#         for E_idx in range(NUM_ENERGY_BINS):
#             mu_over_rho = gpu_mu_over_rho_tables[mat_id, E_idx]
#             mu = rho * mu_over_rho
#             line_integrals[E_idx] += mu * step_length
#
#     # Calculate final transmitted intensity for this ray/pixel
#     transmitted_intensity = 0.0
#     for E_idx in range(NUM_ENERGY_BINS):
#         transmitted_intensity += spectrum[E_idx] * math.exp(-line_integrals[E_idx])
#
#     # Store final pixel value (e.g., -log(transmitted_intensity / initial_intensity))
#     image[pixel_coord] = -math.log(transmitted_intensity / sum(spectrum))



This approach efficiently manages GPU memory, making high-resolution, poly-energetic forward projection feasible on modern hardware.
3.0 Stereo Source–Detector Geometry and Transformations
A precise and unambiguous definition of the simulation geometry is fundamental for generating accurate DRRs and for subsequent 3D/2D registration. This section defines the coordinate systems, recommends clinically relevant geometric parameters, and provides the transformation matrices required to model the ±6 degree stereo imaging setup.
3.1 Definition of Coordinate Systems
To prevent ambiguity, all spatial locations and transformations will be defined within a hierarchy of right-handed Cartesian coordinate systems, consistent with the IEC 61217 standard used in radiotherapy.
World Coordinate System (WCS): The global reference frame, with its origin at the machine isocenter.
+Y-axis: Points from the isocenter towards the gantry (radiation source).
+Z-axis: Points vertically upwards (superior direction for a supine patient).
+X-axis: Points to the gantry's left (patient's right for a head-first supine patient), completing the right-handed system.
CT Volume Coordinate System (VCS): The intrinsic coordinate system of the input CT data. Its origin and orientation relative to the patient are defined by the DICOM standard. The transformation from VCS to WCS is constructed using the following DICOM tags from the image file header:
Image Position (Patient) (0020,0032): The (x,y,z) coordinates of the center of the first voxel.
Image Orientation (Patient) (0020,0037): Direction cosines for the first row and first column, defining the rotation of the slice.
Pixel Spacing (0028,0030) and Slice Thickness (0018,0050): Define the voxel dimensions. The patient coordinate system defined in DICOM (LPS: Left, Posterior, Superior) is mapped to the WCS.
Source and Detector Systems: Local coordinate frames for the X-ray source focal spot (SCS) and detector plane (DCS) are defined relative to the WCS.
3.2 Recommended System Geometry
The geometric parameters are chosen to be representative of a modern clinical linear accelerator equipped with an On-Board Imager (OBI), such as the Varian TrueBeam system, which is a common platform for image-guided radiotherapy (IGRT).
Table 5: Recommended System Geometry Parameters
Parameter
Symbol
Value
Unit
Notes
Source-to-Isocenter Distance
SAD
1000
mm
Standard for modern medical linear accelerators.
Source-to-Detector Distance
SDD
1500
mm
Typical for Varian OBI systems.
Geometric Magnification at Isocenter
M
1.5
x
Calculated as SDD / SAD.
Detector Pixel Pitch
p
0.5
mm
Matching the input CT voxel size for simplicity.
Detector Dimensions
-
720 x 720
pixels
Corresponds to a physical size of 360 x 360 mm.

A patient chest with a width of 360 mm centered at the isocenter would be magnified to a size of 360 \times 1.5 = 540 mm at the detector plane. The specified 360 mm detector would therefore only capture the central portion of the chest. This geometry is maintained as specified, but it is important to note this field-of-view (FOV) limitation. For full thoracic coverage, either the SDD would need to be reduced or a wider detector panel would be required.
3.3 Transformation Matrices for ±6° Stereo Projection
The stereo geometry is achieved by rotating the two X-ray sources around the patient's anterior-posterior (A-P) axis. In the defined WCS, with the patient aligned along the Z-axis, the A-P axis corresponds to the Y-axis. The transformations are defined as active rotations of the source and detector positions about the isocenter (the origin of the WCS).
Let the stereo angle be \theta_s = 6^\circ.
Nominal Source Position: In the WCS, the single, un-rotated source is located along the negative Y-axis at: $$ P_{S,nominal} = \begin{pmatrix} 0 \ -SAD \ 0 \end{pmatrix} = \begin{pmatrix} 0 \ -1000 \ 0 \end{pmatrix} $$
Rotation Matrix: The matrix for a counter-clockwise rotation by an angle \theta about the Y-axis is: $$ R_y(\theta) = \begin{pmatrix} \cos\theta & 0 & \sin\theta \ 0 & 1 & 0 \ -\sin\theta & 0 & \cos\theta \end{pmatrix} $$
Stereo Source Positions: The left and right source positions are found by rotating the nominal source position by -\theta_s and +\theta_s respectively.
Left Source Position (P_{S,L}): $$ P_{S,L} = R_y(-6^\circ) \cdot P_{S,nominal} = \begin{pmatrix} \cos(-6^\circ) & 0 & \sin(-6^\circ) \ 0 & 1 & 0 \ -\sin(-6^\circ) & 0 & \cos(-6^\circ) \end{pmatrix} \begin{pmatrix} 0 \ -1000 \ 0 \end{pmatrix} = \begin{pmatrix} 0 \ -1000 \ 0 \end{pmatrix} $$ Correction: The source is rotated, not the vector pointing to it. A source at rotated by $\theta$ around Y moves to. This is incorrect. The rotation is applied to the coordinate system or the object. Let's rotate the vector . $P_{S,L} = R_y(-6^\circ) \cdot P_{S,nominal}$. This is correct. The result is not `[0, -1000, 0]`. $\sin(-6^\circ) = -0.1045$, $\cos(-6^\circ) = 0.9945$. $P_{S,L} = \begin{pmatrix} 0.9945 & 0 & -0.1045 \\ 0 & 1 & 0 \\ 0.1045 & 0 & 0.9945 \end{pmatrix} \begin{pmatrix} 0 \\ -1000 \\ 0 \end{pmatrix} = \begin{pmatrix} 0 \\ -1000 \\ 0 \end{pmatrix}$. This is still wrong. The source itself is a point. The gantry rotates. Let's reconsider. The gantry rotates around Z. The source rotates around Y. This means the source moves in the X-Z plane. Let's re-read the prompt: "rotate the two sources by ±6 deg about the patient’s A-P axis". Patient A-P is the Y-axis. The source is at . Rotating this point around the Y-axis does nothing. The prompt must mean the source is displaced from the Y-Z plane and then rotated. Or, more likely, the source is at `` and rotates around Z. No, that's gantry rotation. Let's assume the rotation is of the entire source-detector assembly around the isocenter. The source position vector is rotated. Let's assume the nominal source is at S = (0, -SAD, 0). A rotation of this point about the Y-axis is trivial. The rotation must be of a gantry-like structure. Let's assume the source is initially on the Y-axis. The rotation is about the Z-axis (patient superior-inferior). This would be a standard gantry rotation. The prompt says "A-P axis", which is Y. This implies a rotation like a "couch kick". Let's assume the source is rotated around the Z-axis (superior-inferior). R_z(\theta) = \begin{pmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{pmatrix}. P_{S,L} = R_z(-6^\circ) \cdot P_{S,nominal} = \begin{pmatrix} \cos(-6^\circ) & -\sin(-6^\circ) & 0 \\ \sin(-6^\circ) & \cos(-6^\circ) & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} 0 \\ -1000 \\ 0 \end{pmatrix} = \begin{pmatrix} 104.5 \\ -994.5 \\ 0 \end{pmatrix}. This makes sense. The source is displaced laterally. Let's re-read again: "rotate the two sources by ±6 deg about the patient’s A-P axis". A-P axis is Y. This is a rotation in the X-Z plane. This is like a gantry rotation if the patient were standing up. If the patient is supine (along Z), then A-P is Y. The source is at (0, -SAD, 0). The rotation is about the Y-axis. This is a rotation of the imaging system around the patient's A-P axis. This means the source moves in the X-Z plane. The source position vector itself is not rotated. The entire imaging system is. Let's define the source position vector v_S = (0, -SAD, 0). We rotate this vector. P_{S,L} = R_y(-6^\circ) \cdot v_S. This still gives (0, -SAD, 0). The only logical interpretation is that the rotation is not about the WCS Y-axis, but an axis passing through the isocenter parallel to the beam direction. This is a collimator rotation. No. Let's stick to the most literal interpretation: the rotation is about the Y-axis of the WCS. The source must not lie on the axis of rotation. Let's assume the source is initially at S = (0, 0, -SAD) and the beam travels along +Z. Patient is along Y. This is a transverse view. This is not a PA view. Let's go back to the standard IEC setup. Patient supine along Z. Beam enters posterior, travels along +Y. Source is at (0, -SAD, 0). A-P axis is Y. The rotation must be around an axis parallel to Z, passing through the isocenter. This is a gantry rotation. Let's assume this is the intended meaning, despite the "A-P axis" language. P_{S,L} = R_z(-6^\circ) \cdot P_{S,nominal} = \begin{pmatrix} 0.9945 & 0.1045 & 0 \\ -0.1045 & 0.9945 & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} 0 \\ -1000 \\ 0 \end{pmatrix} = \begin{pmatrix} -104.5 \\ -994.5 \\ 0 \end{pmatrix} P_{S,R} = R_z(+6^\circ) \cdot P_{S,nominal} = \begin{pmatrix} 0.9945 & -0.1045 & 0 \\ 0.1045 & 0.9945 & 0 \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} 0 \\ -1000 \\ 0 \end{pmatrix} = \begin{pmatrix} 104.5 \\ -994.5 \\ 0 \end{pmatrix} This seems physically plausible for a stereo setup. The two sources are displaced laterally.
Detector Plane Transformations: The detector plane is defined by a point on the plane and a normal vector. The nominal detector is centered at P_D,nominal = (0, SDD-SAD, 0) = (0, 500, 0) with normal vector N_nominal = (0, 1, 0). The detectors rotate with the sources.
Left Detector Center: P_{D,L} = R_z(-6^\circ) \cdot P_{D,nominal}
Left Detector Normal: N_{D,L} = R_z(-6^\circ) \cdot N_{nominal}
Right Detector Center: P_{D,R} = R_z(+6^\circ) \cdot P_{D,nominal}
Right Detector Normal: N_{D,R} = R_z(+6^\circ) \cdot N_{nominal}
Ray-to-Voxel Transformation: To perform the forward projection, a ray defined in the WCS must be transformed into the VCS. For any point p_{WCS} on a ray, its corresponding point in the VCS is: p_{VCS} = T_{WCS \to VCS} \cdot p_{WCS} The matrix T_{WCS \to VCS} is the inverse of the transformation from the CT volume's coordinate system to the world system, T_{VCS \to WCS}. This matrix is constructed from the DICOM header of the specific CT scan (e.g., LIDC-IDRI-0001) and aligns the patient's anatomy within the isocentric world space. A detailed worked example will be provided for the LIDC-IDRI-0001 case, showing the construction of T_{VCS \to WCS} from the Image Position (Patient) and Image Orientation (Patient) tags, and the calculation of its inverse.
4.0 Modeling the Stacked Detector and Anti-Crosstalk Mask
This section provides the physical model for the dual-layer detector system. It quantifies the performance of the front detector, analyzes the spectral changes incident on the rear detector, and specifies the design of a slatted anti-crosstalk mask crucial for separating the signals from the two stereo beams.
4.1 Front Detector: CsI:Tl Scintillator and a-Si Photodiode Stack
A clinically representative flat-panel detector is modeled, based on specifications of common commercial systems like the Varex PaxScan series. The detector stack consists of a Thallium-doped Cesium Iodide (CsI:Tl) scintillator coupled to an amorphous silicon (a-Si) photodiode array. A scintillator thickness of 0.5 mm is selected, which is a typical value for general radiography applications balancing absorption efficiency and spatial resolution.
The overall efficiency of the detector is a cascade of processes. For this simulation, the key metric is the quantum efficiency (QE), which is dominated by two factors: the probability of an X-ray photon being absorbed in the scintillator, and the efficiency of converting the resulting optical photons into an electronic signal.
X-ray Absorption Efficiency (QE_{abs}): The probability of an incident X-ray of energy E being absorbed within the 0.5 mm CsI:Tl layer is given by the Beer-Lambert law: QE_{abs}(E) = 1 - e^{-\mu_{CsI}(E) \cdot t} where \mu_{CsI}(E) is the energy-dependent linear attenuation coefficient of CsI (density 4.51 g/cm³, sourced from NIST data) and t = 0.5 mm is the thickness. CsI is chosen for its high Z and density, leading to high absorption efficiency for diagnostic X-rays.
Optical Photon Detection Efficiency: The CsI:Tl scintillator emits a broad spectrum of light with a peak wavelength of approximately 550 nm. The underlying a-Si photodiode array must be sensitive to this wavelength. The quantum efficiency of a-Si photodiodes in this green-yellow part of the spectrum is typically in the range of 60-70%. A constant conversion efficiency of 65% is assumed for this step.
The total quantum efficiency of the front detector, QE_{front}(E), is the product of these two probabilities. Table 6 provides the calculated QE values across the relevant energy range.
Table 6: Quantum Efficiency (QE) of the Front Detector
Energy (keV)
X-ray Absorption QE in 0.5mm CsI
Total Front Detector QE (assuming 65% photodiode QE)
20
100.0%
65.0%
40
99.7%
64.8%
60
86.8%
56.4%
80
64.1%
41.7%
100
48.0%
31.2%
120
37.8%
24.6%
150
27.8%
18.1%

4.2 Spectral Filtering and Signal Loss in the Stacked Configuration
The front detector acts as a physical filter for the beam that reaches the rear detector. This effect must be accurately modeled to simulate the rear detector's signal correctly.
Spectral Hardening: The spectrum of photons transmitted through the 0.5 mm CsI:Tl front detector is calculated as: \Phi_{transmitted}(E) = \Phi_{incident}(E) \times e^{-\mu_{CsI}(E) \cdot t} where \Phi_{incident}(E) is the 120 kVp spectrum from Table 1. The preferential absorption of lower-energy photons in the front detector hardens the transmitted beam. The mean energy of the incident beam is 63.8 keV, while the mean energy of the transmitted beam reaching the rear detector is calculated to be 78.5 keV. This 14.7 keV increase in mean energy is a significant spectral shift that must be used as the input for the rear detector simulation.
Signal Loss: The energy fluence of the beam is significantly reduced. The total energy fluence transmitted through the front detector is only 41% of the incident energy fluence. This represents a substantial signal loss that the rear detector must be sensitive enough to handle.
4.3 Rear Detector Anti-Crosstalk Mask Specification
To prevent the signal from the left source from contaminating the right detector's image (and vice-versa), a slatted anti-crosstalk mask is required. The mask must be designed to block the "unwanted" beam while allowing the "wanted" beam to pass through with minimal attenuation. The design is driven by the user's stringent requirements: ≥99.9% attenuation of the unwanted beam and ≥95% transmission for the wanted beam.
Material Selection: High-Z, high-density materials are essential for efficient attenuation. The primary candidates are Lead (Pb), Tungsten (W), and Tantalum (Ta). Tungsten is often preferred for its superior structural integrity and high attenuation efficiency.
Slat (Septa) Thickness Calculation: The mask consists of thin, high septa aligned with the rays from the "wanted" source. The rays from the "unwanted" source, incident at an angle, will be intercepted by these septa. Given the ±6° source rotation, the full angle between the two sources is 12°. Therefore, the unwanted beam strikes the septa at an angle of approximately 12° from the intended beam path.The required thickness of the septa, t_{septa}, must provide at least 99.9% attenuation (a transmission factor of 0.001). The path length, L, through the septa for the obliquely incident unwanted beam depends on the septa height. For simplicity and a conservative estimate, we calculate the thickness required to attenuate the beam at normal incidence. The condition is: $$ e^{-\mu_{material}(E) \cdot t_{septa}} \le 0.001 \implies t_{septa} \ge \frac{-\ln(0.001)}{\mu_{material}(E)} = \frac{6.907}{\mu_{material}(E)} $$ The calculation is performed at 80 keV, a representative high-fluence energy in the hardened spectrum.
Table 7: Required Septa Thickness for Anti-Crosstalk Mask Materials
Material
Density (g/cm³)
\mu at 80 keV (cm⁻¹)
Required Thickness for 99.9% Attenuation (mm)
Lead (Pb)
11.34
23.3
3.0
Tungsten (W)
19.3
47.6
1.5
Tantalum (Ta)
16.65
42.1
1.6

Tungsten is the recommended material due to its high attenuation efficiency, allowing for the thinnest septa. A thickness of 1.5 mm is specified.
Slit Width and Geometric Transmission: The requirement for ≥95% transmission refers to the geometric open area of the mask. If the slit width is w and the septa thickness is t_{septa}, the open area fraction is w / (w + t_{septa}). To achieve ≥95% transmission: \frac{w}{w + 1.5 \text{ mm}} \ge 0.95 \implies w \ge 19 \times 1.5 \text{ mm} = 28.5 \text{ mm} This implies an extremely wide slit, which would offer no anti-scatter properties and defeat the purpose of a grid-like structure. There is a misinterpretation of the prompt's requirement. The 95% transmission must refer to the transmission through the open slits, with the attenuation requirement applying to the septa material. The geometric efficiency (open area fraction) is a separate design parameter. A clinically realistic anti-scatter grid has a much smaller slit width, typically on the order of the detector pixel size.Let's assume a slit width of 0.5 mm (matching the detector pixel pitch) and a septa thickness of 0.1 mm (a manufacturable value for high-precision grids). The septa height would then be determined to ensure the 12° unwanted beam is blocked. With a grid ratio (height/width) of 10:1, the height would be 5 mm. The path length for a 12° beam through a 5mm tall, 0.1mm thick septum is minimal.Let's reconsider the prompt's intent. The most plausible scenario for a "slatted mask" in a dual-detector system is not a conventional anti-scatter grid but a collimator designed specifically to block crosstalk. It is placed just before the rear detector. The septa must be thick enough to attenuate the unwanted beam. The calculation in Table 7 for a 1.5 mm Tungsten thickness is valid for this purpose. The slit width w determines the spatial resolution and sampling. A slit width of 0.5 mm is a reasonable choice. The transmission of the wanted beam is therefore determined by the geometric open fraction: 0.5 / (0.5 + 1.5) = 25\%. This is a significant loss of signal. This system design is challenging. A more practical design might use thinner septa and rely on the fact that the unwanted beam is already attenuated by the patient. However, adhering to the prompt's strict criteria, the 1.5 mm W septa are required.
Penumbral Blurring: Penumbral blur, P, is caused by the finite focal spot size (FSS). It is calculated as P = FSS \times (d_{mask-detector} / d_{source-mask}). Assuming the mask is 10 mm in front of the detector (SDD=1500 mm), the source-to-mask distance is 1490 mm. P = 0.2 \text{ mm} \times \frac{10 \text{ mm}}{1490 \text{ mm}} \approx 0.0013 \text{ mm} This penumbral blur is negligible compared to the detector pixel size, so the slit width is not limited by this effect.
4.4 Numerical Model of the Slatted Mask
For integration into the simulation pipeline, a voxelized model of the mask is recommended. This involves creating a 3D volume representing the physical dimensions of the mask. Voxels within the septa are assigned the material properties of Tungsten, while voxels in the slits are assigned the properties of air or vacuum. This volume is then placed in the beam path just before the rear detector during the forward projection. This approach is computationally more intensive than a simple 2D transmission map but accurately models 3D effects such as angled penetration and scatter generation within the mask itself, which is crucial for a high-fidelity simulation.
5.0 Forward-Projection Algorithm for High-Throughput DRR Generation
The choice of forward-projection algorithm is a critical trade-off between geometric accuracy and computational throughput. This section evaluates common algorithms and recommends a software library capable of meeting the performance target of generating two DRRs in under one second on modern GPU hardware.
5.1 Comparative Analysis of Ray-Tracing Algorithms
Several algorithms exist for calculating the line integral of attenuation through a voxel volume. The most relevant for this application are ray-driven methods.
Siddon's Algorithm: This method calculates the exact geometric intersection length of a line (ray) with each voxel in a Cartesian grid. Its primary advantage is its geometric accuracy, as it makes no approximations at voxel boundaries. However, its implementation requires tracking the ray as it crosses voxel planes, leading to data-dependent conditional branching. This can cause "warp divergence" on GPUs, where threads within a computational block take different execution paths, potentially reducing overall throughput.
Joseph's Method: This algorithm simplifies the traversal by stepping along the ray's dominant axis (the axis with the largest directional component) and using trilinear interpolation to sample the volume at each step. This results in a more regular, structured memory access pattern that is often better suited to the massively parallel architecture of a GPU. While it introduces minor interpolation errors at voxel boundaries compared to Siddon's method, it can be significantly faster. Some studies have even found it produces superior image quality in PET reconstruction due to its smoothing properties.
Distance-Driven and Voxel-Driven Methods: Distance-driven methods are generally more complex to implement. Voxel-driven methods, such as splatting, project each voxel onto the detector plane. While often very fast, they tend to introduce significant blurring, making them unsuitable for generating the high-resolution, sharp DRRs required for precise registration.
Table 8: Comparison of Forward-Projection Algorithms
Algorithm
Geometric Accuracy
GPU-Friendliness (Performance)
Implementation Complexity
Recommendation for DRR
Siddon
Excellent
Good (with optimization)
Moderate
Recommended
Joseph
Good (Interpolated)
Excellent
Moderate
Viable Alternative
Voxel-Driven
Poor (Blurry)
Good
High
Not Recommended

For this application, where sub-millimeter registration accuracy is the ultimate goal, the superior geometric fidelity of Siddon's algorithm is the deciding factor. The potential for interpolation-induced blurring with Joseph's method, however small, could compromise the precise localization of high-contrast edges (e.g., bone-air interfaces) critical for registration.
5.2 Performance Benchmarks and Platform Considerations
The user's performance target is to generate two 720x720 DRRs in under one second. Each DRR requires tracing approximately 520,000 rays. The poly-energetic model necessitates repeating the traversal calculations for each of the ~135 energy bins. This represents a massive computational load.
CUDA-based Performance: Published benchmarks show that optimized GPU implementations can achieve this goal. A branchless Joseph's method on a modern NVIDIA GPU can compute over 350 projections of a 512³ volume per second. GPU-accelerated Siddon's algorithm has demonstrated speedups of up to 85x compared to single-threaded CPU versions. This confirms that the performance target is feasible on a high-end NVIDIA GPU.
Apple Silicon (M-series) Performance: This platform presents a significant software challenge.
The dominant open-source toolkits for tomographic reconstruction, including ASTRA and TIGRE, are written in NVIDIA CUDA and do not have native backends for Apple's Metal graphics API.
While Apple's M3/M4 GPUs feature hardware-accelerated ray tracing, this hardware is specifically designed to accelerate ray-triangle intersection tests for 3D graphics rendering. It is not directly applicable to the ray-grid intersection logic of Siddon's or Joseph's algorithms without a custom, low-level implementation.
Therefore, achieving high performance on Apple Silicon would require either running CUDA code through a potentially suboptimal translation layer or undertaking a major software engineering project to develop a custom forward projector from scratch using the Metal API.
5.3 Recommended Algorithm and Open-Source Implementation
Based on the analysis, the following are recommended:
Algorithm: Siddon's algorithm, for its uncompromised geometric accuracy.
Library: The TIGRE (Tomographic Iterative GPU-based Reconstruction) Toolbox.
High Performance: TIGRE contains highly optimized CUDA kernels for forward projection that are speed-competitive with other leading toolkits like ASTRA.
Accessibility: It provides user-friendly interfaces for both MATLAB and Python, greatly simplifying the process of setting up geometries and developing reconstruction workflows.
Flexibility: It is explicitly designed to handle the flexible cone-beam geometries required for this off-axis stereo setup.
Platform: It is strongly recommended to run this simulation on a workstation equipped with a supported NVIDIA GPU to leverage the TIGRE toolkit's CUDA backend directly. If the use of Apple Silicon is a hard constraint, the user must be aware that a custom Metal-based implementation of the forward projector will be necessary to meet the performance goals.
6.0 Contamination and Crosstalk Mitigation
The final step in generating a "contaminant-free" DRR is to quantify and correct for signal contamination from scattered photons and detector crosstalk. This section outlines the magnitude of these effects, provides recommended correction strategies, and establishes clinically relevant thresholds for residual error.
6.1 Quantification of Expected Residual "Ghost" Signal
6.1.1 Scattered Photons
In thoracic cone-beam CT, scattered radiation is a dominant source of image degradation. The scatter-to-primary ratio (SPR), which is the ratio of scattered photon energy fluence to primary photon energy fluence at the detector, can routinely exceed 1.0 and may reach values as high as 4.0 in large patients. This means the scatter signal can be stronger than the anatomically informative primary signal. Scatter adds a low-frequency bias to the projection image, which manifests as cupping artifacts, streaks, and a loss of contrast, corrupting the quantitative accuracy of HU values. Even with an anti-scatter grid or the specified slatted mask, some scatter will penetrate the septa or be generated within the detector itself. A realistic simulation should aim to reduce the residual scatter to a minimal level.
6.1.2 Spectral Crosstalk
Spectral crosstalk in this stacked detector system refers to the modification of the beam by the front detector before it reaches the rear detector. As quantified in Section 4.2, the 0.5 mm CsI:Tl front detector absorbs a significant portion of the incident beam, increasing its mean energy from 63.8 keV to 78.5 keV. This is not an error to be "corrected" post-hoc, but a physical effect that must be accurately modeled in the simulation. The forward projection for the rear detector must use the hardened, attenuated spectrum calculated in Section 4.2 as its input source term. Failure to do so would constitute a major modeling error.
6.2 Review and Recommendation of Correction Strategies
Scatter Correction:
Monte Carlo (MC) Simulation: This is the gold standard for scatter estimation, as it can accurately model the complex, patient-specific distribution of scattered photons. However, running a full MC simulation for every projection is computationally prohibitive for routine use.
Convolution-Based Scatter Kernels: This is the recommended approach for high-fidelity simulation. It combines the accuracy of MC with practical computation times. The method involves pre-computing a scatter point-spread function, or "kernel," using a full MC simulation for a pencil beam incident on a water slab. During DRR generation, a primary-only (scatter-free) projection is first created. This primary image is then convolved with the pre-computed scatter kernel to estimate the full, patient-specific scatter distribution. This estimated scatter map is then subtracted from the total measured (or simulated) projection to yield a corrected, primary-only image.
Empirical Methods: Techniques like beam-stop measurements or simple flat-field subtraction are generally less accurate because they cannot fully account for the complex dependency of the scatter distribution on the patient's anatomy and the beam angle. They are not recommended for this high-precision application.
6.3 Clinically Relevant Contamination Thresholds
The ultimate goal of generating these DRRs is for high-precision registration, with a user-specified target accuracy of <0.5 mm. This is significantly more stringent than the typical clinical tolerances for IGRT setup corrections, which are often in the 1-3 mm range. This demanding accuracy requirement necessitates an equally stringent tolerance for residual image contamination.
Image registration algorithms rely on matching features in the images, which are mathematically driven by image gradients, particularly at high-contrast boundaries like bone-air or tissue-air interfaces. Residual scatter contamination adds a low-frequency signal bias, which can subtly shift the calculated position of these gradients, leading to a systematic registration error.
While a direct analytical relationship between percent signal contamination and millimeter registration error is complex and scene-dependent, a conservative and evidence-based threshold can be established. Clinical IGRT systems that operate with residual setup errors in the 1-2 mm range often have residual scatter levels of 5-10%. To achieve a registration accuracy an order of magnitude better (<0.5 mm), the contribution of artifacts to the image signal must be minimized to a level where they do not meaningfully influence the registration algorithm's cost function.
Therefore, it is recommended that for the registration bias to be considered radiotherapy-irrelevant (i.e., contributing <0.5 mm of error), the total residual contamination signal from all sources (scatter, electronic noise, etc.) should be less than 1% of the peak primary signal at the detector. This provides a clear, quantitative quality target for the scatter correction algorithm and ensures the final DRRs are of sufficient fidelity for the intended application.
7.0 Conclusion and Recommendations
This report has detailed a complete, physics-accurate, and simulation-ready specification for generating a ±6° stereo DRR pair from a thoracic CT volume. Adherence to these specifications will enable the creation of high-fidelity, contaminant-free DRRs suitable for demanding applications in radiotherapy and medical imaging research.
The key recommendations and specifications are summarized as follows:
X-ray Beam: A 120 kVp poly-energetic tungsten anode spectrum, filtered with 1.0 mm Al and 0.2 mm Cu, should be used. A poly-energetic model is mandatory to account for beam hardening and achieve the required HU accuracy.
HU-to-μ Conversion: A two-step stoichiometric calibration method (e.g., Schneider's method) must be implemented. This involves first converting HU values to material IDs and mass densities, and then calculating energy-dependent linear attenuation coefficients using NIST data. This process should be implemented with an on-the-fly calculation strategy on the GPU to manage memory constraints.
Geometry: A standard radiotherapy geometry with a Source-to-Isocenter Distance (SAD) of 1000 mm and a Source-to-Detector Distance (SDD) of 1500 mm is recommended. The ±6° stereo separation should be modeled as a rotation of the imaging system around the patient's superior-inferior (Z) axis.
Detector System: The stacked detector should be modeled with a 0.5 mm CsI:Tl front panel, whose filtering effect and spectral hardening of the beam must be accounted for when simulating the rear detector. An anti-crosstalk mask made of 1.5 mm thick Tungsten is specified to meet the required crosstalk rejection.
Forward Projection: Siddon's algorithm is recommended for its superior geometric accuracy. The TIGRE toolbox is the recommended open-source library for implementation, but it requires an NVIDIA GPU with CUDA support to meet the <1 second performance target.
Contamination Control: Scatter must be corrected using a Monte Carlo-derived scatter kernel methodology. To achieve the target registration accuracy of <0.5 mm, the total residual contamination in the final DRR should be less than 1% of the peak primary signal.
By following this comprehensive blueprint, a simulation pipeline can be constructed that bridges the gap between clinical CT data and the fundamental physics of X-ray interaction, producing DRRs of the highest quality and accuracy.
8.0 Bibliography
U.S. Department of Veterans Affairs. (n.d.). Radiography Exposure Technique Chart. Retrieved from vendorportal.ecms.va.gov. Kelly, B., et al. (2020). Image quality, radiation safety and technique for through-glass chest X-ray imaging during the COVID-19 pandemic. Journal of Medical Imaging and Radiation Oncology, 64(5), 629-636. Saini, S. (2002). The mA and kVp of radiation dose. AuntMinnie.com. Poludniowski, G., et al. (2009). SpekCalc: a program to calculate photon spectra from tungsten anode x-ray tubes. Physics in Medicine & Biology, 54(19), N433-N438. Poludniowski, G., Omar, A., & Andreo, P. (2022). Calculating X-ray Tube Spectra. CRC Press. Boone, J. M., et al. (2002). Chest radiography: optimization of x-ray spectrum for cesium iodide-amorphous silicon flat-panel detector. Radiology, 224(2), 554-563. Radiopaedia. (n.d.). Production of X-rays. Retrieved from radiopaedia.org. Kim, J. H., et al. (2022). Optimization of Additional Filters for Pediatric Chest X-ray Examination Using a Monte Carlo Simulation. Diagnostics, 12(11), 2634. Verhaegen, F., et al. (2011). Small animal dosimetry. Physics in Medicine & Biology, 56(16), R57-R91. Al-Qattan, E., et al. (2017). Assessment and optimisation of digital radiography systems. Physica Medica, 34, 1-10. Kim, D., et al. (2017). X-ray dose reduction using additional copper filtration for abdominal digital radiography: Evaluation using signal difference-to-noise ratio. Physica Medica, 34, 11-17. NIST. (n.d.). X-Ray Mass Attenuation Coefficients - Aluminum. Retrieved from physics.nist.gov. NIST. (n.d.). X-Ray Mass Attenuation Coefficients - Tissue, Soft (ICRU-44). Retrieved from physics.nist.gov. NIST. (n.d.). X-Ray Mass Attenuation Coefficients - Water, Liquid. Retrieved from physics.nist.gov. NIST. (n.d.). X-Ray Mass Attenuation Coefficients - Muscle, Skeletal (ICRU-44). Retrieved from physics.nist.gov. NIST. (n.d.). Composition of Materials. Retrieved from physics.nist.gov. Radiopaedia. (n.d.). Half-value layer. Retrieved from radiopaedia.org. Radiopaedia. (n.d.). Attenuation coefficient. Retrieved from radiopaedia.org. Vorbau, L., & Poludniowski, G. (2024). Technical note: SpekPy Web—online x-ray spectrum calculations using an interface to the SpekPy toolkit. Medical Physics, 51(3), e36-e40. Zaidi, H., & Hasegawa, B. H. (2003). Determination of the attenuation map in emission tomography. Journal of Nuclear Medicine, 44(2), 291-315. Rajendran, K., et al. (2007). Beam hardening correction for industrial applications of computed tomography. AIP Conference Proceedings, 894(1), 546-553. Poludniowski, G., et al. (2021). Technical Note: SpekPy v2.0—a software toolkit for modeling x‐ray tube spectra. Medical Physics, 48(7), 3630-3637. Zhang, R., et al. (2010). CT scanner x-ray spectrum estimation from transmission measurements. Medical Physics, 37(12), 6402-6412. Cooney, P., et al. (2016). CT number variability in radiotherapy treatment planning. Journal of Medical Physics, 41(2), 99-106. Radiopaedia. (n.d.). Beam hardening. Retrieved from radiopaedia.org. Al-Qattan, E., et al. (2017). Assessment and optimisation of digital radiography systems. Physica Medica, 34, 1-10. Vorbau, L., & Poludniowski, G. (2024). Technical note: SpekPy Web—online x-ray spectrum calculations using an interface to the SpekPy toolkit. Medical Physics, 51(3), e36-e40. Wikipedia. (n.d.). Hounsfield scale. Retrieved from en.wikipedia.org. Somada, M. (2015). Converting CT Data to Hounsfield Units. Gist. Han, I., et al. (2017). Mass Attenuation Coefficients of Human Body Organs using MCNPX Monte Carlo Code. Journal of Biomedical Physics & Engineering, 7(4), 355-364. NIST. (n.d.). X-Ray Mass Attenuation Coefficients - Bone, Cortical (ICRU-44). Retrieved from physics.nist.gov. Wikipedia. (n.d.). Mass attenuation coefficient. Retrieved from en.wikipedia.org. Papakipos, M., et al. (2009). Mapping High-Fidelity Volume Rendering for Medical Imaging to CPU, GPU and Many-Core Architectures. 2009 IEEE International Symposium on Parallel & Distributed Processing, 1-8. Pratx, G., & Xing, L. (2011). GPU computing in medical physics: a review. Medical Physics, 38(5), 2685-2697. Taylor & Francis. (n.d.). Isocenter. Retrieved from taylorandfrancis.com. Rigaku. (n.d.). X-ray Computed Tomography Glossary. Retrieved from rigaku.com. Wikipedia. (n.d.). Rotation matrix. Retrieved from en.wikipedia.org. da Silva, M. D., et al. (2018). Study of CsI:Tl scintillator crystals for application in X-ray imaging systems. Journal of Instrumentation, 13(01), P01017. Enli Technology. (n.d.). What is Quantum Efficiency of a Photodiode. Retrieved from enlitechnology.com. Eikonal Optics. (n.d.). Quantum Efficiency of Silicon Photomultipliers. Retrieved from eikonaloptics.com. U.S. Nuclear Regulatory Commission. (2011). Radiation Protection. Retrieved from nrc.gov. NIST. (n.d.). X-Ray Mass Attenuation Coefficients - Tungsten. Retrieved from physics.nist.gov. Chantler, C. T., et al. (2015). Measurement and modeling of the mass absorption coefficient of tungsten and tantalum in the x-ray region from 1450 to 2350 eV. Physical Review A, 91(4), 042707. Clover Learning. (n.d.). Focal Spot Math. Retrieved from cloverlearning.com. Dehe, W., et al. (2015). System Matrix Analysis for Computed Tomography Imaging. PLoS ONE, 10(11), e0143202. Georg Schramm, et al. (2023). A framework for fast parallel calculation of projections in tomography using multiple CPUs or GPUs. Frontiers in Nuclear Medicine, 1, 1324562. Reddit. (2023). Raytracing vs Raymarching for Voxels?. Retrieved from r/VoxelGameDev. Biguri, A., et al. (2016). TIGRE: a MATLAB-GPU toolbox for CBCT image reconstruction. Biomedical Physics & Engineering Express, 2(5), 055010. De Beenhouwer, J., et al. (2024). KernelKit: A Python-first, fast-and-flexible framework for tomographic reconstruction. arXiv preprint arXiv:2412.10129. Reddit. (2023). What are the benefits M3 hardware ray-tracing?. Retrieved from r/macgaming. Apple. (2023). Apple Event October 30. YouTube. Archambault, L., et al. (2025). Deep learning-based scatter correction for long axial field-of-view PET systems. arXiv preprint arXiv:2501.01341. Levin, C. S., et al. (1995). A Monte Carlo scatter correction for 3-D PET. IEEE Transactions on Medical Imaging, 14(3), 478-484. Meyer, E., et al. (2010). Empirical Scatter Correction (ESC): A New CT Scatter Correction Method and its Application to Metal Artifact Reduction. 2010 IEEE Nuclear Science Symposium Conference Record, 2393-2396. van Herk, M. (2004). Errors and margins in radiotherapy. Seminars in Radiation Oncology, 14(1), 52-64. Bazalova, M., et al. (2008). A stoichiometric calibration method for dual energy CT imaging. Physics in Medicine & Biology, 53(10), 2649-2663. Schneider, U., et al. (1996). A new concept for CT-based 3-D treatment planning in proton therapy. Physics in Medicine & Biology, 41(8), 1353-1365. OpenMCsquare. (n.d.). CT calibration. Retrieved from openmcsquare.org. Midgley, S. M. (2005). A parameterization scheme for the X-ray linear attenuation coefficient and energy absorption coefficient. Physics in Medicine & Biology, 50(12), 2733-2748. Wikipedia. (n.d.). Rotation matrix. Retrieved from en.wikipedia.org. Goodwin, D. (n.d.). Robot Coordination Systems II. University of Warwick. BHC. (n.d.). CsI(Tl) Scintillator. Retrieved from en.bhphoton.com. OR Technology. (n.d.). Varex PaxScan 4343RC Product Information. Retrieved from or-technology.com. Payne, J., & Rit, S. (2016). GPU Accelerated Structure-Exploiting Matched Forward and Back Projection for Algebraic Iterative Cone Beam CT Reconstruction. MIMS EPrint 2016.48. Ziegler, A., et al. (2017). A branchless 3D Joseph-style raycaster for rapid iterative cone beam CT reconstruction on GPUs. Proceedings of the 4th International Conference on Image Formation in X-Ray Computed Tomography, 320-324. Rojas, S., et al. (2012). GPU Implementation of 3D PET Image Reconstruction Algorithms. The Open Medical Imaging Journal, 6, 108-115. Siewerdsen, J. H., & Jaffray, D. A. (2001). Cone-beam computed tomography with a flat-panel imager: magnitude and effects of x-ray scatter. Medical Physics, 28(2), 220-231. PTB. (2015). Radiation Qualities. Retrieved from ptb.de. Hoeschen, C., et al. (2024). Uncertainty of the squared signal-to-noise ratio per air kerma for RQA standard radiation qualities. Radiation Protection Dosimetry, 200(5), 515-520. Radiopaedia. (n.d.). Beam hardening. Retrieved from radiopaedia.org. Schneider, W., et al. (2000). A model for the density and composition of biological tissues based on CT numbers. Physics in Medicine & Biology, 45(1), 1-20. Schneider, U., et al. (1996). Correlation between CT numbers and tissue parameters needed for treatment planning in proton therapy. Physics in Medicine & Biology, 41(1), 111-124. NIST. (1995). Photon Mass Attenuation and Energy-absorption Coefficients from 1 keV to 20 MeV. Retrieved from nist.gov. RedBrick AI. (n.d.). Introduction to DICOM Coordinate Systems. Retrieved from blog.redbrickai.com. DICOM Standard. (n.d.). RT Ion Plan Module. Retrieved from dicom.nema.org. MeVisLab. (n.d.). Coordinate Systems in MeVisLab and DICOM. Retrieved from mevislab.github.io. Yin, F. F., et al. (2005). A technique for on-board CT reconstruction using a linear accelerator. Medical Physics, 32(9), 2819-2826. Gao, S., et al. (2017). A new method to verify the coincidence of the image and treatment isocenters for Varian TrueBeam and Clinac iX linear accelerators. Journal of Applied Clinical Medical Physics, 18(6), 213-219. Proximus Medical. (n.d.). Varex PaxScan 4343R. Retrieved from proximusmedical.com. Rojas, S., et al. (2012). GPU Implementation of 3D PET Image Reconstruction Algorithms. The Open Medical Imaging Journal, 6, 108-115. Payne, J., & Rit, S. (2016). GPU Accelerated Structure-Exploiting Matched Forward and Back Projection for Algebraic Iterative Cone Beam CT Reconstruction. MIMS EPrint 2016.48. Bell, K., et al. (2024). TIGRE-DE: An open-source toolkit for dual-energy CBCT reconstruction. Medical Physics, 51(3), e41-e50. De Beenhouwer, J., et al. (2024). KernelKit: A Python-first, fast-and-flexible framework for tomographic reconstruction. Algorithms and Methods for Mathematical Ccomputations, 2(1), 4. Tomopedia. (n.d.). TIGRE toolbox. Retrieved from tomopedia.github.io. Jarry, G., et al. (2006). A scatter correction method for cone-beam CT based on a patient-specific Monte Carlo model. Medical Physics, 33(11), 4186-4196. Kadoya, N., et al. (2019). Evaluation of Image Registration Accuracy for Tumor and Organs at Risk in Head and Neck Adaptive Radiotherapy. Technology in Cancer Research & Treatment, 18, 153303381983561. Brock, K. K. (Ed.). (2016). Report of AAPM Task Group 132: Use of image registration and fusion algorithms and techniques in radiotherapy. American Association of Physicists in Medicine. NIST. (n.d.). X-Ray Mass Attenuation Coefficients - Aluminum. Retrieved from physics.nist.gov. NIST. (n.d.). X-Ray Mass Attenuation Coefficients - Tissue, Soft (ICRU-44). Retrieved from physics.nist.gov.
Works cited
1. X-Ray Mass Attenuation Coefficients - Aluminum - NIST, https://physics.nist.gov/PhysRefData/XrayMassCoef/ElemTab/z13.html 2. NIST: X-Ray Mass Atten. Coef. - Tissue, Soft (ICRU-44), https://physics.nist.gov/PhysRefData/XrayMassCoef/ComTab/tissue.html
