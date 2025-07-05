# CT Preprocessing Pipeline for High-Fidelity DRR Synthesis

A comprehensive Python pipeline for preprocessing thoracic CT scans to prepare them for high-quality Digitally Reconstructed Radiograph (DRR) synthesis. This implementation follows evidence-based best practices for CT preprocessing as detailed in the research documentation.

## Features

- **Isotropic Resampling**: Converts anisotropic CT volumes to isotropic spacing using B-spline interpolation
- **Noise Suppression**: Adaptive denoising with Non-Local Means (NLM) filtering
- **Slice Thickness Harmonization**: Addresses thick slice artifacts through interpolation-based super-resolution
- **Artifact Correction**: Removes metal artifacts, beam hardening, and optional ring artifacts
- **HU Calibration**: Phantom-free calibration using internal tissue references
- **Attenuation Conversion**: Converts calibrated HU values to linear attenuation coefficients
- **Quality Control**: Automated QC metrics including SSIM, edge preservation, and HU accuracy checks

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ct-preprocessing-drr.git
cd ct-preprocessing-drr
```

2. Install `uv` package manager (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Install dependencies:
```bash
uv sync
```

## Usage

### Basic Usage

Process a DICOM series with default settings:

```bash
source .venv/bin/activate
python preprocess.py /path/to/dicom/directory -o output_directory
```

### Command-Line Options

```bash
python preprocess.py [OPTIONS] DICOM_PATH

Arguments:
  DICOM_PATH    Path to directory containing DICOM files

Options:
  -o, --output PATH           Output directory (default: preprocessed_output)
  -c, --config PATH           Configuration file (JSON)
  --spacing FLOAT             Target isotropic spacing in mm (default: 0.5)
  --no-denoise               Skip denoising step
  --no-artifacts             Skip artifact correction
  --save-intermediate        Save intermediate results
```

### Configuration

Create a custom configuration file (JSON) to override default settings:

```json
{
  "resampling": {
    "enabled": true,
    "target_spacing": 0.5,
    "interpolator": "bspline"
  },
  "denoising": {
    "enabled": true,
    "method": "nlm",
    "adaptive": true
  },
  "artifact_correction": {
    "enabled": true,
    "correct_metal": true,
    "correct_beam_hardening": true,
    "correct_rings": false
  }
}
```

## Pipeline Stages

1. **DICOM Loading**: Reads DICOM series and extracts metadata
2. **Isotropic Resampling**: Resamples to uniform voxel spacing (default: 0.5mm)
3. **Slice Thickness Harmonization**: Addresses anisotropic acquisition artifacts
4. **Noise Suppression**: Applies adaptive denoising based on estimated noise level
5. **Artifact Correction**: Removes common CT artifacts
6. **HU Calibration**: Ensures accurate Hounsfield Unit values
7. **Quality Control**: Generates comprehensive QC report

## Output

The pipeline generates:
- `preprocessed_hu.nii.gz`: Preprocessed CT volume in Hounsfield Units
- `preprocessed_mu.nii.gz`: Linear attenuation coefficient volume
- `processing_info.json`: Detailed processing log
- `qc_report.json`: Quality control metrics

## Requirements

- Python 3.12+
- Apple Silicon Mac (optimized for M-series chips) or x86_64 Linux/Windows
- At least 8GB RAM (48GB recommended for large volumes)

## Dataset

This pipeline was developed and tested using the Lung Image Database Consortium image collection (LIDC-IDRI):

> Clark K, Vendt B, Smith K, et al. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. Journal of Digital Imaging. 2013; 26(6): 1045-1057.

## Development

See `CLAUDE.md` for development guidelines and `research/` directory for detailed technical documentation.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this preprocessing pipeline in your research, please cite:

```bibtex
@software{ct_preprocessing_drr,
  title = {CT Preprocessing Pipeline for High-Fidelity DRR Synthesis},
  year = {2024},
  url = {https://github.com/yourusername/ct-preprocessing-drr}
}
```