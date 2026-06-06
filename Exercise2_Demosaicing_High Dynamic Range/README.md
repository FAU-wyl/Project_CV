# Exercise 2: Demosaicing and High Dynamic Range Imaging

## Overview

This exercise provides a comprehensive exploration of **Bayer demosaicing** and **High Dynamic Range (HDR)** image processing techniques. The project covers the complete pipeline from raw camera sensor data (CR3 files) to final high-quality JPEG images, including white balance correction, tone mapping, and local contrast enhancement.

## Project Structure

```
Exercise2_Demosaicing_High Dynamic Range/
├── config.py                    # Configuration for Bayer patterns and data paths
├── bayer.py                     # Bayer demosaicing algorithms
├── pipeline.py                  # Main processing pipeline (Exercises 2-8)
├── utils.py                     # Utility functions for image I/O and processing
├── run.py                       # Entry point to execute all exercises
├── streamlit_app.py             # Interactive web interface (Streamlit)
├── test.py                      # Testing utilities
├── requirements.txt             # Python dependencies
└── exercise_2_data/             # Sample data directory
    ├── 01/                      # Exercise 1: Bayer pattern identification data
    ├── 02/                      # Exercise 2-4, 8: Single frame processing
    ├── 05/                      # Exercise 5: Sensor linearity verification
    └── 06/                      # Exercise 6-7: HDR bracketed exposures
```

## Key Features

### Exercise 1: Bayer Pattern Identification
- Automatically estimate Bayer color arrangement from camera RAW data
- Uses reference color areas from JPG for statistical analysis
- Validates against multiple Bayer patterns (RGGB, GRBG, GBRG, BGGR)

### Exercise 2: Basic Demosaicing
- Convert Bayer RAW data to full RGB using simple interpolation
- Percentile normalization to handle wide dynamic range
- Basic preview output as JPEG

### Exercise 3: Brightness Mapping Analysis
- Compare multiple tone mapping curves:
  - Gamma correction (γ = 0.3, 0.4, 0.5)
  - Logarithmic curve for dynamic range compression
  - Normalized linear mapping
- Visual comparison of different brightness representations

### Exercise 4: White Balance Correction
- Implement Gray World white balance algorithm
- Assumes image average color should be close to neutral gray
- Independent per-channel gain adjustment to match overall image mean
- Applied before brightne ss mapping for more natural results

### Exercise 5: Sensor Linearity Verification
- Verify linear response of camera sensor using bracketed exposures
- Multiple exposures with different shutter speeds (1/10s to 1/320s)
- Linear regression analysis with R² coefficient
- Confirms sensor output is proportional to exposure time

### Exercise 6: HDR Image Merging
- Merge multiple bracketed exposures into single HDR RAW
- Lecture method: Replace saturated pixels with values from shorter exposures
- Automatic adjustment for exposure differences (2× multiplier per step)
- Output: HDR image with extended dynamic range

### Exercise 7: iCAM06 Local Tone Mapping
- Implement simplified iCAM06 algorithm for local contrast enhancement
- Separate luminance into:
  - **Base layer**: Low-frequency component (bilateral filtering)
  - **Detail layer**: High-frequency component
- Compress dynamic range independently while preserving local contrast
- Operate in logarithmic domain for better HDR handling

### Exercise 8: Complete RAW-to-JPG Pipeline
- End-to-end processing from CR3 RAW to final JPEG:
  1. Black level subtraction
  2. Bayer demosaicing
  3. Gray World white balance
  4. Percentile normalization
  5. Gamma correction
  6. Brightness/contrast/saturation enhancement
  7. Final display adjustment
  8. JPEG export

## Technical Details

### Bayer Pattern Explanation

Camera sensors use Bayer color filter arrays to capture color information:

```
RGGB pattern:    GRBG pattern:
R G              G R
G B              B G

GBRG pattern:    BGGR pattern:
G B              B G
R G              G R
```

Each 2×2 unit contains:
- 1 Red pixel
- 2 Green pixels (in different positions)
- 1 Blue pixel

### Demosaicing Algorithm

The simple demosaicing uses mask-based interpolation:

1. Create binary masks for each color channel marking real sensor positions
2. Apply convolution to aggregate neighboring pixels of same color
3. Normalize by number of samples in neighborhood
4. Result: Full-resolution RGB from Bayer pattern

```python
Channel = (Raw ⊗ Mask) ⊗ Kernel / (Mask ⊗ Kernel)
```

Where ⊗ denotes convolution.

### White Balance (Gray World Assumption)

Adjusts each color channel by:
```
ScaleFactor_channel = MeanImageIntensity / MeanChannelIntensity
```

This assumes the average color should be neutral gray, correcting color casts.

### iCAM06 Tone Mapping

Separates luminance into base and detail components in log domain:

1. Extract input intensity: `I = (20R + 40G + B) / 61`
2. Calculate color ratios for later restoration
3. In log domain:
   - `log_base = bilateral_filter(log(I))`
   - `log_details = log(I) - log_base`
4. Compress base layer: `compression = log(output_range) / (max - min)`
5. Recombine: `output = exp(log_base × compression + offset + log_details)`
6. Restore colors using original ratios

## Installation & Setup

### Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` - Numerical computations
- `scipy` - Scientific algorithms
- `opencv-python` - Image processing and tone mapping
- `rawpy` - CR3 RAW file reading
- `Pillow` - Image I/O
- `matplotlib` - Plotting and visualization
- `streamlit` - Web interface (optional)

### Data Setup

Place image files in the appropriate `exercise_2_data/` subdirectories:

- **Exercise 1**: `01/IMG_9939.npy`, `01/IMG_9939.JPG`
- **Exercise 2-4, 8**: `02/IMG_4782.CR3`
- **Exercise 5**: `05/IMG_3044.CR3` through `05/IMG_3049.CR3`
- **Exercise 6-7**: `06/00.CR3` through `06/10.CR3`

## Usage

### Run All Exercises

```bash
python run.py
```

This executes exercises 1-8 sequentially, generating output images:
- `exercise2_demosaicing_preview.jpg`
- `exercise3_*.jpg` (normalized, gamma variants, log curve)
- `exercise4_white_balance_*.jpg` (normalized, gamma variants, log curve)
- `ex5_linearity_plot.png` (sensor linearity graph)
- `ex6_hdr_result.png` (HDR with log mapping)
- `ex7_icam06_fine.png` (iCAM06 tone mapped result)
- `exercise8_process_raw_result.jpg` (complete pipeline output)

### Interactive Web Interface

```bash
streamlit run streamlit_app.py
```

Launch a local web browser interface for real-time parameter tuning and preview.

### Individual Exercise Usage

```python
from pipeline import exercise2_demosaic, process_raw
from config import EX2_RAW, PATTERN_CR3

# Exercise 2: Basic demosaicing
exercise2_demosaic(
    raw_path=EX2_RAW,
    output_path="my_output.jpg",
    pattern=PATTERN_CR3
)

# Exercise 8: Full pipeline with custom parameters
process_raw(
    raw_path="image.CR3",
    jpg_path="output.jpg",
    pattern=PATTERN_CR3,
    gamma=0.38,
    enhance_brightness=0.10,
    enhance_contrast=1.75,
    enhance_saturation=1.25
)
```

## Configuration

Edit `config.py` to adjust:

```python
# Bayer patterns for different camera sources
PATTERN_CR3 = "RGGB"        # Canon CR3 files
PATTERN_NPY = "GBRG"        # Numpy RAW data

# Data directory paths
DATA_DIR = "exercise_2_data"
EX1_NPY = f"{DATA_DIR}/01/IMG_9939.npy"
EX2_RAW = f"{DATA_DIR}/02/IMG_4782.CR3"
EX5_DIR = f"{DATA_DIR}/05"
EX6_DIR = f"{DATA_DIR}/06"
```

## Advanced Parameters

### White Balance
- Operates in full-image statistics
- No parameter tuning required
- Results depend on image content composition

### Tone Mapping (Exercise 3)
```python
exercise3_luminosity(
    raw_path=EX2_RAW,
    pattern=PATTERN_CR3
)
```
Outputs:
- `exercise3_normalized.jpg` - Linear percentile normalization
- `exercise3_gamma_03.jpg` - γ = 0.3 (brightest)
- `exercise3_gamma_04.jpg` - γ = 0.4 (balanced)
- `exercise3_gamma_05.jpg` - γ = 0.5 (darkest)
- `exercise3_log_curve.jpg` - Logarithmic compression

### iCAM06 Parameters
```python
ex7_apply_icam06(
    hdr_bgr_float,
    output_range=6.0,        # Dynamic range compression ratio
    d=15,                    # Bilateral filter kernel diameter
    sigma_color=0.8,         # Color space sigma
    sigma_space=4.0          # Coordinate space sigma
)
```

### Process Raw Parameters
```python
process_raw(
    raw_path="image.CR3",
    jpg_path="output.jpg",
    black_percentile=0.01,       # Black level estimation
    norm_low=0.01,               # Normalization percentiles
    norm_high=99.99,
    gamma=0.38,                  # Tone curve exponent
    enhance_brightness=0.10,     # Enhancement factors
    enhance_contrast=1.75,
    enhance_saturation=1.25,
    final_brightness=0.0,        # Final display adjustment
    final_contrast=1.0,
    jpg_quality=99               # JPEG compression quality
)
```

## Output Files

| File | Exercise | Description |
|------|----------|-------------|
| `exercise2_demosaicing_preview.jpg` | 2 | Simple demosaiced RGB output |
| `exercise3_*.jpg` | 3 | Different tone mapping comparisons |
| `exercise4_white_balance_*.jpg` | 4 | White-balanced results with tone variations |
| `ex5_linearity_plot.png` | 5 | Sensor linearity verification graph |
| `ex6_hdr_result.png` | 6 | Merged HDR with log tone mapping |
| `ex7_icam06_fine.png` | 7 | iCAM06 local tone mapped result |
| `exercise8_process_raw_result.jpg` | 8 | Final enhanced output |

## Performance Notes

- **Demosaicing**: ~50-200ms per megapixel image
- **White Balance**: ~100-300ms per image
- **iCAM06 Tone Mapping**: ~1-3 seconds (due to bilateral filtering)
- **HDR Merging**: ~500ms-2s depending on image count and resolution

Memory requirements scale with image resolution (typically 100-500 MB for 20-40 MP images).

## Troubleshooting

### Issue: "No CR3 files found"
- Verify image files exist in `exercise_2_data/` subdirectories
- Check file paths in `config.py` match actual data location

### Issue: Demosaicing produces strange colors
- Verify correct Bayer pattern in `config.py`
- Test with `test_four_bayer_patterns()` to visually compare all patterns
- Output files will be named `final_test_pattern_*.jpg`

### Issue: White balance removes all color
- This is normal for images with mixed lighting
- Gray World assumes the average image pixel is neutral
- For images with strong color dominance, consider scene-dependent white balance

### Issue: Bilateral filter is too slow
- Reduce `d` parameter (kernel diameter) in `ex7_apply_icam06()`
- Use smaller `sigma_color` and `sigma_space` values
- Consider using Gaussian filtering instead for real-time applications

## References

- **Bayer Demosaicing**: "Demosaicing" - B. E. Bayer, 1976
- **iCAM06**: "Improved Color Appearance Model with Application to Image Quality and Gamut Mapping" - F. Ebner & M. D. Fairchild, 2013
- **Tone Mapping**: "High Dynamic Range Imaging" - E. Reinhard et al., 2010
- **Gray World White Balance**: "Computational Models of Color Constancy" - B. V. Funt et al., 1992

## License

This educational project is provided as-is for learning purposes.

## Authors

Computer Vision Exercise Series - Exercise 2

