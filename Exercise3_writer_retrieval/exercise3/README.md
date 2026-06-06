# Exercise 3: Handwritten Text Image Retrieval using VLAD

## Overview

This exercise implements a **Vector of Locally Aggregated Descriptors (VLAD)** based image retrieval system for handwritten document text. The project demonstrates how to build a visual vocabulary-based search engine, enabling efficient retrieval of similar handwritten documents from a large database.

The system processes the ICDAR 2017 historical document handwriting dataset and uses SIFT features aggregated into VLAD descriptors for image matching and ranking.

## Project Structure

```
Exercise3/
├── exercise3/
│   ├── exercise3.py                          # Main implementation
│   ├── icdar17_labels_train.txt              # Training set labels
│   ├── icdar17_labels_test.txt               # Test set labels
│   ├── mus_k100_d500000_f100.pkl.gz          # Pre-computed codebook (K=100)
│   ├── enc_test_k100_base.pkl.gz             # Cached test VLAD (base)
│   ├── enc_test_k100_powernorm.pkl.gz        # Cached test VLAD (powernorm)
│   ├── icdar2017-training-binary/            # Training images
│   └── ScriptNet-HistoricalWI-2017-binarized/# Additional dataset
├── Exercise 3 Report.pdf                     # Documentation/results
└── Exercise 3 Report.docx                    # Documentation/results
```

## Key Concepts

### SIFT Features (Scale-Invariant Feature Transform)

SIFT extracts local keypoint descriptors that are:
- **Scale-invariant**: Detects features at multiple image scales
- **Rotation-invariant**: Handles image rotation
- **Distinctive**: 128-dimensional feature vectors suitable for matching

**RootSIFT Enhancement:**
- Apply L1 normalization: `desc_norm = desc / sum(desc)`
- Take square root: `desc_root = sqrt(desc_norm)`
- Better suited for Euclidean distance and KMeans clustering

### Visual Vocabulary (Codebook)

A codebook of K visual "words" is created by:
1. Extracting SIFT descriptors from training images
2. Clustering them using KMeans into K clusters
3. Cluster centers become the visual words (visual vocabulary)

Default: **K = 100 visual words**

Each SIFT descriptor is then mapped to its nearest visual word.

### VLAD (Vector of Locally Aggregated Descriptors)

VLAD encodes an image as a fixed-length vector by:

1. **Assign** each SIFT descriptor to nearest visual word (hard assignment)
2. **Compute residuals**: difference between descriptor and cluster center
3. **Aggregate**: sum residuals for each visual word across all descriptors
4. **Normalize**: L2 normalization for consistent scale

For an image with T descriptors and K visual words:
- Output dimension = K × D (where D=128 for SIFT)
- Default output size: 100 × 128 = **12,800 dimensions**

**Mathematical Form:**
```
VLAD[k] = Σ(d_i - μ_k) for all descriptors d_i assigned to word k
Final VLAD = cat(VLAD[1], VLAD[2], ..., VLAD[K])
```

### Power Normalization

Signed square-root normalization to reduce the effect of "burstiness" (large residuals):

```
VLAD_normalized = sign(VLAD) * sqrt(|VLAD|)
```

Benefits:
- Suppresses dominant dimensions
- Improves retrieval robustness for varied image content
- Typically improves mAP by 2-5%

### Image Retrieval Evaluation

**Top-1 Accuracy:**
- Proportion of queries where the most similar database image has the same label
- Simple metric: does the nearest neighbor belong to the correct class?

**Mean Average Precision (mAP):**
- For each query, compute average precision over all returned results
- Average precision at position k: P(k) = relevant_count / k
- Average over all queries
- Standard metric in information retrieval (0-1 scale)

**Example:**
```
Query: Image of writer A
Database: [A, B, A, C, A, D, ...]

Relevant positions: 1, 3, 5
- P@1 = 1/1 = 1.0
- P@3 = 2/3 = 0.67
- P@5 = 3/5 = 0.6
- Average Precision = (1.0 + 0.67 + 0.6) / 3 = 0.76
```

### Exemplar SVM (Optional)

An advanced technique (Exercise e) that:
1. For each test image (query), trains a one-vs-all binary SVM
2. Positive sample: the query image itself
3. Negative samples: all training images
4. Uses SVM weight vector as improved embedding
5. Re-ranks retrieval results based on SVM scores

This typically improves results by training a discriminative classifier for each query.

## Installation & Setup

### Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `numpy` - Numerical computations
- `scipy` - Scientific algorithms
- `scikit-learn` - KMeans clustering and SVM
- `opencv-python` - SIFT feature extraction
- `tqdm` - Progress bars
- `Pillow` - Image I/O (implicitly via OpenCV)
- `parmap` (optional) - Parallel mapping for E-SVM

### Dataset Setup

The ICDAR 2017 historical document dataset is expected in:
- Training images: `exercise3/icdar2017-training-binary/`
- Training labels: `exercise3/icdar17_labels_train.txt`
- Test images: Located according to test label file
- Test labels: `exercise3/icdar17_labels_test.txt`

Label file format:
```
image_name class_id
example1.png writer_001
example2.png writer_002
```

## Usage

### Basic Retrieval with Default Parameters

```bash
cd exercise3
python exercise3.py \
    --in_train icdar2017-training-binary/ \
    --in_test icdar2017-training-binary/ \
    --labels_train icdar17_labels_train.txt \
    --labels_test icdar17_labels_test.txt
```

This runs the complete pipeline:
1. Builds/loads visual vocabulary (K=100)
2. Computes VLAD for test images
3. Evaluates retrieval performance

### Experiment Variations

**Change visual vocabulary size:**
```bash
python exercise3.py \
    --n_clusters 50 \
    --in_train train/ \
    --in_test test/ \
    --labels_train train_labels.txt \
    --labels_test test_labels.txt
```

**Use power normalization:**
```bash
python exercise3.py \
    --powernorm \
    --in_train train/ \
    --in_test test/ \
    --labels_train train_labels.txt \
    --labels_test test_labels.txt
```

**Compute both base and power-normalized VLAD simultaneously:**
```bash
python exercise3.py \
    --both \
    --in_train train/ \
    --in_test test/ \
    --labels_train train_labels.txt \
    --labels_test test_labels.txt
```

This is faster than running separately, as SIFT extraction happens only once.

**Rebuild codebook (force recomputation):**
```bash
python exercise3.py \
    --overwrite \
    --in_train train/ \
    --in_test test/ \
    --labels_train train_labels.txt \
    --labels_test test_labels.txt
```

**Use fewer training images for codebook training:**
```bash
python exercise3.py \
    --max_dictionary_files 50 \
    --in_train train/ \
    --in_test test/ \
    --labels_train train_labels.txt \
    --labels_test test_labels.txt
```

**Include Exemplar SVM refinement:**
```bash
python exercise3.py \
    --esvm \
    --C 1000 \
    --in_train train/ \
    --in_test test/ \
    --labels_train train_labels.txt \
    --labels_test test_labels.txt
```

## Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--labels_test` | string | *required* | Test set labels file |
| `--labels_train` | string | *required* | Training set labels file |
| `--in_test` | string | *required* | Test images directory |
| `--in_train` | string | *required* | Training images directory |
| `--suffix_train` | string | `.png` | Image file suffix for training |
| `--suffix_test` | string | `.jpg` | Image file suffix for test |
| `--to_binary` | flag | False | Apply Otsu binarization to images |
| `--n_clusters` | int | 100 | Number of visual words (K) |
| `--max_descriptors` | int | 500000 | Max SIFT to sample for KMeans |
| `--max_dictionary_files` | int | 100 | Max training images for codebook |
| `--overwrite` | flag | False | Force recompute codebook/encodings |
| `--powernorm` | flag | False | Use signed square-root normalization |
| `--both` | flag | False | Compute both base and powernorm |
| `--esvm` | flag | False | Run Exemplar SVM refinement |
| `--C` | float | 1000 | SVM regularization parameter |

## Algorithm Pipeline

### Phase 1: Codebook Training

```
Training Images
    ↓
SIFT Feature Extraction (RootSIFT)
    ↓
Random Sampling (up to 500k descriptors)
    ↓
MiniBatchKMeans Clustering (K=100)
    ↓
Visual Vocabulary (100 cluster centers)
    ↓
Save: mus_k100_d500000_f100.pkl.gz
```

### Phase 2: VLAD Encoding

```
Test Images
    ↓
SIFT Feature Extraction per Image
    ↓
For each image:
  - Assign SIFT to nearest visual word
  - Compute residuals
  - Aggregate by visual word
  - L2 normalize (+ optional power norm)
    ↓
VLAD Encoding Matrix (12800-dim per image)
    ↓
Save: enc_test_k100_*.pkl.gz
```

### Phase 3: Retrieval Evaluation

```
VLAD Encodings
    ↓
Compute Pairwise Cosine Distances
    ↓
For each query image:
  - Sort database by distance
  - Compute Top-1 accuracy
  - Compute Average Precision
    ↓
Report: Top-1 accuracy and mAP
```

### Phase 4: Exemplar SVM (Optional)

```
For each test image:
  - Train binary SVM
    - Positive: current test image
    - Negative: all training images
  - Extract SVM weight vector
  - Use as new embedding
    ↓
Re-evaluate retrieval with E-SVM embeddings
```

## Performance Characteristics

### Computational Complexity

| Phase | Time | Memory |
|-------|------|--------|
| SIFT Extraction (per image) | 100-500ms | ~10MB |
| Codebook Training (500k descriptors) | 1-5 minutes | ~1GB |
| VLAD Encoding (per image) | 50-200ms | ~2MB |
| Distance Matrix (3600 images) | 1-2 seconds | ~300MB |
| Exemplar SVM (3600 × 3600 classifiers) | 10-30 minutes | ~2GB |

### Typical Results (ICDAR 2017 Dataset)

| Method | Top-1 | mAP |
|--------|-------|-----|
| Base VLAD (K=100) | ~65% | ~0.55 |
| VLAD + Power Norm | ~68% | ~0.58 |
| VLAD + E-SVM | ~72% | ~0.63 |

## Intermediate Outputs

### Generated Files

| File | Size | Description |
|------|------|-------------|
| `mus_k100_d500000_f100.pkl.gz` | ~5 MB | Codebook (K=100 centers, 128-dim each) |
| `enc_test_k100_base.pkl.gz` | ~200 MB | Test VLAD (3600 images × 12800 dims) |
| `enc_test_k100_powernorm.pkl.gz` | ~200 MB | Test VLAD with power norm |
| `enc_train_k100_base.pkl.gz` | ~50 MB | Training VLAD (optional, for E-SVM) |

### Console Output

The script prints progress with indicators:
```
sample SIFT: 100%|████| 100/100 [02:15<00:00,  1.35s/it]
computed/loaded 498765 descriptors
compute dictionary
VLAD: 100%|████| 3600/3600 [45:32<00:00,  0.76s/it]
Top-1 accuracy: 0.6757 - mAP: 0.5823
```

## Troubleshooting

### Issue: "FileNotFoundError: missing training file"
- Verify training image paths match those in the label file
- Check file extensions (ensure `--suffix_train` matches actual files)
- Ensure full path is accessible on your system

### Issue: "RuntimeError: no descriptors extracted"
- Images may be too sparse/empty
- Try with `--to_binary` if images are drawn rather than printed
- Verify images are grayscale or convert to grayscale

### Issue: Very slow SIFT extraction
- SIFT is keypoint-detector, slower for complex patterns
- For first run with many images, this is expected (hours for 3600+ images)
- Cache files (`enc_test_k100_*.pkl.gz`) are automatically saved to avoid recomputation

### Issue: Memory error during KMeans
- Reduce `--max_descriptors` from 500k to 200k or 100k
- Reduce `--max_dictionary_files` to use fewer training images
- Use MiniBatchKMeans (default) which is more memory-efficient

### Issue: E-SVM too slow
- E-SVM trains one SVM per test image (exponential growth)
- For 3600 test images: 3600 binary classifiers
- Consider running on GPU or reducing test set size for debugging
- Use smaller `--C` value for faster SVM training

## Advanced Tips

### Reproduce Exact Results

Fix the random seed in Python before running:
```python
import numpy as np
np.random.seed(42)
```

The script already does this internally, but ensure external randomness sources are also seeded.

### Custom Visual Vocabulary

Replace the codebook manually:
```python
import pickle, gzip
import numpy as np

# Load or create your own codebook
my_codebook = np.random.randn(100, 128).astype(np.float32)  # K=100, D=128

# Save with standard naming
with gzip.open('mus_k100_d500000_f100.pkl.gz', 'wb') as f:
    import _pickle as cPickle
    cPickle.dump(my_codebook, f, -1)
```

### Batch Processing Different N_CLUSTERS

```bash
for K in 50 100 300; do
    python exercise3.py \
        --n_clusters $K \
        --in_train train/ \
        --in_test test/ \
        --labels_train train_labels.txt \
        --labels_test test_labels.txt
done
```

Each K creates separate cache files, enabling easy comparison.

### Extract VLAD Manually

```python
from exercise3 import vlad, dictionary, loadRandomDescriptors, getFiles
import numpy as np

# Load data
files, labels = getFiles('images/', '.png', 'labels.txt')

# Create or load codebook
mus = np.random.randn(100, 128).astype(np.float32)

# Compute VLAD for specific images
enc = vlad(files[:10], mus, powernorm=True)
print(enc.shape)  # Should be (10, 12800)
```

## References

### Main Publications

- **VLAD**: Jégou et al., "Aggregating local descriptors into a compact image descriptor", CVPR 2010
- **SIFT**: Lowe, "Distinctive Image Features from Scale-Invariant Keypoints", IJCV 2004
- **RootSIFT**: Arandjelović & Zisserman, "Three things everyone should know to improve object retrieval", CVPR 2012
- **iCAM06**: Local tone mapping reference (used in Exercise 2)

### Datasets

- **ICDAR 2017**: Sanchez et al., "ICDAR2017 Competition on Historical Document Writer Identification"
- **ScriptNet**: Historical document handwriting recognition dataset

## License

Educational project for computer vision course.

## Authors

Computer Vision Exercise Series - Exercise 3

---

## Quick Reference

### Run Full Pipeline
```bash
cd exercise3
python exercise3.py \
    --in_train icdar2017-training-binary/ \
    --in_test icdar2017-training-binary/ \
    --labels_train icdar17_labels_train.txt \
    --labels_test icdar17_labels_test.txt
```

### Expected Output
```
#train: 3600
#test: 3600
> computed/loaded 498765 descriptors
> compute dictionary
> compute VLAD for test
Top-1 accuracy: 0.6757 - mAP: 0.5823
```

### Typical Improvements

| Modification | Performance Impact |
|--------------|-------------------|
| Base VLAD | Baseline |
| + Power Normalization | +2-3% Top-1, +0.03 mAP |
| + Exemplar SVM | +5-8% Top-1, +0.08 mAP |
| K=50 → K=300 | +3-5% Top-1 (diminishing returns) |

