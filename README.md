# Box Detection and Size Estimation (Classical Vision + Point Cloud)

This project processes depth camera data stored in `.mat` files (Amplitude images, Distance images, and Point clouds) to detect and estimate the sizes of rectangular boxes.

Key features:
- Floor plane detection (RANSAC)
- Box-top plane detection (RANSAC)
- Box height, length and width estimation
- 2D/3D visualization of results
- Streamlit-based UI for interactive parameter tuning

---

## 1. Project structure

- `main.py`: Offline processing pipeline
- `streamlit_app.py`: Optional Streamlit UI for interactive parameter tuning and visualization
- `data_utils.py`: Data loading and valid-point extraction
- `ransac.py`: Plane fitting and RANSAC utilities
- `mask_utils.py`: Mask construction and cleanup helpers
- `geometry_utils.py`: Size estimation and corner detection functions
- `visualization.py`: 2D/3D visualization routines
- `DataRead.py`: Utilities to inspect `.mat` files in batch
- `data/`: Directory for input `.mat` files

---

## 2. Expected data format

Each `.mat` file should include fields matching the following prefixes (the code extracts matching fields automatically):

- `amplitudes*`: Intensity (amplitude) image (2D)
- `distances*`: Distance/depth map (2D)
- `cloud*`: Point cloud array with shape `H x W x 3`

The package validates that the extracted arrays conform to expected shapes.

---

## 3. Environment and installation

Recommended Python: 3.10 or newer.

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 4. How to run

### 4.1 Run the offline script

```bash
python main.py
```

By default this script loads `data/example1kinect.mat`.

### 4.2 Run the Streamlit interactive UI

```bash
streamlit run streamlit_app.py
```

From the web UI you can:
- Choose a `.mat` file from the local `data/` folder or upload a file
- Adjust RANSAC and mask-cleaning parameters
- Inspect size estimation results and visualizations

---

## 5. Adjustable parameters

The Streamlit UI exposes several parameters that control detection and post-processing. Typical parameters include:

- `Floor threshold`: Distance threshold for inliers when fitting the ground plane
- `Box top threshold`: Distance threshold for inliers when fitting the box top plane
- `Max iterations`: Maximum RANSAC iterations
- `Min floor component size`: Minimum connected-component area for the floor mask
- `Max internal hole size`: Maximum hole size allowed when filling internal holes in the mask
- `Length/width estimation method`: Method for length/width estimation (`pca` or `simple`)

Adjusting these values helps adapt the pipeline to different data quality and box shapes.

---

## 6. Output

The pipeline produces:

- Box height (distance between ground plane and box-top plane)
- Box length and width (estimated from the top point cloud, PCA by default)
- Approximate corner points of the box top
- 2D segmentation overlays and oriented labels (`top`, `bottom`, `left`, `right`)

---

---

## Advantages and limitations

This section summarizes the main pros and cons of the methods used in this project so you can better understand when the pipeline performs well and where it may fail.


- RANSAC plane fitting (ground and box-top detection)
  - Advantages: robust to a substantial fraction of outliers; easy to implement and tune for planar surfaces.
  - Limitations: detection quality depends on thresholds and iteration count; can be slow if many iterations are needed; may fail for very small or highly fragmented planes.

- PCA-based length/width estimation (top cloud)
  - Advantages: simple, fast, and provides a principal orientation for the box top; works well when the top is mostly planar and dense.
  - Limitations: sensitive to outliers and missing regions; assumes the top surface approximates a planar, rectangular shape.

- Mask construction and morphological cleanup
  - Advantages: removes small noisy components and small holes, producing cleaner segments for downstream processing.
  - Limitations: aggressive parameters may remove thin legitimate parts or overfill true holes; requires parameter tuning per dataset.

## 7. Troubleshooting and common issues

- "No valid points": The point cloud contains too few valid points (e.g. most Z values are zero). Check the input `.mat` file.
- Unstable box-top detection: Increase `Max iterations` and fine-tune the two RANSAC thresholds.
- Fragmented floor mask: Increase `Min floor component size` to remove small components.
- Incorrect hole filling: Reduce `Max internal hole size` to avoid filling real holes in the object.

If you need reproducible environments, consider pinning package versions in `requirements.txt` (e.g. `opencv-python==4.7.0.72`).
