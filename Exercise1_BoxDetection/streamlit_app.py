import io
import os

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.io import loadmat

from data_utils import list_mat_files, get_valid_points_with_indices
from geometry_utils import estimate_box_dimensions, get_box_top_corners_pca
from mask_utils import (
    create_mask_from_indices,
    filter_floor_mask,
    get_non_floor_mask,
    get_points_from_mask,
    largest_connected_component,
)
from ransac import ransac_plane
from visualization import get_rectangle_from_mask


def load_from_mat_dict(data):
    """从 loadmat 返回字典中提取 A、D、PC 三类核心数据。"""
    a_key = None
    d_key = None
    pc_key = None

    for key in data.keys():
        if key.startswith("__"):
            continue
        if key.startswith("amplitudes"):
            a_key = key
        elif key.startswith("distances"):
            d_key = key
        elif key.startswith("cloud"):
            pc_key = key

    if a_key is None or d_key is None or pc_key is None:
        raise ValueError(
            "Could not find amplitude/distance/cloud fields in this mat file."
        )

    a = np.squeeze(data[a_key])
    d = np.squeeze(data[d_key])
    pc = data[pc_key]

    if a.shape != d.shape:
        raise ValueError(f"A and D shape mismatch: {a.shape} vs {d.shape}")
    if pc.ndim != 3 or pc.shape[2] != 3:
        raise ValueError(f"PC should be HxWx3, got {pc.shape}")
    if a.shape != pc.shape[:2]:
        raise ValueError(f"A/D and PC shape mismatch: {a.shape} vs {pc.shape[:2]}")

    return a, d, pc


def load_mat_from_file(path_or_buffer):
    """从文件路径或二进制缓冲区读取 .mat，并返回 A/D/PC。"""
    data = loadmat(path_or_buffer)
    return load_from_mat_dict(data)


def percentile_range(image, low=1, high=99):
    """计算图像有效像素的分位数显示范围。"""
    valid = image[np.isfinite(image)]
    if valid.size == 0:
        return None, None
    return np.percentile(valid, low), np.percentile(valid, high)


def draw_gray_image(image, title):
    """用灰度图显示二维图像。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    vmin, vmax = percentile_range(image, 1, 99)
    ax.imshow(image, cmap="gray", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def draw_mask(mask, title):
    """显示布尔掩码图像。"""
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.imshow(mask, cmap="gray")
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


def draw_final_overlay(floor_mask, box_top_mask):
    """绘制最终检测叠加图（带边框与方向文字）。"""
    h, w = floor_mask.shape
    result = np.zeros((h, w, 3), dtype=float)

    floor_color = np.array([0.55, 1.00, 0.40])
    non_floor_color = np.array([0.00, 0.00, 0.45])
    box_top_color = np.array([0.60, 0.00, 0.00])

    result[:, :] = non_floor_color
    result[floor_mask] = floor_color
    result[box_top_mask] = box_top_color

    corners = get_rectangle_from_mask(box_top_mask)
    closed_corners = np.vstack([corners, corners[0]])

    top_point = corners[np.argmin(corners[:, 1])]
    bottom_point = corners[np.argmax(corners[:, 1])]
    left_point = corners[np.argmin(corners[:, 0])]
    right_point = corners[np.argmax(corners[:, 0])]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(result)
    ax.set_title("Final Detection Overlay")

    ax.plot(
        closed_corners[:, 0],
        closed_corners[:, 1],
        color="cyan",
        linewidth=3,
    )
    ax.scatter(
        corners[:, 0],
        corners[:, 1],
        color="cyan",
        s=30,
    )

    ax.text(top_point[0], top_point[1] - 35, "top", color="black", fontsize=12, ha="center")
    ax.text(
        bottom_point[0],
        bottom_point[1] + 35,
        "bottom",
        color="black",
        fontsize=12,
        ha="center",
    )
    ax.text(
        left_point[0] - 45,
        left_point[1],
        "left",
        color="black",
        fontsize=12,
        ha="center",
    )
    ax.text(
        right_point[0] + 45,
        right_point[1],
        "right",
        color="black",
        fontsize=12,
        ha="center",
    )

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")
    st.pyplot(fig)
    plt.close(fig)


st.set_page_config(page_title="Box Detection Tuner", layout="wide")
st.title("Box Detection Parameter Tuner")
st.caption("Load a .mat point-cloud sample and tune key traditional CV parameters.")

with st.sidebar:
    st.header("Data Source")
    source = st.radio("Choose input", ["Use local data file", "Upload .mat file"])

    selected_path = None
    uploaded = None

    if source == "Use local data file":
        if os.path.exists("../data"):
            mat_files = list_mat_files("../data")
        else:
            mat_files = []
        if not mat_files:
            st.warning("No .mat file found in ./data")
        else:
            selected_path = st.selectbox("Select .mat file", mat_files)
    else:
        uploaded = st.file_uploader("Upload .mat", type=["mat"])

    st.header("RANSAC Parameters")
    floor_threshold = st.slider("Floor threshold", 0.001, 0.2, 0.05, 0.001)
    box_threshold = st.slider("Box top threshold", 0.001, 0.05, 0.005, 0.001)
    max_iterations = st.slider("Max iterations", 100, 5000, 1000, 100)

    st.header("Mask Cleaning Parameters")
    min_floor_size = st.slider("Min floor component size", 10, 5000, 100, 10)
    max_hole_size = st.slider("Max internal hole size", 10, 5000, 300, 10)

    st.header("Geometry Parameters")
    size_method = st.selectbox("Length/width estimation method", ["pca", "simple"])

    run = st.button("Run Detection", type="primary")


if run:
    try:
        if source == "Use local data file":
            if not selected_path:
                st.error("Please choose a local .mat file first.")
                st.stop()
            a, d, pc = load_mat_from_file(selected_path)
            st.success(f"Loaded: {selected_path}")
        else:
            if uploaded is None:
                st.error("Please upload a .mat file first.")
                st.stop()
            buffer = io.BytesIO(uploaded.read())
            a, d, pc = load_mat_from_file(buffer)
            st.success(f"Loaded uploaded file: {uploaded.name}")

        col1, col2, col3 = st.columns(3)
        col1.metric("Image Height", int(pc.shape[0]))
        col2.metric("Image Width", int(pc.shape[1]))
        col3.metric("Valid Points", int(np.sum(pc[:, :, 2] != 0)))

        points, rows, cols = get_valid_points_with_indices(pc)
        if points.shape[0] < 3:
            st.error("Not enough valid points to run RANSAC.")
            st.stop()

        floor_model, floor_inliers = ransac_plane(
            points,
            threshold=floor_threshold,
            max_iterations=max_iterations,
        )
        raw_floor_mask = create_mask_from_indices(pc.shape[:2], rows, cols, floor_inliers)
        filtered_floor_mask = filter_floor_mask(
            raw_floor_mask,
            min_floor_size=min_floor_size,
            max_hole_size=max_hole_size,
        )

        non_floor_mask = get_non_floor_mask(filtered_floor_mask)
        non_floor_points, non_floor_rows, non_floor_cols = get_points_from_mask(
            pc, non_floor_mask
        )
        if non_floor_points.shape[0] < 3:
            st.error("Not enough non-floor points to detect box top plane.")
            st.stop()

        box_model, box_inliers = ransac_plane(
            non_floor_points,
            threshold=box_threshold,
            max_iterations=max_iterations,
        )
        raw_box_mask = create_mask_from_indices(
            pc.shape[:2], non_floor_rows, non_floor_cols, box_inliers
        )
        box_top_mask = largest_connected_component(raw_box_mask)

        height, length, width = estimate_box_dimensions(
            pc,
            floor_model,
            box_model,
            box_top_mask,
            method=size_method,
        )
        corners = get_box_top_corners_pca(pc, box_top_mask)

        st.subheader("Estimated Box Dimensions")
        d1, d2, d3 = st.columns(3)
        d1.metric("Height (m)", f"{height:.4f}")
        d2.metric("Length (m)", f"{length:.4f}")
        d3.metric("Width (m)", f"{width:.4f}")

        st.subheader("Estimated Box Top Corners (3D)")
        st.dataframe(corners, use_container_width=True)

        st.subheader("Visualization")
        img_col1, img_col2 = st.columns(2)
        with img_col1:
            draw_gray_image(a, "Amplitude")
        with img_col2:
            d_disp = d.copy().astype(float)
            d_disp[d_disp == 0] = np.nan
            draw_gray_image(d_disp, "Distance")

        m1, m2, m3 = st.columns(3)
        with m1:
            draw_mask(raw_floor_mask, "Raw Floor Mask")
        with m2:
            draw_mask(filtered_floor_mask, "Filtered Floor Mask")
        with m3:
            draw_mask(box_top_mask, "Box Top Mask")

        draw_final_overlay(filtered_floor_mask, box_top_mask)

    except Exception as exc:
        st.exception(exc)
else:
    st.info("Set parameters in the sidebar and click 'Run Detection'.")
