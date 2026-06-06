# pipeline.py

import os
import numpy as np
import matplotlib.pyplot as plt
import rawpy
import glob

from scipy.ndimage import gaussian_filter

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from config import PATTERN_CR3, EX2_RAW, EX5_DIR, EX6_DIR
from utils import (
    load_raw_cr3,
    save_float_image_as_jpg,
    percentile_normalize,
    gamma_correction,
    log_curve,
    apply_brightness_contrast,
)
from bayer import demosaic_simple


# ============================================================
# Exercise 2
# Objective: Read a single CR3 RAW file, perform basic demosaicing,
# and export visualization preview.
# 目标：读取单张 CR3 RAW，做最基础的 demosaicing，并导出可视化预览。
# ============================================================

def exercise2_demosaic(
    raw_path=EX2_RAW,
    output_path="exercise2_demosaicing_preview.jpg",
    pattern=PATTERN_CR3
):
    # Load raw Bayer data from the camera's RAW.
    # 读取相机 RAW 中的二维 Bayer 数据。
    raw = load_raw_cr3(raw_path)

    print("Raw shape:", raw.shape)
    print("Raw min/max:", raw.min(), raw.max())

    # Interpolate single-channel RAW into RGB according to the specified Bayer pattern.
    # 按指定 Bayer 排列把单通道 RAW 插值成 RGB。
    rgb = demosaic_simple(raw, pattern=pattern)

    print("RGB shape:", rgb.shape)
    print("RGB min/max:", rgb.min(), rgb.max())

    # RAW value range is typically much larger than [0, 1],
    # normalize using percentiles before saving.
    # RAW 数值范围通常远大于 [0, 1]，保存前先做百分位归一化。
    rgb_norm, a, b = percentile_normalize(rgb, 0.01, 99.99)
    save_float_image_as_jpg(rgb_norm, output_path, quality=98)

    print("Saved:", output_path)
    return rgb


# ============================================================
# Exercise 3
# Objective: Compare the impact of different brightness mapping curves
# (gamma correction, log curve, etc.) on demosaicing results.
# 目标：比较不同亮度映射曲线对 demosaicing 结果的影响。
# ============================================================

def exercise3_luminosity(
    raw_path=EX2_RAW,
    pattern=PATTERN_CR3
):
    raw = load_raw_cr3(raw_path)
    rgb = demosaic_simple(raw, pattern=pattern)

    # First map RAW RGB to display-friendly [0, 1].
    # 先把 RAW RGB 映射到显示友好的 [0, 1]。
    rgb_norm, a, b = percentile_normalize(rgb, low=0.01, high=99.99)

    # Lower gamma brightens shadows more noticeably.
    # gamma 越小，暗部提亮越明显。
    rgb_gamma_03 = gamma_correction(rgb_norm, gamma=0.3)
    rgb_gamma_04 = gamma_correction(rgb_norm, gamma=0.4)
    rgb_gamma_05 = gamma_correction(rgb_norm, gamma=0.5)

    # Log curve is another way to compress dynamic range.
    # log 曲线是另一种压缩动态范围的方式。
    rgb_log = log_curve(rgb_norm, alpha=10.0)

    # Save each version separately for visual comparison of different curves.
    # 分别保存，方便肉眼比较不同曲线的观感。
    save_float_image_as_jpg(rgb_norm, "exercise3_normalized.jpg", quality=98)
    save_float_image_as_jpg(rgb_gamma_03, "exercise3_gamma_03.jpg", quality=98)
    save_float_image_as_jpg(rgb_gamma_04, "exercise3_gamma_04.jpg", quality=98)
    save_float_image_as_jpg(rgb_gamma_05, "exercise3_gamma_05.jpg", quality=98)
    save_float_image_as_jpg(rgb_log, "exercise3_log_curve.jpg", quality=98)

    print("Saved Exercise 3 images.")
    return rgb_norm, rgb_gamma_03, rgb_log


# ============================================================
# Exercise 4
# Objective: Apply simple white balance using Gray World assumption.
# 目标：使用 Gray World 假设做简单白平衡。
# ============================================================

def gray_world_white_balance(rgb, verbose=True):
    """
    Gray World White Balance.

    Assumes the average color across an image should be close to gray,
    so it scales R/G/B channels to balance their means.

    Gray World 白平衡。

    假设整张图平均颜色应该接近灰色，因此把 R/G/B 三个通道均值拉到一致。
    """
    rgb = rgb.astype(np.float32).copy()

    # Compute the mean intensity and per-channel means.
    # 统计全图平均亮度以及三个颜色通道的平均值。
    mean_image = rgb.mean()
    mean_r = rgb[:, :, 0].mean()
    mean_g = rgb[:, :, 1].mean()
    mean_b = rgb[:, :, 2].mean()

    eps = 1e-8

    scale_r = mean_image / max(mean_r, eps)
    scale_g = mean_image / max(mean_g, eps)
    scale_b = mean_image / max(mean_b, eps)

    if verbose:
        print("\nGray World white balance:")
        print(f"mean image = {mean_image:.4f}")
        print(f"before: R={mean_r:.4f}, G={mean_g:.4f}, B={mean_b:.4f}")
        print(f"scales: R={scale_r:.4f}, G={scale_g:.4f}, B={scale_b:.4f}")

    # Scale each channel by the ratio of image mean to channel mean.
    # 分通道乘以缩放系数，实现简单白平衡。
    rgb[:, :, 0] *= scale_r
    rgb[:, :, 1] *= scale_g
    rgb[:, :, 2] *= scale_b

    if verbose:
        print(
            f"after: R={rgb[:, :, 0].mean():.4f}, "
            f"G={rgb[:, :, 1].mean():.4f}, "
            f"B={rgb[:, :, 2].mean():.4f}"
        )

    return rgb


def exercise4_white_balance(
    raw_path=EX2_RAW,
    pattern=PATTERN_CR3
):
    raw = load_raw_cr3(raw_path)
    rgb = demosaic_simple(raw, pattern=pattern)

    # Apply white balance first, then display mapping.
    # 先白平衡，再做显示映射。
    rgb_wb = gray_world_white_balance(rgb)

    rgb_wb_norm, a, b = percentile_normalize(rgb_wb, low=0.01, high=99.99)

    # Save multiple brightness versions to observe differences after white balance.
    # 保存多种亮度版本，观察白平衡后的结果差异。
    rgb_wb_gamma_03 = gamma_correction(rgb_wb_norm, gamma=0.3)
    rgb_wb_gamma_04 = gamma_correction(rgb_wb_norm, gamma=0.4)
    rgb_wb_gamma_05 = gamma_correction(rgb_wb_norm, gamma=0.5)
    rgb_wb_log = log_curve(rgb_wb_norm, alpha=10.0)

    save_float_image_as_jpg(rgb_wb_norm, "exercise4_white_balance_normalized.jpg", quality=98)
    save_float_image_as_jpg(rgb_wb_gamma_03, "exercise4_white_balance_gamma_03.jpg", quality=98)
    save_float_image_as_jpg(rgb_wb_gamma_04, "exercise4_white_balance_gamma_04.jpg", quality=98)
    save_float_image_as_jpg(rgb_wb_gamma_05, "exercise4_white_balance_gamma_05.jpg", quality=98)
    save_float_image_as_jpg(rgb_wb_log, "exercise4_white_balance_log_curve.jpg", quality=98)

    print("Saved Exercise 4 images.")
    return rgb_wb_gamma_04


# ============================================================
# Exercise 5
# Objective: Using different exposures in RAW to verify sensor data is linear
# 目标：用不同曝光时间的 RAW 验证传感器线性响应。
# ============================================================

def ex5_show_linear():
    base = os.path.dirname(__file__)
    data_folder = os.path.join(base, 'exercise_2_data', '05')
    images = [
        ('IMG_3044.CR3', 1/10),    # 1/10 seconds
        ('IMG_3045.CR3', 1/20),    # 1/20 seconds
        ('IMG_3046.CR3', 1/40),    # 1/40 seconds
        ('IMG_3047.CR3', 1/80),    # 1/80 seconds
        ('IMG_3048.CR3', 1/160),   # 1/160 seconds
        ('IMG_3049.CR3', 1/320),   # 1/320 seconds
    ]

    exposure_times = []
    average_values = []

    print("Processing images...")
    for filename, exp_time in images:
        filepath = os.path.join(data_folder, filename)
        if not os.path.exists(filepath):
            print(f"Warning: {filepath} not found, skipping...")
            continue

        # Load the raw data
        raw = rawpy.imread(filepath)
        raw_data = raw.raw_image_visible.astype(np.float32)

        # Compute mean of the entire raw data (not channel-wise)
        mean_val = raw_data.mean()

        exposure_times.append(exp_time)
        average_values.append(mean_val)

        print(f"  {filename}: exposure={exp_time:.6f}s, avg_raw={mean_val:.2f}")

    # Plot
    exposure_times = np.array(exposure_times)
    average_values = np.array(average_values)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(exposure_times, average_values, s=100, color='blue', label='Measured data')

    # Fit a line to verify linearity
    slope_intercept = np.polyfit(exposure_times, average_values, 1) # It finds the equation of the straight line that best fits your data points using the Least Squares Method.
    p = np.poly1d(slope_intercept) # It converts the raw coefficients in z into a convenient polynomial function object.
    fit_line = p(exposure_times)
    ax.plot(exposure_times, fit_line, 'r--', linewidth=2, label=f'Linear fit: y={slope_intercept[0]:.1f}x+{slope_intercept[1]:.1f}')

    # Calculate R^2 to evaluate the goodness of fit
    # R2 = 1 - sum((y - y_pred)**2) / sum((y - y_mean)**2)
    residuals = average_values - fit_line
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((average_values - average_values.mean())**2)
    r_squared = 1 - (ss_res / ss_tot)

    ax.set_xlabel('Exposure Time (seconds)', fontsize=12)
    ax.set_ylabel('Average Raw Data Value', fontsize=12)
    ax.set_title('Sensor Linearity: Average Raw Value vs Exposure Time', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    # Add R^2 to plot
    textstr = f'$R^2 = {r_squared:.4f}$'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save plot
    out_path = os.path.join(base, 'ex5_linearity_plot.png')
    plt.savefig(out_path, dpi=150)
    print(f"\nPlot saved to: {out_path}")

    # Display plot
    plt.show()

    print("\nLinearity Analysis:")
    print(f"  Slope: {slope_intercept[0]:.2f} (raw value per second)")
    print(f"  Intercept: {slope_intercept[1]:.2f}")
    print(f"  R² (goodness of fit): {r_squared:.4f}")
    print(f"  Linear relationship confirmed: {r_squared > 0.99}")

# ============================================================
# Exercise 6
# Objective: Merge a series of  photos into HDR RAW according to the lecture method.
# 目标：按课堂方法把一组包围曝光 RAW 合成为 HDR RAW。
# ============================================================
def ex6_create_hdr():
    # Configuration and Loading
    data_dir = "exercise_2_data/06"
    file_pattern = os.path.join(data_dir, "*.CR3")
    files = sorted(glob.glob(file_pattern))

    if not files:
        raise FileNotFoundError(f"No .CR3 files found in '{data_dir}'")
    # Lecture method:
    #1. Load the brightest raw data (longest exposure) 00.CR3
    with rawpy.imread(files[0]) as raw_0:
        # Extract raw data as float32 to prevent overflow during HDR multiplication
        # 2. Call it h
        # 提取 raw 数据为 float32 以防止 HDR 乘法时溢出；记为 h
        h = raw_0.raw_image.astype(np.float32).copy()

        # Keep camera white balance for post-demosaicing [Red, Green, Blue, Green2]
        wb = raw_0.camera_whitebalance

    # 3. Set the threshold t (A value of 0.8 * max(h) should be fine)
    t = 0.8 * np.max(h)

    # We track the unscaled sensor values to evaluate the threshold in future iterations.
    # This prevents evaluating the threshold against already-multiplied HDR values
    # and safely avoids the 'plateau' (saturation) in subsequent shorter exposures.
    # 跟踪未缩放的传感器值以评估未来迭代中的阈值。
    # 这可以防止根据已乘以的 HDR 值评估阈值，并安全地避免后续较短曝光中的"高原"（饱和）。
    current_sensor_values = h.copy()

    # 4. Loop through each next raw file:
    for idx, file_path in enumerate(files[1:], start=1):
        with rawpy.imread(file_path) as raw_i:
            # Load it and name it i
            i = raw_i.raw_image.astype(np.float32)

            # Multiply i by the exposure difference to the first photo
            # Assuming exposures halve each time: 0th=1x, 1st=2x, 2nd=4x, 3rd=8x...(按与第一张照片的曝光时间差乘以 i. )
            multiplier = 2 ** idx
            i_scaled = i * multiplier

            # Values in h which are above threshold t get replaced by corresponding values in i
            # h 中超过阈值 t 的值被替换为 i 中对应位置的值
            mask = current_sensor_values > t
            h[mask] = i_scaled[mask]

            # Update the tracked sensor values. If 'i' is also on the plateau (saturated),
            # this ensures it will trigger the mask > t in the NEXT loop and get replaced again.
            # 更新跟踪的传感器值。如果 'i' 也在高原（饱和），
            # 这确保它将在下一个循环中触发 mask > t 并再次被替换。
            current_sensor_values[mask] = i[mask]

    # --- Post-Processing ---
    print("HDR Raw merge complete. Processing image...")

    # Map float32 data linearly to 16-bit integer range to meet OpenCV requirements.
    # 将 float32 数据线性映射到 16-bit 整数范围内，以满足 OpenCV 的要求
    h_max = np.max(h)

    h_16bit = np.clip((h / h_max) * 65535.0, 0, 65535).astype(np.uint16)

    # 5. Apply the demosaicing algorithm, Bayer → BGR
    # 应用 demosaicing 算法
    hdr_bgr_16bit = cv2.cvtColor(h_16bit, cv2.COLOR_BayerBG2BGR)

    # After demosaicing, convert back to float32 to continue with white balance and log operation.
    # 去马赛克完成后，转换回 float32 继续进行白平衡和对数运算
    hdr_bgr = hdr_bgr_16bit.astype(np.float32)

    # 6. Apply the white balance
    # wb array contains [R_scale, G_scale, B_scale, G2_scale]
    b, g, r = cv2.split(hdr_bgr)
    b = b * wb[2]  # Blue channel multiplier
    g = g * wb[1]  # Green channel multiplier
    r = r * wb[0]  # Red channel multiplier
    hdr_wb = cv2.merge((b, g, r))

    # 7. Decrease the dynamic range by computing the logarithm of this data
    # 通过计算数据的对数来减少动态范围
    # Add 1e-6 to avoid taking log of 0, which results in -inf
    # 加上 1e-6 避免对 0 取对数导致 -inf
    hdr_log = np.log(hdr_wb + 1e-6)

    # 8. Apply a log scale to these values, normalize the result in [0, 255]
    log_min = np.min(hdr_log)
    log_max = np.max(hdr_log)
    hdr_normalized = (hdr_log - log_min) / (log_max - log_min) * 255.0

    # 9. Save the resulting image
    hdr_final = hdr_normalized.astype(np.uint8)
    output_path = "ex6_hdr_result.png"
    cv2.imwrite(output_path, hdr_final)
    print(f"HDR image saved successfully as {output_path}")
    # Return the white balance adjusted HDR image before log compression
    # (for use in other exercises like iCAM06)
    return hdr_wb


def load_hdr_raw_paths(folder=EX6_DIR):
    """Load HDR bracketed exposure file paths in order from 00.CR3 to 10.CR3.
    按 00.CR3 到 10.CR3 的顺序加载 HDR 包围曝光文件路径。
    """
    paths = []

    for k in range(11):
        path = os.path.join(folder, f"{k:02d}.CR3")
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        paths.append(path)

    return paths


def merge_hdr_raw_lecture_method(folder=EX6_DIR, threshold_ratio=0.8):
    paths = load_hdr_raw_paths(folder)

    # Start with the first image as the HDR estimate h.
    # 从第一张图开始作为 HDR 估计值 h。
    h = load_raw_cr3(paths[0]).astype(np.float32)

    # Pixels above this threshold are considered near saturation and need replacement
    # with shorter exposures.
    # 超过该阈值的像素认为接近饱和，需要用更短曝光替换。
    threshold = threshold_ratio * h.max()

    print("Initial h:", paths[0])
    print("h min/max:", h.min(), h.max())
    print("threshold:", threshold)

    for k in range(1, len(paths)):
        i = load_raw_cr3(paths[k]).astype(np.float32)

        # Assume each subsequent exposure is half: 0th=1x, 1st=2x, 2nd=4x, etc.
        # So shorter exposures need multiplying by 2^k to align to the first exposure.
        # 假设每后一张曝光减半，所以短曝光需要乘以 2^k 对齐到第一张曝光。
        scale_factor = 2 ** k
        i_scaled = i * scale_factor

        # Replace overly bright pixels in current h with values from shorter exposure.
        # 当前 h 中过亮的像素，用短曝光中对应位置的值替换。
        replace_mask = h > threshold
        num_replaced = int(replace_mask.sum())

        h[replace_mask] = i_scaled[replace_mask]

        print(
            f"{os.path.basename(paths[k])}: "
            f"scale={scale_factor}, replaced={num_replaced}"
        )

    print("HDR raw min/max:", h.min(), h.max())
    return h


def log_tone_mapping(rgb):
    """Apply simple log tone mapping to HDR RGB to compress dynamic range for JPG display.
    对 HDR RGB 做简单 log tone mapping，压缩动态范围用于 JPG 显示。
    """
    rgb = np.maximum(rgb, 0.0)
    log_rgb = np.log(rgb + 1.0)
    log_rgb_norm, a, b = percentile_normalize(log_rgb, low=0.01, high=99.99)
    return log_rgb_norm


def exercise6_initial_hdr(
    folder=EX6_DIR,
    output_path="exercise6_hdr_log_result.jpg",
    pattern=PATTERN_CR3
):
    # After HDR merging, the result is still Bayer RAW, needs demosaicing first,
    # then white balance and tone mapping.
    # HDR 合成后仍是 Bayer RAW，需要先 demosaic，再白平衡和 tone mapping。
    hdr_raw = merge_hdr_raw_lecture_method(folder=folder, threshold_ratio=0.8)

    hdr_rgb = demosaic_simple(hdr_raw, pattern=pattern)
    hdr_rgb_wb = gray_world_white_balance(hdr_rgb)

    hdr_log = log_tone_mapping(hdr_rgb_wb)

    save_float_image_as_jpg(hdr_log, output_path, quality=98)
    print("Saved:", output_path)

    return hdr_raw, hdr_rgb_wb, hdr_log


# ============================================================
# Exercise 7
# Objective: Implement a simplified iCAM06-based local tone mapping.
# 目标：实现一个简化版 iCAM06 局部 tone mapping。
# ============================================================

def bilateral_filter_image(img, diameter=9, sigma_color=0.4, sigma_space=9):
    """Prefer OpenCV bilateral filtering; fall back to gaussian filtering if cv2 unavailable.
    优先使用 OpenCV 双边滤波；没有 cv2 时退化为高斯滤波。
    """
    img = img.astype(np.float32)

    if HAS_CV2:
        return cv2.bilateralFilter(
            img,
            d=diameter,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space
        )

    print("Warning: cv2 not installed, using gaussian_filter instead.")
    return gaussian_filter(img, sigma=2.0)


def icam06_tone_mapping(
    rgb,
    output_range=4.0,
    bilateral_diameter=9,
    sigma_color=0.4,
    sigma_space=9
):
    """Simplified iCAM06: separate luminance into base and detail layers,
    compress the base layer, then restore color ratios.

    简化 iCAM06：分离亮度基础层/细节层，压缩基础层后恢复颜色比例。
    """
    eps = 1e-8

    rgb = rgb.astype(np.float32)
    rgb = np.maximum(rgb, 0.0)

    # Estimate input intensity using a green-weighted formula.
    # 用偏重绿色的亮度公式估计输入 intensity。
    red = rgb[:, :, 0]
    green = rgb[:, :, 1]
    blue = rgb[:, :, 2]

    input_intensity = (20.0 * red + 40.0 * green + blue) / 61.0
    input_intensity = np.maximum(input_intensity, eps)

    r_ratio = red / input_intensity
    g_ratio = green / input_intensity
    b_ratio = blue / input_intensity

    # In log domain, separate base layer and detail layer.
    # Better suited for handling HDR dynamic range.
    # 在 log 域中分离基础层和细节层，更适合处理 HDR 动态范围。
    log_intensity = np.log(input_intensity)

    log_base = bilateral_filter_image(
        log_intensity,
        diameter=bilateral_diameter,
        sigma_color=sigma_color,
        sigma_space=sigma_space
    )

    log_details = log_intensity - log_base

    # Compress the base layer to a specified output dynamic range.
    # 将基础层压缩到指定输出动态范围。
    base_min = log_base.min()
    base_max = log_base.max()

    compression = np.log(output_range) / max(base_max - base_min, eps)
    log_offset = -base_max * compression

    output_intensity = np.exp(
        log_base * compression + log_offset + log_details
    )

    # Maintain original color ratios; only replace the compressed luminance.
    # 保持原始颜色比例，只替换压缩后的亮度。
    out_r = r_ratio * output_intensity
    out_g = g_ratio * output_intensity
    out_b = b_ratio * output_intensity

    out_rgb = np.stack([out_r, out_g, out_b], axis=2)

    out_rgb_norm, a, b = percentile_normalize(out_rgb, low=0.01, high=99.99)
    return out_rgb_norm


def exercise7_icam06(
    folder=EX6_DIR,
    output_path="exercise7_icam06_result.jpg",
    pattern=PATTERN_CR3
):
    # Exercise 7 reuses the HDR RAW merge result from Exercise 6.
    # Exercise 7 复用 Exercise 6 的 HDR RAW 合成结果。
    hdr_raw = merge_hdr_raw_lecture_method(folder=folder, threshold_ratio=0.8)

    hdr_rgb = demosaic_simple(hdr_raw, pattern=pattern)
    hdr_rgb_wb = gray_world_white_balance(hdr_rgb)

    icam_img = icam06_tone_mapping(
        hdr_rgb_wb,
        output_range=4.0,
        bilateral_diameter=9,
        sigma_color=0.4,
        sigma_space=9
    )

    save_float_image_as_jpg(icam_img, output_path, quality=98)
    print("Saved:", output_path)

    return icam_img

def ex7_apply_icam06(hdr_bgr_float, output_range=4.0, d=5, sigma_color=0.5, sigma_space=2.0):
    """
    Implement iCAM06 HDR tone mapping algorithm.
    :param hdr_bgr_float: Input HDR image (float32, uncompressed linear RGB)
                          输入的 HDR 图像 (float32, 尚未压缩动态范围的线性 RGB)
    :param output_range: Compression ratio for dynamic range (tunable)
                         动态范围的压缩倍数（可调整）
    :param d: Bilateral filter kernel size; set small (e.g. 3 or 5) during dev for speed
              双边滤波的核大小，开发时设小一点(如 3 或 5)以加快速度
    :param sigma_color: Sigma for color space of bilateral filter (tunable)
                        色彩空间滤波器的 sigma 值（可调整）
    :param sigma_space: Sigma for coordinate space of bilateral filter (tunable)
                        坐标空间滤波器的 sigma 值（可调整）
    """
    print(f"Running iCAM06 Tone Mapping... (d={d}, output_range={output_range})")

    # Extract BGR channels (OpenCV default format)
    b, g, r = cv2.split(hdr_bgr_float)

    # Define a very small epsilon to prevent division by zero or log(0) errors.
    eps = 1e-6

    # Following the Pseudocode in the Slides:
    # 1. input_intensity = 1/61 * (20*red + 40*green + blue)
    input_intensity = (20.0 * r + 40.0 * g + 1.0 * b) / 61.0
    input_intensity = np.clip(input_intensity, eps, None)

    # 2. r, g, b = rgb / input_intensity (extract color coefficients / Chromaticity)
    r_chroma = r / input_intensity
    g_chroma = g / input_intensity
    b_chroma = b / input_intensity

    # 3. log_base = bilat_filt(log(input_intensity))
    log_intensity = np.log(input_intensity)
    # OpenCV bilateral filter requires float32 data type
    # OpenCV 的双边滤波需要 float32 数据类型
    log_intensity_f32 = log_intensity.astype(np.float32)
    # Use bilateral filter to extract base layer (Base Layer)
    # 使用双边滤波器提取基础层
    log_base = cv2.bilateralFilter(log_intensity_f32, d, sigma_color, sigma_space)

    # 4. log_details = log(input_intensity) - log_base (extract detail layer)(提取细节层)
    log_details = log_intensity - log_base

    # 5. compression = log(output_range) / (max(log_base) - min(log_base))
    base_max = np.max(log_base)
    base_min = np.min(log_base)

    range_diff = base_max - base_min
    if range_diff < eps:  # Prevent division by zero for solid color images
                          # 防止纯色图片导致除以 0
        range_diff = eps

    compression = np.log(output_range) / range_diff

    # 6. log_offset = -max(log_base) * compression
    # This step aligns the maximum of the compressed base layer to 0, since exp(0) = 1
    # 这一步是为了将压缩后的基础层最大值对齐到 0，因为 exp(0) = 1
    log_offset = -base_max * compression

    # 7. output_intensity = exp(log_base * compression + log_offset + log_detail)
    # Recombine: compressed base layer + offset + preserved detail layer, then exponentiate
    # 重新组合：压缩基础层 + 偏移量 + 保留的细节层，最后用指数还原
    output_intensity = np.exp(log_base * compression + log_offset + log_details)

    # 8. rgb = r*output_intensity, g*output_intensity, b*output_intensity
    r_out = r_chroma * output_intensity
    g_out = g_chroma * output_intensity
    b_out = b_chroma * output_intensity

    # Merge channels
    # 合并通道
    output_bgr = cv2.merge((b_out, g_out, r_out))

    # Due to the effect of log_offset, the brightest areas are roughly 1.0, Clip to [0.0, 1.0] and convert to uint8 [0, 255] for saving.
    # 由于 log_offset 的作用，最亮的地方大约为 1.0, 将其限制在 [0.0, 1.0] 内，并转换为 [0, 255] 的 uint8 图像进行保存
    output_bgr = np.clip(output_bgr, 0.0, 1.0)
    final_image = (output_bgr * 255.0).astype(np.uint8)

    return final_image

# ============================================================
# Exercise 8
# Objective: Encapsulate the complete RAW -> JPG processing pipeline.
# 目标：封装最终 RAW -> JPG 的完整处理函数。
# ============================================================

def enhance_for_process_raw(
    rgb_norm,
    brightness=0.10,
    contrast=1.75,
    saturation=1.25,
):
    """
    Final enhancement step for Exercise 8.

    Simulates manual image adjustment: increase brightness, enhance contrast,
    and slightly boost saturation.

    Exercise 8 的最终增强步骤。

    这里模拟手动调图观察：提高亮度、增强对比度、略微提高饱和度。
    """
    img = np.clip(rgb_norm, 0.0, 1.0).astype(np.float32)

    # Brighten the overall image.
    # 提亮整体画面。
    img = img + brightness
    img = np.clip(img, 0.0, 1.0)

    # Enhance contrast centered at 0.5.
    # 以 0.5 为中心增强对比度。
    img = (img - 0.5) * contrast + 0.5
    img = np.clip(img, 0.0, 1.0)

    # Boost saturation by interpolating between gray and original image.
    # 通过灰度图和原图插值增强饱和度。
    gray = img.mean(axis=2, keepdims=True)
    img = gray + saturation * (img - gray)
    img = np.clip(img, 0.0, 1.0)

    return img


def process_raw(
    raw_path,
    jpg_path,
    pattern=PATTERN_CR3,
    black_percentile=0.01,
    norm_low=0.01,
    norm_high=99.99,
    gamma=0.38,
    enhance_brightness=0.10,
    enhance_contrast=1.75,
    enhance_saturation=1.25,
    final_brightness=0.0,
    final_contrast=1.0,
    jpg_quality=99,
    verbose=True,
):
    """
    Exercise 8 final function.

    Processing pipeline:
        CR3 RAW
        -> Black level subtraction (黑电平校正)
        -> Simple demosaicing
        -> Gray World white balance
        -> Percentile normalization (百分位归一化)
        -> Gamma correction (gamma 校正)
        -> Brightness/contrast/saturation enhancement(亮度/对比度/饱和度增强)
        -> Optional: final display brightness/contrast(最终显示亮度/对比度)
        -> Save JPG
    """
    raw = load_raw_cr3(raw_path)
    rgb_final = process_bayer_raw_to_rgb(
        raw,
        pattern=pattern,
        black_percentile=black_percentile,
        norm_low=norm_low,
        norm_high=norm_high,
        gamma=gamma,
        enhance_brightness=enhance_brightness,
        enhance_contrast=enhance_contrast,
        enhance_saturation=enhance_saturation,
        final_brightness=final_brightness,
        final_contrast=final_contrast,
        verbose=verbose,
    )

    save_float_image_as_jpg(rgb_final, jpg_path, quality=jpg_quality)

    if verbose:
        print("Saved processed JPG:", jpg_path)
    return rgb_final


def process_raw_to_array(
    raw_path,
    pattern=PATTERN_CR3,
    black_percentile=0.01,
    norm_low=0.01,
    norm_high=99.99,
    gamma=0.38,
    enhance_brightness=0.10,
    enhance_contrast=1.75,
    enhance_saturation=1.25,
    final_brightness=0.0,
    final_contrast=1.0,
    verbose=False,
):
    """
    Same pipeline as process_raw, but does not write to disk;
    for use in interactive interfaces (e.g., Streamlit).
    """
    raw = load_raw_cr3(raw_path)
    return process_bayer_raw_to_rgb(
        raw,
        pattern=pattern,
        black_percentile=black_percentile,
        norm_low=norm_low,
        norm_high=norm_high,
        gamma=gamma,
        enhance_brightness=enhance_brightness,
        enhance_contrast=enhance_contrast,
        enhance_saturation=enhance_saturation,
        final_brightness=final_brightness,
        final_contrast=final_contrast,
        verbose=verbose,
    )


def process_bayer_raw_to_rgb(
    raw,
    pattern=PATTERN_CR3,
    black_percentile=0.01,
    norm_low=0.01,
    norm_high=99.99,
    gamma=0.38,
    enhance_brightness=0.10,
    enhance_contrast=1.75,
    enhance_saturation=1.25,
    final_brightness=0.0,
    final_contrast=1.0,
    verbose=False,
):
    """
    Run the complete Exercise 8 pipeline on an already-loaded 2D Bayer RAW (float32),
    returning RGB float32.

    对已加载的二维 Bayer RAW（float32）跑完整 Exercise 8 管线，返回 RGB float32。
    """
    raw = np.array(raw, dtype=np.float32, copy=True)

    black = np.percentile(raw, black_percentile)
    raw = raw - black
    raw = np.clip(raw, 0.0, None)

    rgb = demosaic_simple(raw, pattern=pattern)
    rgb_wb = gray_world_white_balance(rgb, verbose=verbose)

    rgb_wb = np.nan_to_num(rgb_wb, nan=0.0, posinf=0.0, neginf=0.0)
    rgb_wb = np.clip(rgb_wb, 0.0, None)

    rgb_norm, a, b = percentile_normalize(rgb_wb, low=norm_low, high=norm_high)
    rgb_gamma = gamma_correction(rgb_norm, gamma=gamma)
    rgb_final = enhance_for_process_raw(
        rgb_gamma,
        brightness=enhance_brightness,
        contrast=enhance_contrast,
        saturation=enhance_saturation,
    )
    return apply_brightness_contrast(
        rgb_final, brightness=final_brightness, contrast=final_contrast
    )


def test_four_bayer_patterns(
    raw_path=EX2_RAW
):
    """Output four Bayer arrangement results for the same RAW to visually determine correct phase.
    对同一张 RAW 输出四种 Bayer 排列结果，用于肉眼判断正确相位。
    """
    patterns = ["RGGB", "GRBG", "GBRG", "BGGR"]

    for pattern in patterns:
        output_path = f"final_test_pattern_{pattern}.jpg"
        print("\n=================================")
        print("Testing pattern:", pattern)
        print("=================================")
        process_raw(raw_path, output_path, pattern=pattern)
        print("Saved:", output_path)
