import os
import glob
import numpy as np
import cv2
import rawpy


def create_hdr():
    # 1. Configuration and Loading
    data_dir = "exercise_2_data/06"
    file_pattern = os.path.join(data_dir, "*.CR3")
    files = sorted(glob.glob(file_pattern))

    if not files:
        raise FileNotFoundError(f"No .CR3 files found in '{data_dir}'")

    # Load the brightest raw data (longest exposure) 00.CR3
    with rawpy.imread(files[0]) as raw_0:
        # Extract raw CFA data as float32 to prevent overflow during HDR multiplication
        # 2. Let's call it h
        h = raw_0.raw_image.astype(np.float32).copy()

        # Keep camera white balance for post-demosaicing [Red, Green, Blue, Green2]
        wb = raw_0.camera_whitebalance

    # 3. Set the threshold t
    # A value of 0.8 * max(h) should be fine
    t = 0.8 * np.max(h)

    # We track the unscaled sensor values to evaluate the threshold in future iterations.
    # This prevents evaluating the threshold against already-multiplied HDR values
    # and safely avoids the 'plateau' (saturation) in subsequent shorter exposures.
    current_sensor_values = h.copy()

    # Loop through each next raw file:
    for idx, file_path in enumerate(files[1:], start=1):
        with rawpy.imread(file_path) as raw_i:
            # Load it and name it i
            i = raw_i.raw_image.astype(np.float32)

            # Multiply i by the exposure difference to the first photo
            # Assuming exposures halve each time: 0th=1x, 1st=2x, 2nd=4x, 3rd=8x...
            multiplier = 2 ** idx
            i_scaled = i * multiplier

            # Values in h which are above threshold t get replaced by corresponding values in i
            mask = current_sensor_values > t
            h[mask] = i_scaled[mask]

            # Update the tracked sensor values. If 'i' is also on the plateau (saturated),
            # this ensures it will trigger the mask > t in the NEXT loop and get replaced again.
            current_sensor_values[mask] = i[mask]

    print("HDR Raw merge complete. Processing image...")

    # --- Post-Processing ---
    print("HDR Raw merge complete. Processing image...")

    # --- 修复 OpenCV Demosaicing 报错 ---
    # 将 float32 数据线性映射到 16-bit 整数范围内，以满足 OpenCV 的要求
    h_max = np.max(h)
    # 避免除以 0 的情况
    if h_max > 0:
        h_16bit = np.clip((h / h_max) * 65535.0, 0, 65535).astype(np.uint16)
    else:
        h_16bit = h.astype(np.uint16)

    # 4. Apply the demosaicing algorithm
    # 现在输入的是合法的 16-bit 数据
    hdr_bgr_16bit = cv2.cvtColor(h_16bit, cv2.COLOR_BayerBG2BGR)

    # 去马赛克完成后，转换回 float32 继续进行白平衡和对数运算
    hdr_bgr = hdr_bgr_16bit.astype(np.float32)

    # 5. Apply the white balance
    # wb 数组包含 [R_scale, G_scale, B_scale, G2_scale]
    b, g, r = cv2.split(hdr_bgr)
    b = b * wb[2]  # 蓝色通道乘数
    g = g * wb[1]  # 绿色通道乘数
    r = r * wb[0]  # 红色通道乘数
    hdr_wb = cv2.merge((b, g, r))

    # 6. Decrease the dynamic range by computing the logarithm of this data
    # 加上 1e-6 避免对 0 取对数导致 -inf
    hdr_log = np.log(hdr_wb + 1e-6)

    # 7. Downscale it to the [0, 255] interval
    log_min = np.min(hdr_log)
    log_max = np.max(hdr_log)
    hdr_normalized = (hdr_log - log_min) / (log_max - log_min) * 255.0

    # 8. Save the resulting image
    hdr_final = hdr_normalized.astype(np.uint8)
    output_path = "hdr_result.png"
    cv2.imwrite(output_path, hdr_final)

    print(f"HDR image saved successfully as {output_path}")


if __name__ == "__main__":
    create_hdr()