# import os
# import rawpy
# import numpy as np
# import cv2
# import imageio
#
# def list_cr3_files(folder):
#     files = [f for f in os.listdir(folder) if f.lower().endswith('.cr3')]
#     # sort by numeric prefix (00.CR3 -> 0)
#     def keyfn(name):
#         base = os.path.splitext(name)[0]
#         try:
#             return int(base)
#         except:
#             return base
#     files.sort(key=keyfn)
#     return [os.path.join(folder, f) for f in files]
#
# def get_bayer_cv_code(raw):
#     # Attempt to determine the 2x2 Bayer pattern and map to OpenCV code.
#     # raw.raw_pattern is usually a 2x2 array with indices into raw.color_desc
#     try:
#         patt = raw.raw_pattern  # 2x2 numpy array of indices
#         desc = raw.color_desc.decode() if isinstance(raw.color_desc, bytes) else raw.color_desc
#         # build char grid
#         chars = np.empty((2,2), dtype=str)
#         for r in range(2):
#             for c in range(2):
#                 idx = int(patt[r, c])
#                 # color_desc could be like "RGB" or "RGBG": pick safe indexing
#                 if idx < len(desc):
#                     chars[r, c] = desc[idx]
#                 else:
#                     chars[r, c] = desc[idx % len(desc)]
#         pattern = ''.join([chars[0,0], chars[0,1], chars[1,0], chars[1,1]])
#         # map to OpenCV codes
#         mapping = {
#             'BGGR': cv2.COLOR_BayerBG2BGR,
#             'GBRG': cv2.COLOR_BayerGB2BGR,
#             'RGGB': cv2.COLOR_BayerRG2BGR,
#             'GRBG': cv2.COLOR_BayerGR2BGR
#         }
#         if pattern in mapping:
#             return mapping[pattern]
#     except Exception:
#         pass
#     # fallback (common for Canon: RGGB)
#     return cv2.COLOR_BayerRG2BGR
#
# def combine_raws_simple(files):
#     # Open first file (assumed longest exposure)
#     raw0 = rawpy.imread(files[0])
#     h_raw = raw0.raw_image_visible.astype(np.float32)  # working Bayer image
#     white_level = float(getattr(raw0, 'white_level', h_raw.max()))
#     # Lower threshold (0.6 instead of 0.8) to more aggressively replace overexposed pixels
#     t = 0.6 * h_raw.max()
#
#     # Iterate through remaining images and replace values in h_raw above t
#     for idx, f in enumerate(files[1:], start=1):
#         r = rawpy.imread(f)
#         i_raw = r.raw_image_visible.astype(np.float32)
#
#         # More aggressive saturation detection (0.95 instead of 0.98) to avoid using nearly-saturated pixels
#         sat_mask = i_raw >= 0.95 * float(getattr(r, 'white_level', i_raw.max()))
#
#         # Scale i to exposure of first image. Since each image has half exposure of previous,
#         # file at index idx has scale 2**idx relative to file 0.
#         scale = 2.0 ** idx
#         i_scaled = i_raw * scale
#
#         # Replace only where h_raw > t AND i_raw is NOT saturated (avoid plateau).
#         replace_mask = (h_raw > t) & (~sat_mask)
#
#         # Update h_raw at replace_mask with scaled values
#         h_raw[replace_mask] = i_scaled[replace_mask]
#
#     return h_raw, raw0  # return also raw0 so we can inspect pattern/white_level
#
# def demosaic_and_wb(h_raw, reference_raw):
#     # Clip to sensible range (0 .. white_level), convert to 16-bit for OpenCV
#     white_level = float(getattr(reference_raw, 'white_level', np.max(h_raw)))
#     h_clipped = np.clip(h_raw, 0, white_level)
#
#     # Scale into full uint16 range proportional to white_level (to keep relative intensities)
#     # map 0..white_level -> 0..65535
#     if white_level <= 0:
#         white_level = h_clipped.max() if h_clipped.max() > 0 else 1.0
#     scaled = (h_clipped / white_level * 65535.0).astype(np.uint16)
#
#     # determine bayer conversion code
#     cv_code = get_bayer_cv_code(reference_raw)
#
#     # Use OpenCV demosaic (returns BGR)
#     bgr = cv2.cvtColor(scaled, cv_code).astype(np.float32)  # shape (H, W, 3), float32
#
#     # Convert to linear float values relative to white_level again
#     bgr = bgr / 65535.0 * white_level
#
#     # Simple white balance via gray-world
#     # compute mean per channel (B, G, R)
#     means = bgr.reshape(-1, 3).mean(axis=0)
#     # avoid division by zero
#     means[means == 0] = 1e-6
#     mean_gray = means.mean()
#     gains = mean_gray / means
#     wb = bgr * gains[np.newaxis, np.newaxis, :]
#
#     # Clip to non-negative
#     wb = np.clip(wb, 0, None)
#     return wb
#
# def log_compress_and_scale(img, out_min=0, out_max=255):
#     # img assumed non-negative float. Use log compression with power law
#     # for better control of bright and dark areas
#
#     # Normalize to 0..1 first
#     mn = img.min()
#     mx = img.max()
#     if mx - mn < 1e-9:
#         norm = np.zeros_like(img)
#     else:
#         norm = (img - mn) / (mx - mn)
#
#     # Apply stronger log compression with gamma correction
#     # log compression: compress bright areas more than dark areas
#     eps = 1e-6
#     img_log = np.log(norm + eps)
#
#     # Apply gamma to further compress highlights (gamma < 1 brightens, > 1 darkens)
#     # We use gamma > 1 to darken the still-bright areas
#     gamma = 1.2
#     img_gamma = np.power(norm, 1.0 / gamma)
#
#     # Combine log and gamma: stronger compression for overexposed areas
#     img_compressed = np.log(img_gamma + eps)
#
#     # Re-normalize to 0..1
#     comp_min = img_compressed.min()
#     comp_max = img_compressed.max()
#     if comp_max - comp_min < 1e-9:
#         norm_final = np.zeros_like(img_compressed)
#     else:
#         norm_final = (img_compressed - comp_min) / (comp_max - comp_min)
#
#     out = (norm_final * (out_max - out_min) + out_min).astype(np.uint8)
#     return out
#
# def main():
#     base = os.path.dirname(__file__)  # directory containing the script
#     data_folder = os.path.join(base, 'exercise_2_data', '06')
#     files = list_cr3_files(data_folder)
#     if not files:
#         print("No CR3 files found in", data_folder)
#         return
#
#     print("Found files (in order):")
#     for f in files:
#         print("  ", os.path.basename(f))
#
#     # Combine raws
#     h_raw, ref_raw = combine_raws_simple(files)
#     # Demosaic & white balance
#     img_wb = demosaic_and_wb(h_raw, ref_raw)
#     # Log compress and scale to 8-bit
#     img_8u = log_compress_and_scale(img_wb)
#     # Convert from BGR (OpenCV) to RGB for saving
#     img_rgb = img_8u[..., ::-1]
#
#     out_path = os.path.join(base, 'hdr_result.png')
#     imageio.imwrite(out_path, img_rgb)
#     print("Saved HDR result to:", out_path)
#
# if __name__ == "__main__":
#     main()

import rawpy
import numpy as np
from PIL import Image
import os

def main():
    data_dir = r'D:\myCode\PyCharm projects\Project_CV\Exercise2_Demsaicing_HDR\exercise_2_data\06'

    files = [os.path.join(data_dir, f'{i:02d}.CR3') for i in range(11)]
    # Step 1: 加载最亮图（曝光最长）
    with rawpy.imread(files[0]) as raw:
        h = raw.raw_image.copy().astype(np.float64)

    # Step 2: 逐步用短曝光图替换过曝区域
    for idx in range(1, len(files)):
        scale = 2 ** idx  # 每张曝光减半，补偿倍数翻倍

        with rawpy.imread(files[idx]) as raw:
            i = raw.raw_image.copy().astype(np.float64)

        i_scaled = i * scale

        # 阈值：当前 h 最大值的 80%
        t = 0.8 * h.max()

        # 替换 h 中高于阈值的像素
        mask = h > t
        h[mask] = i_scaled[mask]

    # Step 3: Demosaicing（在合并后的 HDR RAW 上做）
    # 用 rawpy 借助原始文件的元数据来做 demosaicing
    with rawpy.imread(files[0]) as raw:
        # 把 HDR 数据写回 raw_image（归一化到原始量程）
        raw_max = raw.white_level
        h_clipped = np.clip(h, 0, raw_max)
        raw.raw_image[:] = h_clipped.astype(raw.raw_image.dtype)

        # Demosaicing + 白平衡
        rgb = raw.postprocess(
            no_auto_bright=True,
            use_auto_wb=False,
            use_camera_wb=True,
            output_bps=16
        )

    # Step 4: 对数压缩动态范围
    rgb_float = rgb.astype(np.float64)
    # hdr_log = np.log1p(rgb_float)
    # 改为以10为底（压缩更强，整体更暗）
    hdr_log = np.log10(rgb_float + 1)

    # Step 5: 归一化到 [0, 255]
    lower = np.percentile(hdr_log, 1)
    upper = np.percentile(hdr_log, 95)
    hdr_01 = np.clip((hdr_log - lower) / (upper - lower), 0, 1)
    # hdr_norm = (hdr_log - hdr_log.min()) / (hdr_log.max() - hdr_log.min()) * 255

    gamma = 1.8
    hdr_norm = (hdr_01 ** gamma * 255).astype(np.uint8)
    result = hdr_norm.astype(np.uint8)

    # Step 6: 保存
    Image.fromarray(result).save('hdr_output.png')
    print("HDR image saved.")

if __name__ == "__main__":
    main()