# bayer.py

import numpy as np
from scipy.ndimage import convolve


def create_bayer_masks(shape, pattern="RGGB"):
    """
    根据 Bayer 排列创建 R/G/B 三个采样 mask。

    pattern 字符串的顺序表示 2x2 Bayer 基元：

        pattern[0] pattern[1]
        pattern[2] pattern[3]

    Example:
        "RGGB" means:

            R G
            G B

        "GBRG" means:

            G B
            R G
    """
    h, w = shape

    # mask 中 1 表示该像素位置真实采样了对应颜色，0 表示需要插值。
    r_mask = np.zeros((h, w), dtype=np.float32)
    g_mask = np.zeros((h, w), dtype=np.float32)
    b_mask = np.zeros((h, w), dtype=np.float32)

    # 四个起始位置分别对应 2x2 Bayer 单元中的四个像素。
    positions = [
        (0, 0, pattern[0]),
        (0, 1, pattern[1]),
        (1, 0, pattern[2]),
        (1, 1, pattern[3]),
    ]

    for row_start, col_start, color in positions:
        if color == "R":
            r_mask[row_start::2, col_start::2] = 1.0
        elif color == "G":
            g_mask[row_start::2, col_start::2] = 1.0
        elif color == "B":
            b_mask[row_start::2, col_start::2] = 1.0
        else:
            raise ValueError(f"Unknown color in Bayer pattern: {color}")

    return r_mask, g_mask, b_mask


def interpolate_channel(raw, mask, kernel):
    """
    课堂公式形式的单通道插值：

        C = ((Mc * X) convolved with K) / (Mc convolved with K)

    raw 是原始 Bayer 图，mask 是某个颜色通道的采样位置。
    """
    # numerator 聚合邻域内已经真实采样到的该颜色像素。
    numerator = convolve(raw * mask, kernel, mode="mirror")

    # denominator 统计邻域内有多少个该颜色采样点，用于求平均。
    denominator = convolve(mask, kernel, mode="mirror")
    channel = numerator / np.maximum(denominator, 1e-8)
    return channel.astype(np.float32)


def demosaic_simple(raw, pattern="RGGB", kernel_size=3):
    """
    使用 mask + 卷积平均的简单 demosaicing。

    这不是高质量商业算法，但足够展示 Bayer 到 RGB 的基本过程。
    """
    r_mask, g_mask, b_mask = create_bayer_masks(raw.shape, pattern)

    # 全 1 卷积核表示在局部窗口里做均值插值。
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32)

    # 分别插值得到完整分辨率的 R/G/B 通道。
    r = interpolate_channel(raw, r_mask, kernel)
    g = interpolate_channel(raw, g_mask, kernel)
    b = interpolate_channel(raw, b_mask, kernel)

    rgb = np.stack([r, g, b], axis=2).astype(np.float32)
    return rgb
