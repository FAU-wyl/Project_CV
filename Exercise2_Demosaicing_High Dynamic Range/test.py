# test.py
# Bayer 排列检测辅助脚本。
#
# 思路：
# 1. 读取 numpy RAW 和相机 JPG。
# 2. 把 JPG 粗略映射到 RAW 坐标。
# 3. 在 JPG 中找出明显的红/绿/蓝区域。
# 4. 对四种 Bayer 排列分别统计这些区域中的 RAW 响应。
# 5. 得分最高的排列作为估计结果。

import numpy as np
import matplotlib.pyplot as plt

from config import EX1_NPY, EX1_JPG


def load_data(raw_path, jpg_path):
    """读取 Exercise 1 的 numpy RAW 和 JPG 参考图。"""
    raw = np.load(raw_path).astype(np.float32)
    jpg = plt.imread(jpg_path).astype(np.float32)

    # matplotlib 读 JPG 时可能返回 [0, 1]，这里统一到 [0, 255] 方便阈值判断。
    if jpg.max() <= 1.0:
        jpg *= 255.0

    return raw, jpg


def map_jpg_to_raw_coordinates(raw, jpg):
    """
    将 JPG 图像粗略映射到 RAW 坐标系。

    RAW 和 JPG 的分辨率相近但不完全一致，因此用比例缩放做近似匹配。
    """
    raw_h, raw_w = raw.shape
    jpg_h, jpg_w = jpg.shape[:2]

    # yy/xx 是 RAW 像素坐标，yy_jpg/xx_jpg 是对应的 JPG 采样坐标。
    yy, xx = np.indices(raw.shape)

    yy_jpg = np.clip((yy * jpg_h / raw_h).astype(int), 0, jpg_h - 1)
    xx_jpg = np.clip((xx * jpg_w / raw_w).astype(int), 0, jpg_w - 1)

    jpg_on_raw = jpg[yy_jpg, xx_jpg]

    return jpg_on_raw, yy, xx


def detect_color_regions(jpg_on_raw, yy, xx):
    """
    从映射到 RAW 坐标的 JPG 中检测红、绿、蓝笔区域。

    region 限制用于避开背景和桌面，只关注图中笔所在的大致区域。
    """
    R = jpg_on_raw[:, :, 0]
    G = jpg_on_raw[:, :, 1]
    B = jpg_on_raw[:, :, 2]

    # 只在经验上包含彩色笔的位置范围内做颜色阈值判断。
    region = (
        (yy > 1000) & (yy < 3000) &
        (xx > 1500) & (xx < 5200)
    )

    # 颜色区域通过“主通道明显大于另外两个通道”来检测。
    red_area = region & (R > 120) & (R > 1.35 * G) & (R > 1.35 * B)
    green_area = region & (G > 80) & (G > 1.20 * R) & (G > 1.20 * B)
    blue_area = region & (B > 80) & (B > 1.25 * R) & (B > 1.15 * G)

    return red_area, green_area, blue_area


def create_boolean_bayer_masks(shape, pattern):
    """
    创建 bool 类型的 Bayer mask。

    pattern 顺序：
        pattern[0] pattern[1]
        pattern[2] pattern[3]

    Example:
        GBRG means:
            G B
            R G
    """
    h, w = shape

    # bool mask 更适合后面直接和颜色区域 mask 做逻辑与。
    r_mask = np.zeros((h, w), dtype=bool)
    g_mask = np.zeros((h, w), dtype=bool)
    b_mask = np.zeros((h, w), dtype=bool)

    positions = [
        (0, 0, pattern[0]),
        (0, 1, pattern[1]),
        (1, 0, pattern[2]),
        (1, 1, pattern[3]),
    ]

    for row_start, col_start, color in positions:
        # 当前 2x2 相位对应的所有像素位置。
        m = np.zeros((h, w), dtype=bool)
        m[row_start::2, col_start::2] = True

        if color == "R":
            r_mask |= m
        elif color == "G":
            g_mask |= m
        elif color == "B":
            b_mask |= m
        else:
            raise ValueError(f"Unknown Bayer color: {color}")

    return r_mask, g_mask, b_mask


def mean_raw_value(raw, color_mask, area_mask):
    """计算某个颜色采样位置在指定图像区域内的 RAW 平均值。"""
    mask = color_mask & area_mask

    # 某个颜色区域检测失败时返回 nan，避免误算。
    if mask.sum() == 0:
        return np.nan

    return raw[mask].mean()


def evaluate_pattern(raw, pattern, red_area, green_area, blue_area):
    """
    评估一个 Bayer 排列是否合理。

    合理的排列应该满足：
        红色区域：R 响应大于 B 响应
        蓝色区域：B 响应大于 R 响应
        绿色区域：G 响应较高

    由于 RAW 中绿色通道通常更亮，红/蓝区域的证据更可靠。
    """
    r_mask, g_mask, b_mask = create_boolean_bayer_masks(raw.shape, pattern)

    # 分别统计红/绿/蓝区域内，三种 Bayer 颜色采样点的平均 RAW 值。
    red_R = mean_raw_value(raw, r_mask, red_area)
    red_G = mean_raw_value(raw, g_mask, red_area)
    red_B = mean_raw_value(raw, b_mask, red_area)

    green_R = mean_raw_value(raw, r_mask, green_area)
    green_G = mean_raw_value(raw, g_mask, green_area)
    green_B = mean_raw_value(raw, b_mask, green_area)

    blue_R = mean_raw_value(raw, r_mask, blue_area)
    blue_G = mean_raw_value(raw, g_mask, blue_area)
    blue_B = mean_raw_value(raw, b_mask, blue_area)

    # 分数越高，说明该排列越符合红区像红、蓝区像蓝、绿区像绿的预期。
    red_score = red_R - red_B
    blue_score = blue_B - blue_R
    green_score = green_G - max(green_R, green_B)

    total_score = red_score + blue_score + 0.5 * green_score

    return {
        "pattern": pattern,
        "red_R": red_R,
        "red_G": red_G,
        "red_B": red_B,
        "green_R": green_R,
        "green_G": green_G,
        "green_B": green_B,
        "blue_R": blue_R,
        "blue_G": blue_G,
        "blue_B": blue_B,
        "red_score": red_score,
        "blue_score": blue_score,
        "green_score": green_score,
        "total_score": total_score,
    }


def visualize_detected_regions(red_area, green_area, blue_area):
    """把自动检测到的红/绿/蓝区域可视化，方便检查阈值是否合理。"""
    mask_vis = np.zeros((*red_area.shape, 3), dtype=np.float32)

    # RGB 三个通道分别显示三类检测区域。
    mask_vis[:, :, 0] = red_area.astype(np.float32)
    mask_vis[:, :, 1] = green_area.astype(np.float32)
    mask_vis[:, :, 2] = blue_area.astype(np.float32)

    plt.figure(figsize=(10, 6))
    plt.imshow(mask_vis)
    plt.title("Automatically detected red / green / blue pen regions")
    plt.axis("off")
    plt.show()


def investigate_bayer_pattern_elegant(raw_path=EX1_NPY, jpg_path=EX1_JPG):
    """完整执行 Bayer 排列检测流程，并打印四种排列的排名。"""
    raw, jpg = load_data(raw_path, jpg_path)

    print("RAW shape:", raw.shape)
    print("JPG shape:", jpg.shape)
    print("RAW min:", raw.min())
    print("RAW max:", raw.max())

    jpg_on_raw, yy, xx = map_jpg_to_raw_coordinates(raw, jpg)

    # 从 JPG 参考图中自动提取彩色笔区域。
    red_area, green_area, blue_area = detect_color_regions(jpg_on_raw, yy, xx)

    print("Detected red pixels:", int(red_area.sum()))
    print("Detected green pixels:", int(green_area.sum()))
    print("Detected blue pixels:", int(blue_area.sum()))

    visualize_detected_regions(red_area, green_area, blue_area)

    patterns = ["RGGB", "GRBG", "GBRG", "BGGR"]
    results = []

    for pattern in patterns:
        # 对每一种 Bayer 排列计算颜色响应分数。
        result = evaluate_pattern(
            raw,
            pattern,
            red_area,
            green_area,
            blue_area
        )
        results.append(result)

        print("\nPattern:", pattern)
        print("Red area means:   R={:.2f}, G={:.2f}, B={:.2f}".format(
            result["red_R"], result["red_G"], result["red_B"]
        ))
        print("Green area means: R={:.2f}, G={:.2f}, B={:.2f}".format(
            result["green_R"], result["green_G"], result["green_B"]
        ))
        print("Blue area means:  R={:.2f}, G={:.2f}, B={:.2f}".format(
            result["blue_R"], result["blue_G"], result["blue_B"]
        ))
        print("red_score   = {:.2f}".format(result["red_score"]))
        print("blue_score  = {:.2f}".format(result["blue_score"]))
        print("green_score = {:.2f}".format(result["green_score"]))
        print("total_score = {:.2f}".format(result["total_score"]))

    print("\nRanking:")
    # 按总分从高到低排列，第一名作为最终估计。
    ranked = sorted(results, key=lambda x: x["total_score"], reverse=True)

    for result in ranked:
        print("{}: total_score = {:.2f}".format(
            result["pattern"],
            result["total_score"]
        ))

    best = ranked[0]

    print("\nBest Bayer pattern for IMG_9939.npy:", best["pattern"])
    print("Pattern layout:")
    print(best["pattern"][0], best["pattern"][1])
    print(best["pattern"][2], best["pattern"][3])

    return best["pattern"], ranked


if __name__ == "__main__":
    investigate_bayer_pattern_elegant()
