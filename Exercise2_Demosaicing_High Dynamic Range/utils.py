# utils.py

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import rawpy


def load_raw_cr3(path):
    """
    从 CR3 文件中读取可见区域的 RAW 传感器数据。

    返回值是二维 Bayer 阵列，还没有经过 demosaicing。
    """
    # rawpy 会解析相机 RAW 容器，这里只取可见像素区域，避开黑边/校准区。
    raw = rawpy.imread(path)
    array = np.array(raw.raw_image_visible).astype(np.float32)
    return array


def save_float_image_as_jpg(img_float, output_path, quality=98):
    """
    将 [0, 1] 范围内的 float RGB 图像保存成 JPG。

    算法内部尽量保持 float，只有导出图片时才转成 uint8。
    """
    # 防止增强或归一化后的数值越界，JPG 只能表达 0-255。
    img_float = np.clip(img_float, 0.0, 1.0)
    img_uint8 = (img_float * 255.0).astype(np.uint8)
    Image.fromarray(img_uint8).save(output_path, quality=quality)


def show_image(img, title):
    """显示一张 float RGB 图像，主要用于调试和课堂观察结果。"""
    plt.figure(figsize=(10, 6))
    plt.imshow(np.clip(img, 0.0, 1.0))
    plt.title(title)
    plt.axis("off")
    plt.show()


def percentile_normalize(data, low=0.01, high=99.99):
    """
    使用百分位数把数据归一化到 [0, 1]。

    这样可以忽略极少数过暗/过亮异常值，比直接用 min/max 更稳定。
    """
    # a 和 b 分别作为黑场和白场参考点。
    a = np.percentile(data, low)
    b = np.percentile(data, high)

    # 避免整张图几乎没有动态范围时除以 0。
    if abs(b - a) < 1e-8:
        normalized = np.zeros_like(data, dtype=np.float32)
    else:
        normalized = (data - a) / (b - a)

    normalized = np.clip(normalized, 0.0, 1.0)
    return normalized.astype(np.float32), a, b


def gamma_correction(data, gamma=0.3):
    """
    对归一化后的图像做 gamma 校正：
        y = x^gamma

    gamma 小于 1 时会提亮暗部。
    """
    data = np.clip(data, 0.0, 1.0)
    return np.power(data, gamma).astype(np.float32)


def apply_brightness_contrast(rgb, brightness=0.0, contrast=1.0):
    """
    在 [0, 1] 浮点图上做最终显示调整：亮度为加性偏移，对比度绕 0.5 缩放。
    """
    img = np.clip(rgb.astype(np.float32), 0.0, 1.0)
    img = img + brightness
    img = np.clip(img, 0.0, 1.0)
    img = (img - 0.5) * contrast + 0.5
    return np.clip(img, 0.0, 1.0).astype(np.float32)


def log_curve(data, alpha=10.0):
    """
    Exercise 3 中使用的非 gamma 亮度曲线：
        y = log(1 + alpha*x) / log(1 + alpha)

    log 曲线可以压缩高亮，同时保留暗部可见性。
    """
    data = np.clip(data, 0.0, 1.0)
    return (np.log1p(alpha * data) / np.log1p(alpha)).astype(np.float32)
