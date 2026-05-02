# mask_utils.py

import numpy as np
from scipy.ndimage import label, generate_binary_structure
from scipy import ndimage

def create_mask_from_indices(image_shape, rows, cols, selected_mask):
    """
    把一维内点掩码映射回二维图像掩码。

    参数：
        image_shape: 图像尺寸，通常为 PC.shape[:2]
        rows, cols: 有效点在图像中的坐标
        selected_mask: RANSAC 输出的布尔内点掩码

    返回：
        2D 布尔掩码，True 表示内点。
    """

    if len(rows) != len(cols):
        raise ValueError("rows and cols must have the same length.")

    if len(selected_mask) != len(rows):
        raise ValueError(
            f"selected_mask length {len(selected_mask)} does not match "
            f"rows/cols length {len(rows)}."
        )

    mask = np.zeros(image_shape, dtype=bool)
    mask[rows[selected_mask], cols[selected_mask]] = True

    return mask


def remove_small_true_components(mask, min_size=100):
    """
    删除二值掩码中过小的 True 连通域。

    该步骤用于去除孤立噪点。
    """

    structure = generate_binary_structure(2, 2)
    labeled_mask, num_components = label(mask, structure=structure)

    if num_components == 0:
        return np.zeros_like(mask, dtype=bool)

    cleaned = np.zeros_like(mask, dtype=bool)

    for component_id in range(1, num_components + 1):
        component = labeled_mask == component_id
        size = np.sum(component)

        if size >= min_size:
            cleaned[component] = True

    return cleaned


def fill_small_false_holes(mask, max_hole_size=300):
    """
    仅填充掩码内部较小的 False 空洞。

    说明：
        不会填充大空洞，因此箱体区域（较大 False 区域）可保留。

    参数：
        mask: 2D 布尔掩码，True=地面，False=非地面
        max_hole_size: 仅填充面积不超过该阈值、且不接触边界的空洞

    返回：
        填洞后的掩码。
    """

    false_mask = ~mask

    structure = generate_binary_structure(2, 2)
    labeled_false, num_components = label(false_mask, structure=structure)

    result = mask.copy()

    for component_id in range(1, num_components + 1):
        component = labeled_false == component_id
        size = np.sum(component)

        # If a false region touches the image border, it is probably background,
        # not an internal hole.
        touches_border = (
            np.any(component[0, :]) or
            np.any(component[-1, :]) or
            np.any(component[:, 0]) or
            np.any(component[:, -1])
        )

        # Fill only small internal holes.
        if (not touches_border) and size <= max_hole_size:
            result[component] = True

    return result


def filter_floor_mask(mask, min_floor_size=100, max_hole_size=300):
    """
    清理原始地面掩码，减少噪声并保持箱体区域。

    输入语义：
        True=地面，False=非地面

    步骤：
        1. 删除小型地面连通域；
        2. 填充小型内部空洞。
    """

    cleaned = mask.astype(bool)

    # 创建结构元素（圆形核）
    kernel = ndimage.generate_binary_structure(2, 2)
    kernel_size = 3

    # Closing先，后Opening
    dilated = ndimage.binary_dilation(cleaned, structure=kernel, iterations=kernel_size)
    closed = ndimage.binary_erosion(dilated, structure=kernel, iterations=kernel_size)
    eroded = ndimage.binary_erosion(closed, structure=kernel, iterations=kernel_size)
    cleaned = ndimage.binary_dilation(eroded, structure=kernel, iterations=kernel_size)

    cleaned = remove_small_true_components(
        cleaned,
        min_size=min_floor_size
    )

    cleaned = fill_small_false_holes(
        cleaned,
        max_hole_size=max_hole_size
    )

    return cleaned.astype(bool)


def get_non_floor_mask(floor_mask):
    """
    由地面掩码得到非地面掩码。

    输入：
        floor_mask: True=地面，False=非地面

    输出：
        non_floor_mask: True=非地面，False=地面
    """

    return ~floor_mask.astype(bool)


def get_points_from_mask(PC, mask):
    """
    根据二维掩码从点云中提取三维点。

    参数：
        PC: H x W x 3 点云
        mask: H x W 布尔掩码

    返回：
        points: N x 3 三维点
        rows, cols: 选中点对应的图像坐标
    """

    if PC.shape[:2] != mask.shape:
        raise ValueError(
            f"PC shape {PC.shape[:2]} and mask shape {mask.shape} do not match."
        )

    rows, cols = np.where(mask)
    points = PC[rows, cols, :]

    # Remove invalid measurements.
    valid = points[:, 2] != 0

    return points[valid], rows[valid], cols[valid]


def largest_connected_component(mask):
    """
    仅保留最大的 True 连通域。

    该函数用于提取箱顶的主要区域。
    """

    structure = generate_binary_structure(2, 2)
    labeled_mask, num_components = label(mask, structure=structure)

    if num_components == 0:
        return np.zeros_like(mask, dtype=bool)

    component_sizes = np.bincount(labeled_mask.ravel())

    # Label 0 is background, ignore it.
    component_sizes[0] = 0

    largest_label = np.argmax(component_sizes)

    return labeled_mask == largest_label


def remove_small_components(mask, min_size=100):
    """
    通用别名函数：保留面积不小于 min_size 的 True 连通域。
    """

    return remove_small_true_components(mask, min_size=min_size)