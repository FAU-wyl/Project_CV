
# geometry_utils.py

import numpy as np


def plane_distance(plane1, plane2):
    """
    计算两近平行平面之间的距离。

    平面格式：
        normal · x = d
        plane = (normal, d)

    说明：
        法向量在 ransac 中已单位化，这里仍会再归一化一次。
    """

    n1, d1 = plane1
    n2, d2 = plane2

    n1 = np.asarray(n1, dtype=float)
    n2 = np.asarray(n2, dtype=float)

    # Normalize again for safety
    n1 = n1 / np.linalg.norm(n1)
    n2 = n2 / np.linalg.norm(n2)

    # If normals point in opposite directions, flip the second plane
    if np.dot(n1, n2) < 0:
        n2 = -n2
        d2 = -d2

    height = abs(d1 - d2)

    return height


def get_box_top_points(PC, box_top_mask):
    """
    提取箱顶掩码对应的有效三维点。

    参数：
        PC: H x W x 3 点云
        box_top_mask: H x W 布尔掩码（True 表示箱顶）

    返回：
        points: N x 3
    """

    if PC.shape[:2] != box_top_mask.shape:
        raise ValueError(
            f"PC shape {PC.shape[:2]} and mask shape {box_top_mask.shape} do not match."
        )

    points = PC[box_top_mask]

    # Remove invalid points
    valid = points[:, 2] != 0
    points = points[valid]

    if points.shape[0] == 0:
        raise ValueError("No valid box top points found.")

    return points


def estimate_length_width_simple(PC, box_top_mask):
    """
    Calculate the difference between the maximum of X and y and minimum of x and y directly。

    disadvantage:
        If the box is rotated, the result may be overestimated
    """

    points = get_box_top_points(PC, box_top_mask)

    xs = points[:, 0]
    ys = points[:, 1]

    length = np.max(xs) - np.min(xs)
    width = np.max(ys) - np.min(ys)

    # Put larger one as length, smaller one as width
    length, width = sorted([length, width], reverse=True)

    return length, width

def estimate_length_width_pca(PC, box_top_mask):
    """
    1. Use PCA to get the top 2 directions of the box_top_mask, One represents the width direction, and the other represents the length direction.
    2. Project points onto the two dominant directions
    3. Then calculate the maximum spread of the points along these two directions(width and length).
    思路：
        1. 取箱顶三维点；
        2. 中心化；
        3. 用 SVD/PCA 求顶面两个主方向；
        4. 投影到主方向；
        5. 投影范围作为长宽。

    Advantage:
        Estimate the true dimensions of an object regardless of its orientation by aligning the measurement axes with the object's principal directions of spread rather than the fixed image coordinates.
    """

    points = get_box_top_points(PC, box_top_mask)

    # Center points
    centroid = np.mean(points, axis=0)
    centered = points - centroid

    # SVD:
    # Vt[0] = first principal direction
    # Vt[1] = second principal direction
    # Vt[2] = normal-like direction
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    dir1 = vt[0]
    dir2 = vt[1]

    # Project points onto the two dominant directions
    proj1 = centered @ dir1
    proj2 = centered @ dir2

    size1 = np.max(proj1) - np.min(proj1)
    size2 = np.max(proj2) - np.min(proj2)

    length, width = sorted([size1, size2], reverse=True)

    return length, width

def estimate_box_dimensions(
    PC,
    floor_model,
    box_model,
    box_top_mask,
    method="pca",
):
    """
    estimate the height length and width

    Parameters:：
        PC: H x W x 3 point cloud
        floor_model: (normal, d)
        box_model: (normal, d)
        box_top_mask:
        method: "pca" or "simple"

    返回：
        height, length, width
    """

    height = plane_distance(floor_model, box_model)

    if method == "pca":
        length, width = estimate_length_width_pca(PC, box_top_mask)
    elif method == "simple":
        length, width = estimate_length_width_simple(PC, box_top_mask)
    return height, length, width


def get_box_top_corners_pca(PC, box_top_mask):
    """
    基于 PCA 坐标估计箱顶四个三维角点。

    返回：
        corners: 4 x 3 近似角点坐标

    说明：
        该方法先在 PCA 坐标中构造外接矩形，再映射回三维空间，
        因此是近似结果。
    """

    points = get_box_top_points(PC, box_top_mask)

    centroid = np.mean(points, axis=0)
    centered = points - centroid

    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    dir1 = vt[0] # 长边方向向量
    dir2 = vt[1] # 宽边方向向量

    proj1 = centered @ dir1
    proj2 = centered @ dir2

    min1, max1 = np.min(proj1), np.max(proj1)
    min2, max2 = np.min(proj2), np.max(proj2)

    corners = np.array([
        centroid + min1 * dir1 + min2 * dir2,
        centroid + max1 * dir1 + min2 * dir2,
        centroid + max1 * dir1 + max2 * dir2,
        centroid + min1 * dir1 + max2 * dir2,
    ])

    return corners