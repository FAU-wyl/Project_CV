# ransac.py

import numpy as np


def fit_plane_from_points(p1, p2, p3):
    """
    根据三个三维点拟合平面。

    平面表达式：
        normal · x = d

    返回：
        normal: 形状为 (3,) 的单位法向量
        d: 平面常数项
    """

    v1 = p2 - p1
    v2 = p3 - p1

    normal = np.cross(v1, v2)
    norm = np.linalg.norm(normal)

    # 三个点共线，无法确定平面
    if norm < 1e-12:
        return None

    normal = normal / norm
    d = np.dot(normal, p1)

    return normal, d


def point_plane_distances(points, normal, d):
    """
    计算一组三维点到平面的距离。

    由于 normal 是单位向量，因此距离为：
        distance = |normal · point - d|
    """

    return np.abs(points @ normal - d)


def ransac_plane(points, threshold=0.01, max_iterations=1000):
    """
    使用 RANSAC 在点云中寻找主平面。

    参数：
        points: N x 3 点云数组
        threshold: 内点距离阈值
        max_iterations: RANSAC 最大迭代次数

    返回：
        best_plane: (normal, d)
        best_inliers: 形状为 (N,) 的布尔内点掩码
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape N x 3, got {points.shape}")

    n_points = points.shape[0]

    if n_points < 3:
        raise ValueError("Need at least 3 points to fit a plane.")

    best_plane = None
    best_inliers = np.zeros(n_points, dtype=bool)
    best_count = 0

    for _ in range(max_iterations):
        # 随机选三个点
        sample_indices = np.random.choice(n_points, 3, replace=False)
        p1, p2, p3 = points[sample_indices]

        result = fit_plane_from_points(p1, p2, p3)

        # 如果三点共线，跳过
        if result is None:
            continue

        normal, d = result

        distances = point_plane_distances(points, normal, d)

        inliers = distances < threshold
        count = np.sum(inliers)

        if count > best_count:
            best_count = count
            best_plane = (normal, d)
            best_inliers = inliers

            # 如果所有点都属于这个平面，可以提前结束
            if best_count == n_points:
                break

    if best_plane is None:
        raise RuntimeError("RANSAC failed to find a valid plane.")

    print("RANSAC result:")
    print("  best inliers:", best_count, "/", n_points)
    print("  inlier ratio:", best_count / n_points)

    return best_plane, best_inliers
