# data_utils.py

import os
import re
import numpy as np
from scipy.io import loadmat


def list_mat_files(data_dir="data"):
    """
    返回数据目录中的所有 .mat 文件路径。
    """
    mat_files = []

    for filename in os.listdir(data_dir):
        if filename.lower().endswith(".mat"):
            mat_files.append(os.path.join(data_dir, filename))

    mat_files.sort()
    return mat_files


def load_mat_file(path):
    """
    加载单个 .mat 文件并返回 A、D、PC。

    文件中的字段映射：
        amplitudesX -> A（幅值图）
        distancesX  -> D（距离图）
        cloudX      -> PC（三维点云）
    """

    data = loadmat(path)

    A_key = None
    D_key = None
    PC_key = None

    for key in data.keys():
        if key.startswith("__"):
            continue

        if key.startswith("amplitudes"):
            A_key = key
        elif key.startswith("distances"):
            D_key = key
        elif key.startswith("cloud"):
            PC_key = key

    if A_key is None or D_key is None or PC_key is None:
        raise ValueError(
            f"Could not find amplitude, distance, or point cloud in {path}.\n"
            f"Available keys: {list(data.keys())}"
        )

    A = data[A_key]
    D = data[D_key]
    PC = data[PC_key]

    # 防止有些图像是 H x W x 1
    A = np.squeeze(A)
    D = np.squeeze(D)

    print(f"Loaded: {path}")
    print(f"  A  key: {A_key}, shape: {A.shape}, dtype: {A.dtype}")
    print(f"  D  key: {D_key}, shape: {D.shape}, dtype: {D.dtype}")
    print(f"  PC key: {PC_key}, shape: {PC.shape}, dtype: {PC.dtype}")

    check_data_shapes(A, D, PC)

    return A, D, PC


def load_example(example_id, data_dir="data"):
    """
    按编号加载示例数据。

    示例：
        A, D, PC = load_example(1)
    """

    filename = f"example{example_id}kinect.mat"
    path = os.path.join(data_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    return load_mat_file(path)


def check_data_shapes(A, D, PC):
    """
    检查 A、D、PC 的维度是否匹配。
    """

    if A.shape != D.shape:
        raise ValueError(f"A and D shapes do not match: {A.shape} vs {D.shape}")

    if PC.ndim != 3 or PC.shape[2] != 3:
        raise ValueError(f"PC should have shape H x W x 3, but got {PC.shape}")

    if A.shape != PC.shape[:2]:
        raise ValueError(f"A/D and PC shapes do not match: {A.shape} vs {PC.shape[:2]}")


def print_data_info(A, D, PC):
    """
    打印 A、D、PC 的统计信息与有效点信息。
    """

    print("\nData information:")
    print(f"A shape:  {A.shape}, dtype: {A.dtype}")
    print(f"D shape:  {D.shape}, dtype: {D.dtype}")
    print(f"PC shape: {PC.shape}, dtype: {PC.dtype}")

    print("\nAmplitude image:")
    print(f"  min:  {np.nanmin(A)}")
    print(f"  max:  {np.nanmax(A)}")
    print(f"  mean: {np.nanmean(A)}")

    print("\nDistance image:")
    print(f"  min:  {np.nanmin(D)}")
    print(f"  max:  {np.nanmax(D)}")
    print(f"  mean: {np.nanmean(D)}")

    x = PC[:, :, 0]
    y = PC[:, :, 1]
    z = PC[:, :, 2]

    print("\nPoint cloud:")
    print(f"  x min/max/mean: {np.nanmin(x)}, {np.nanmax(x)}, {np.nanmean(x)}")
    print(f"  y min/max/mean: {np.nanmin(y)}, {np.nanmax(y)}, {np.nanmean(y)}")
    print(f"  z min/max/mean: {np.nanmin(z)}, {np.nanmax(z)}, {np.nanmean(z)}")

    invalid_count = np.sum(z == 0)
    total_count = z.size

    print("\nInvalid points:")
    print(f"  z == 0: {invalid_count} / {total_count}")


def get_valid_point_mask(PC):
    """
    返回二维布尔掩码，表示点云中有效像素。

    True  = 有效三维点
    False = 无效点（z == 0）
    """

    return PC[:, :, 2] != 0


def get_valid_points(PC):
    """
    返回所有有效三维点，形状为 N x 3。
    """

    valid_mask = get_valid_point_mask(PC)
    points = PC[valid_mask]

    return points


def get_valid_points_with_indices(PC):
    """
    返回有效点及其在原图中的行列坐标。

    返回：
        points: N x 3
        rows:   N
        cols:   N

    作用：
        RANSAC 在 N x 3 点集上运行，
        后续需要把内点重新映射回二维掩码。
    """

    valid_mask = get_valid_point_mask(PC)

    rows, cols = np.where(valid_mask)
    points = PC[valid_mask]

    return points, rows, cols


def create_mask_from_indices(image_shape, rows, cols, selected_mask):
    """
    把选中的点索引还原成二维掩码。

    参数说明：
        image_shape: 通常为 PC.shape[:2]
        rows, cols: 有效点在原图中的像素坐标
        selected_mask: RANSAC 输出的布尔内点掩码
    """

    mask = np.zeros(image_shape, dtype=bool)

    mask[rows[selected_mask], cols[selected_mask]] = True

    return mask